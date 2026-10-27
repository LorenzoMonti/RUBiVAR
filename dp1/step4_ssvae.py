"""
STEP 4 — Semi-Supervised VAE with SupCon loss + tabular features
=================================================================
Unsupervised VAE on all available light curves (lc_strict + labeled extras)
combined with Supervised Contrastive loss on 4-class labeled objects.

Tabular features (late fusion):
  log10_period  — Gaia period where available, LS period otherwise
  ls_power      — LS periodogram power (data-quality indicator)

Encoder:
  CNN(6x100) → 256D
  tab[log10_period, ls_power] → FC(2→32) → 32D
  concat(256+32) → mu(32D), logvar(32D)

Data sources (read from periods_final.csv):
  unlabeled  → folded_strict/        (LS period)
  labeled RR/ECL/Ceph → folded_labeled/  (Gaia period, best quality)
  labeled LPV → folded_gentle_extra/ or folded_xmatch/ (LS period, Gaia period as feature)

Output:
  latent/ssvae_latent_vectors.npy    — all training objects
  latent/ssvae_latent_index.csv
  latent/ssvae_labeled.npy           — labeled subset only
  latent/ssvae_labeled_index.csv
  latent/ssvae_best.pt
  latent/ssvae_tab_stats.npz
  latent/ssvae_umap_preview.png
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

BASE           = os.path.dirname(os.path.abspath(__file__))
LATENT_DIR     = os.path.join(BASE, "latent")
PERIODS_FINAL  = os.path.join(BASE, "periods", "periods_final.csv")
CATALOG_CSV    = os.path.join(BASE, "crossmatch", "catalogue_labeled.csv")
CLASSIFIER_CSV = os.path.join(BASE, "crossmatch", "catalogue_lsst_classifier.csv")

FOLDED_DIRS = {
    "folded_strict"       : os.path.join(BASE, "folded_strict"),
    "folded_gentle_extra" : os.path.join(BASE, "folded_gentle_extra"),
    "folded_xmatch"       : os.path.join(BASE, "folded_xmatch"),
    "folded_labeled"      : os.path.join(BASE, "folded_labeled"),
}

LATENT_DIM    = 32
TAB_DIM       = 2
BATCH_SIZE    = 256
EPOCHS        = 200
LR            = 1e-3
BETA_MAX      = 0.1
BETA_WARMUP   = 50
FREE_BITS     = 0.5
LAMBDA_MAX    = 1.0
LAMBDA_WARMUP = 80
TEMP          = 0.07

# All classes used for SupCon — more classes → more repulsion → better separation.
# Step5 then filters to only the 3 target classes (RR, ECL, LPV).
CLASSES      = ["AGN", "CV", "DSCT|GDOR|SXPHE", "ECL", "LPV",
                "RR",  "RS", "S",  "SOLAR_LIKE",  "YSO"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES    = len(CLASSES)
SCORE_THR    = 0.5   # min classifier score to include in SupCon

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data loading ──────────────────────────────────────────────────────────────

def build_training_index():
    """
    Returns a list of (fold_path, diaObjectId) for all training objects.
    Labeled objects use their best available fold (Gaia > LS).
    Unlabeled objects use folded_strict.
    No object appears twice.
    """
    periods = pd.read_csv(PERIODS_FINAL, dtype={"diaObjectId": str})
    index   = []
    seen    = set()
    for _, row in periods.iterrows():
        oid  = row["diaObjectId"]
        fdir = row["fold_dir"]
        if pd.isna(fdir) or fdir not in FOLDED_DIRS:
            continue
        path = os.path.join(FOLDED_DIRS[fdir], f"folded_{oid}.npy")
        if not os.path.exists(path) or oid in seen:
            continue
        seen.add(oid)
        index.append((path, oid))
    return index


def load_tab_features():
    """
    Load period and ls_power from periods_strict.csv + periods_gentle_extra.csv.
    Normalisation computed on strict only (original approach).
    """
    strict_csv = os.path.join(BASE, "periods", "periods_strict.csv")
    extra_csv  = os.path.join(BASE, "periods", "periods_gentle_extra.csv")

    periods  = pd.read_csv(strict_csv, dtype={"diaObjectId": str})
    log_p    = np.log10(np.clip(periods["period_days"].values, 1e-3, None))
    ls_pow   = periods["ls_power"].values.astype(float)

    tab_mean = np.array([log_p.mean(), ls_pow.mean()], dtype=np.float32)
    tab_std  = np.array([log_p.std(),  ls_pow.std()],  dtype=np.float32)
    tab_std  = np.where(tab_std < 1e-6, 1.0, tab_std)

    tab_map = {}
    for oid, lp, lsp in zip(periods["diaObjectId"], log_p, ls_pow):
        tab_map[str(oid)] = (np.array([lp, lsp], dtype=np.float32) - tab_mean) / tab_std

    if os.path.exists(extra_csv):
        extra = pd.read_csv(extra_csv, dtype={"diaObjectId": str})
        lp_e  = np.log10(np.clip(extra["period_days"].values, 1e-3, None))
        lsp_e = extra["ls_power"].values.astype(float)
        for oid, lp, lsp in zip(extra["diaObjectId"], lp_e, lsp_e):
            tab_map[str(oid)] = (np.array([lp, lsp], dtype=np.float32) - tab_mean) / tab_std

    tab_fallback = np.zeros(2, dtype=np.float32)
    return tab_map, tab_fallback, tab_mean, tab_std


# ── Dataset ───────────────────────────────────────────────────────────────────

class FoldedDataset(Dataset):
    def __init__(self, index, label_map, tab_map, tab_fallback):
        self.index  = index
        self.labels = [label_map.get(oid, -1) for _, oid in index]
        self.tabs   = [tab_map.get(oid, tab_fallback) for _, oid in index]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, _ = self.index[idx]
        arr  = np.load(path)
        mask = ~np.isnan(arr)
        x    = torch.from_numpy(np.where(mask, arr, 0.0)).float()
        m    = torch.from_numpy(mask).float()
        tab  = torch.from_numpy(self.tabs[idx])
        return x, m, self.labels[idx], tab


# ── Model ─────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, ld, tab_dim=TAB_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 32, 5, padding=2), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2, stride=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=2, stride=2), nn.BatchNorm1d(128), nn.GELU(),
        )
        self.fc_cnn = nn.Sequential(nn.Linear(128 * 25, 256), nn.GELU())
        self.fc_tab = nn.Sequential(nn.Linear(tab_dim, 32), nn.GELU())
        self.fc_mu  = nn.Linear(256 + 32, ld)
        self.fc_lv  = nn.Linear(256 + 32, ld)

    def forward(self, x, tab):
        h_cnn = self.fc_cnn(self.conv(x).flatten(1))
        h_tab = self.fc_tab(tab)
        h     = torch.cat([h_cnn, h_tab], dim=1)
        return self.fc_mu(h), self.fc_lv(h)


class Decoder(nn.Module):
    def __init__(self, ld):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ld, 256), nn.GELU(),
            nn.Linear(256, 128 * 25), nn.GELU(),
        )
        self.dconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.ConvTranspose1d(64, 32, 5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.ConvTranspose1d(32, 6, 5, padding=2),
        )

    def forward(self, z):
        return self.dconv(self.fc(z).view(-1, 128, 25))


class SSVAE(nn.Module):
    def __init__(self, ld=LATENT_DIM, tab_dim=TAB_DIM):
        super().__init__()
        self.encoder = Encoder(ld, tab_dim)
        self.decoder = Decoder(ld)

    def reparameterize(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)

    def forward(self, x, tab):
        mu, lv = self.encoder(x, tab)
        return self.decoder(self.reparameterize(mu, lv)), mu, lv


# ── Losses ────────────────────────────────────────────────────────────────────

def vae_loss(recon, x, mask, mu, lv, beta):
    n_valid    = mask.sum().clamp(min=1)
    recon_loss = ((recon - x) ** 2 * mask).sum() / n_valid
    kl         = (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).clamp(min=FREE_BITS).sum(1).mean()
    return recon_loss + beta * kl, recon_loss, kl


def supcon_loss(features, labels, temp=TEMP):
    N = features.size(0)
    if N < 2:
        return torch.tensor(0.0, device=features.device)
    sim  = torch.mm(features, features.T).clamp(-1.0, 1.0) / temp
    eye  = torch.eye(N, device=features.device).bool()
    sim.masked_fill_(eye, float("-inf"))
    labels_row = labels.unsqueeze(0).expand(N, N)
    labels_col = labels.unsqueeze(1).expand(N, N)
    pos_mask   = (labels_row == labels_col) & ~eye
    has_pos    = pos_mask.any(1)
    if not has_pos.any():
        return torch.tensor(0.0, device=features.device)
    n_pos    = pos_mask.float().sum(1).clamp(min=1)
    log_prob = F.log_softmax(sim, dim=1)
    loss     = -(log_prob.masked_fill(~pos_mask, 0.0)).sum(1) / n_pos
    return loss[has_pos].mean()


def beta_schedule(epoch):
    return BETA_MAX * min(1.0, epoch / BETA_WARMUP)


def lambda_schedule(epoch):
    if epoch <= BETA_WARMUP:
        return 0.0
    return LAMBDA_MAX * min(1.0, (epoch - BETA_WARMUP) / LAMBDA_WARMUP)


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, labeled_x, labeled_tab, labeled_y, optimizer, epoch):
    model.train()
    beta   = beta_schedule(epoch)
    lam    = lambda_schedule(epoch)
    totals = dict(loss=0, recon=0, kl=0, sc=0, mu_abs=0)

    for x, mask, _, tab in loader:
        x, mask, tab = x.to(DEVICE), mask.to(DEVICE), tab.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, lv = model(x, tab)
        loss, rl, kl  = vae_loss(recon, x, mask, mu, lv, beta)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals["loss"]   += loss.item()
        totals["recon"]  += rl.item()
        totals["kl"]     += kl.item()
        totals["mu_abs"] += mu.abs().mean().item()

    if lam > 0 and labeled_x is not None:
        optimizer.zero_grad()
        mu_lab, _ = model.encoder(labeled_x, labeled_tab)
        z_norm    = F.normalize(mu_lab, dim=1)
        sc        = supcon_loss(z_norm, labeled_y)
        (lam * sc).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals["sc"] = sc.item()

    nb = len(loader)
    return {k: v / nb for k, v in totals.items()}, beta, lam


# ── Latent extraction ─────────────────────────────────────────────────────────

class _InferenceDataset(Dataset):
    def __init__(self, index, tabs):
        self.index = index
        self.tabs  = tabs

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, _ = self.index[idx]
        arr = np.load(path)
        x   = torch.from_numpy(np.where(~np.isnan(arr), arr, 0.0)).float()
        tab = torch.from_numpy(self.tabs[idx])
        return x, tab


@torch.no_grad()
def extract_latents(model, index, tab_map, tab_fallback):
    model.eval()
    tabs = [tab_map.get(oid, tab_fallback) for _, oid in index]
    ds     = _InferenceDataset(index, tabs)
    loader = DataLoader(ds, batch_size=512, num_workers=4, pin_memory=True)
    vecs   = []
    for x, tab in tqdm(loader, desc="Encoding", leave=False):
        mu, _ = model.encoder(x.to(DEVICE), tab.to(DEVICE))
        vecs.append(mu.cpu().numpy())
    return np.vstack(vecs)


# ── UMAP ──────────────────────────────────────────────────────────────────────

def umap_preview(all_index, all_latents, label_map, out_path):
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    print("UMAP fit...")
    emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                    random_state=42).fit_transform(all_latents)

    # Target classes: large, prominent markers
    TARGET = {"RR": "#e63946", "ECL": "#2a9d8f", "LPV": "#e9c46a"}
    # Anchor classes: small markers — used for SupCon only, not reported in step5
    ANCHOR = {"AGN": "#8338ec", "SOLAR_LIKE": "#118ab2", "YSO": "#06d6a0",
              "RS": "#ffd166", "S": "#ff9f1c", "CV": "#ff006e",
              "DSCT|GDOR|SXPHE": "#cbf3f0"}

    fig, ax = plt.subplots(figsize=(12, 9))

    # Unlabeled: small grey dots
    unlabeled_mask = np.array([label_map.get(oid, -1) == -1 for _, oid in all_index])
    ax.scatter(emb[unlabeled_mask, 0], emb[unlabeled_mask, 1],
               s=1, c="#cccccc", alpha=0.3, rasterized=True, label="unlabeled")

    # Anchor classes: small dots, plotted first (background)
    for cls, color in ANCHOR.items():
        if cls not in CLASS_TO_IDX:
            continue
        idx = CLASS_TO_IDX[cls]
        mask = np.array([label_map.get(oid, -1) == idx for _, oid in all_index])
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], s=12, c=color, alpha=0.6,
                   marker="o", zorder=3, edgecolors="none",
                   label=f"{cls} ({mask.sum()})")

    # Target classes: larger markers, foreground
    for cls, color in TARGET.items():
        idx = CLASS_TO_IDX[cls]
        mask = np.array([label_map.get(oid, -1) == idx for _, oid in all_index])
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], s=35, c=color, alpha=0.9,
                   marker="*", zorder=6, edgecolors="k", linewidths=0.3,
                   label=f"{cls} ★ ({mask.sum()})")

    ax.legend(loc="upper right", fontsize=7, markerscale=1.2, framealpha=0.8,
              title="● anchor class  ★ target class")
    ax.set_title("UMAP — SS-VAE + SupCon (10 classes for SupCon, 3 target ★)\n"
                 "Gaia period for RR/ECL/LPV, LS period for anchor classes", fontsize=11)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"UMAP → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    os.makedirs(LATENT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Period table + tabular features
    tab_map, tab_fallback, tab_mean, tab_std = load_tab_features()
    np.savez(os.path.join(LATENT_DIR, "ssvae_tab_stats.npz"), mean=tab_mean, std=tab_std)
    print(f"Tab mean: log10_period={tab_mean[0]:.3f}  ls_power={tab_mean[1]:.3f}")
    print(f"Tab std:  log10_period={tab_std[0]:.3f}  ls_power={tab_std[1]:.3f}")

    # Labels — classifier only (score≥0.5), all 10 classes.
    # This matches the original configuration that gave inter-centroid=2.178.
    clf = pd.read_csv(CLASSIFIER_CSV, dtype={"lsst_id": str})
    label_map = {oid: CLASS_TO_IDX[cls]
                 for oid, cls, sc in zip(clf["lsst_id"], clf["best_class_name"], clf["best_class_score"])
                 if sc >= SCORE_THR and cls in CLASS_TO_IDX}

    # Training index — strict + gentle_extra (original configuration)
    def _oid(f): return os.path.basename(f).replace("folded_", "").replace(".npy", "")
    files_strict = sorted(glob.glob(os.path.join(FOLDED_DIRS["folded_strict"], "folded_*.npy")))
    files_extra  = sorted(glob.glob(os.path.join(FOLDED_DIRS["folded_gentle_extra"], "folded_*.npy")))
    train_index  = [(f, _oid(f)) for f in files_strict + files_extra]

    n_labeled   = sum(1 for _, oid in train_index if oid in label_map)
    n_no_period = sum(1 for _, oid in train_index if oid not in tab_map)
    print(f"\nTraining objects: {len(train_index):,}  "
          f"(strict={len(files_strict)}, extra={len(files_extra)})")
    print(f"  Labeled : {n_labeled}")
    print(f"  Classes : {CLASSES}\n")

    dataset = FoldedDataset(train_index, label_map, tab_map, tab_fallback)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=False)

    # Pre-load labeled samples on GPU for SupCon pass
    lab_index = [(p, oid) for p, oid in train_index if oid in label_map]
    lab_y     = torch.tensor([label_map[oid] for _, oid in lab_index], device=DEVICE)
    lab_arrs, lab_tabs = [], []
    for path, oid in lab_index:
        arr = np.load(path)
        lab_arrs.append(torch.from_numpy(np.where(~np.isnan(arr), arr, 0.0)).float())
        lab_tabs.append(torch.from_numpy(tab_map.get(oid, tab_fallback)))
    labeled_x   = torch.stack(lab_arrs).to(DEVICE)
    labeled_tab = torch.stack(lab_tabs).to(DEVICE)

    cls_counts = dict(zip(*np.unique(lab_y.cpu().numpy(), return_counts=True)))
    cls_named  = {CLASSES[k]: v for k, v in cls_counts.items()}
    print(f"Labeled batch: {labeled_x.shape}")
    print("  " + "  ".join(f"{c}={n}" for c, n in cls_named.items()) + "\n")

    model     = SSVAE().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}  "
          f"Epochs: {EPOCHS}  β_max={BETA_MAX}  λ_max={LAMBDA_MAX}  τ={TEMP}\n")

    best_recon = float("inf")
    ckpt       = os.path.join(LATENT_DIR, "ssvae_best.pt")

    for epoch in range(1, EPOCHS + 1):
        m, beta, lam = train_epoch(model, loader, labeled_x, labeled_tab,
                                   lab_y, optimizer, epoch)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"recon={m['recon']:.4f}  KL={m['kl']:.4f}  "
                  f"SC={m['sc']:.4f}  |μ|={m['mu_abs']:.3f}  "
                  f"β={beta:.3f}  λ={lam:.3f}")
        if m["recon"] < best_recon:
            best_recon = m["recon"]
            torch.save(model.state_dict(), ckpt)

    print(f"\nBest recon: {best_recon:.4f}  → {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    # Extract and save all latents
    all_latents = extract_latents(model, train_index, tab_map, tab_fallback)
    all_oids    = [oid for _, oid in train_index]
    np.save(os.path.join(LATENT_DIR, "ssvae_latent_vectors.npy"), all_latents)
    pd.DataFrame({"diaObjectId": all_oids, "idx": range(len(all_oids))}) \
      .to_csv(os.path.join(LATENT_DIR, "ssvae_latent_index.csv"), index=False)
    print(f"Latents: {all_latents.shape}")

    # Labeled subset for step5 — use catalogue_labeled (all 325 objects: RR/ECL/LPV/Ceph)
    # This is broader than step4's SupCon set (322 classifier objects) and gives step5
    # more labeled data to train on.
    catalog   = pd.read_csv(CATALOG_CSV, dtype={"diaObjectId": str})
    oid_to_idx_all = {oid: i for i, oid in enumerate(all_oids)}
    lab_rows  = catalog[catalog["diaObjectId"].isin(oid_to_idx_all)].copy()
    lab_idxs  = [oid_to_idx_all[o] for o in lab_rows["diaObjectId"]]
    lab_latents = all_latents[lab_idxs]
    np.save(os.path.join(LATENT_DIR, "ssvae_labeled.npy"), lab_latents)
    pd.DataFrame({"diaObjectId": list(lab_rows["diaObjectId"]),
                  "label":       list(lab_rows["label"]),
                  "idx":         range(len(lab_rows))}) \
      .to_csv(os.path.join(LATENT_DIR, "ssvae_labeled_index.csv"), index=False)
    print(f"Labeled latents (catalogue_labeled): {lab_latents.shape}  "
          f"— {lab_rows['label'].value_counts().to_dict()}")

    # UMAP
    oid_to_idx = {oid: i for i, oid in enumerate(all_oids)}
    umap_preview(train_index, all_latents, label_map,
                 os.path.join(LATENT_DIR, "ssvae_umap_preview.png"))

    # Inter-centroid diagnostic
    from scipy.spatial.distance import cdist as scipy_cdist
    print("\nCentroid separation (labeled objects):")
    centroids = {}
    for oid, lat in zip(lab_oids, lab_latents):
        centroids.setdefault(label_map[oid], []).append(lat)
    cents    = {c: np.mean(v, axis=0) for c, v in centroids.items()}
    cls_list = sorted(cents.keys())
    c_mat    = np.array([cents[c] for c in cls_list])
    dists    = scipy_cdist(c_mat, c_mat)
    print(f"  Min inter-centroid: {dists[dists > 0].min():.3f}")
    for c in cls_list:
        vecs  = np.array(centroids[c])
        intra = np.linalg.norm(vecs - cents[c], axis=1).mean()
        print(f"  {CLASSES[c]:6s}: n={len(vecs):3d}  intra-spread={intra:.3f}")


if __name__ == "__main__":
    main()
