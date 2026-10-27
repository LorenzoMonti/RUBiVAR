"""
STEP 5b — 3-class classifier (RR / ECL / LPV) + centroid-distance OOD detection.

The centroid distance flags objects whose latent embedding is far from all known
class clusters — likely AGN, SOLAR_LIKE, YSO, etc. that the model has never seen
as labeled examples.

Input:  latent/ssvae_latent_vectors.npy + ssvae_latent_index.csv
        latent/ssvae_labeled.npy + ssvae_labeled_index.csv
Output: results/clf_predictions.csv
          diaObjectId, prob_ECL, prob_LPV, prob_RR,
          pred_class, pred_prob,
          dist_ECL, dist_LPV, dist_RR, dist_nearest, is_ood
        results/clf_eval.txt
        results/clf_umap.png
        latent/clf_multiclass.pt
        latent/clf_centroids.npz
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, balanced_accuracy_score
from scipy.spatial.distance import cdist

BASE        = os.path.dirname(os.path.abspath(__file__))
LATENT_DIR  = os.path.join(BASE, "latent")
RESULTS_DIR = os.path.join(BASE, "results")

CLASSES      = ["ECL", "LPV", "RR"]
N_CLASSES    = len(CLASSES)
K_FOLDS      = 5
RANDOM_SEED  = 42
LATENT_DIM   = 32
HIDDEN       = [128, 64]
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 400
PATIENCE     = 50
BATCH_SIZE   = 64
OOD_SIGMA    = 2.0   # objects beyond mean + OOD_SIGMA * std of labeled dist_nearest → OOD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, dropout):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    Z_all   = np.load(os.path.join(LATENT_DIR, "ssvae_latent_vectors.npy"))
    idx_all = pd.read_csv(os.path.join(LATENT_DIR, "ssvae_latent_index.csv"),
                          dtype={"diaObjectId": str})
    Z_lab   = np.load(os.path.join(LATENT_DIR, "ssvae_labeled.npy"))
    idx_lab = pd.read_csv(os.path.join(LATENT_DIR, "ssvae_labeled_index.csv"),
                          dtype={"diaObjectId": str})

    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
    mask    = idx_lab["label"].isin(CLASSES)
    idx_lab = idx_lab[mask].reset_index(drop=True)
    Z_lab   = Z_lab[mask.values]
    y_lab   = np.array([cls_to_idx[l] for l in idx_lab["label"]])

    return (Z_all, list(idx_all["diaObjectId"]),
            Z_lab, list(idx_lab["diaObjectId"]), y_lab)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X_tr, y_tr, seed=RANDOM_SEED):
    torch.manual_seed(seed)
    counts  = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
    class_w = torch.tensor(counts.sum() / (N_CLASSES * counts + 1e-6),
                           dtype=torch.float32).to(DEVICE)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=True)

    model     = MLP(LATENT_DIM, HIDDEN, N_CLASSES, DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_loss, patience_cnt, best_state = float("inf"), 0, None
    for _ in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        sched.step()
        avg = total / len(X_tr)
        if avg < best_loss - 1e-4:
            best_loss, patience_cnt = avg, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
        if patience_cnt >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_proba(model, Z):
    model.eval()
    logits = model(torch.tensor(Z, dtype=torch.float32).to(DEVICE))
    return torch.softmax(logits, dim=1).cpu().numpy()


# ── Centroid distance ─────────────────────────────────────────────────────────

def compute_centroids(Z_lab, y_lab):
    return np.array([Z_lab[y_lab == i].mean(axis=0) for i in range(N_CLASSES)])


def centroid_distances(Z, centroids):
    return cdist(Z, centroids, metric="euclidean")


def ood_threshold(Z_lab, y_lab, centroids):
    dists   = centroid_distances(Z_lab, centroids)
    nearest = dists.min(axis=1)
    thr     = float(nearest.mean() + OOD_SIGMA * nearest.std())
    return thr, nearest


# ── UMAP ──────────────────────────────────────────────────────────────────────

def plot_umap(Z_all, all_ids, probs, dist_nearest, ood_thr, lab_ids, y_lab, out_path):
    import umap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {"ECL": "#2a9d8f", "LPV": "#e9c46a", "RR": "#e63946"}
    print("UMAP fit...")
    emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                    random_state=RANDOM_SEED, verbose=False).fit_transform(Z_all)

    id_to_idx = {oid: i for i, oid in enumerate(all_ids)}
    pred_cls  = probs.argmax(axis=1)
    is_ood    = dist_nearest > ood_thr

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Panel 1: predicted class + OOD
    ax = axes[0]
    ax.scatter(emb[is_ood, 0], emb[is_ood, 1],
               s=1, c="#cccccc", alpha=0.2, rasterized=True, label=f"OOD ({is_ood.sum()})")
    for i, cls in enumerate(CLASSES):
        m = (~is_ood) & (pred_cls == i)
        ax.scatter(emb[m, 0], emb[m, 1], s=2, c=COLORS[cls],
                   alpha=0.6, rasterized=True, label=f"{cls} ({m.sum()})")
    for i, cls in enumerate(CLASSES):
        pts = [id_to_idx[o] for o, yi in zip(lab_ids, y_lab)
               if yi == i and o in id_to_idx]
        if pts:
            ax.scatter(emb[pts, 0], emb[pts, 1], s=35, c=COLORS[cls],
                       marker="*", edgecolors="k", linewidths=0.3, zorder=6,
                       label=f"  ★ {cls} labeled ({len(pts)})")
    ax.set_title("3-class predictions + OOD (grey)", fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=7, markerscale=1.5)

    # Panel 2: centroid distance heatmap
    ax = axes[1]
    sc = ax.scatter(emb[:, 0], emb[:, 1], s=2, c=dist_nearest,
                    cmap="viridis_r", alpha=0.7, rasterized=True)
    plt.colorbar(sc, ax=ax, label="dist to nearest centroid")
    ax.set_title(f"Centroid distance  (OOD thr={ood_thr:.2f})", fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    fig.suptitle("STEP 5b — 3-class classifier (ECL / LPV / RR)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"UMAP → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== STEP 5b — 3-class Classifier + OOD Detection ===\n")
    print(f"Device: {DEVICE}\n")

    Z_all, all_ids, Z_lab, lab_ids, y_lab = load_data()
    print(f"All objects : {len(all_ids):,}")
    print(f"Labeled     : {len(lab_ids)}  —  " +
          "  ".join(f"{c}={int((y_lab==i).sum())}" for i, c in enumerate(CLASSES)))
    print()

    # 5-fold CV
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_probs = np.zeros((len(lab_ids), N_CLASSES))
    fold_bacc = []

    for fold, (tr, va) in enumerate(skf.split(Z_lab, y_lab), 1):
        print(f"Fold {fold}/{K_FOLDS}  train={len(tr)}  val={len(va)}")
        model         = train_model(Z_lab[tr], y_lab[tr], seed=RANDOM_SEED + fold)
        probs         = predict_proba(model, Z_lab[va])
        oof_probs[va] = probs
        bacc = balanced_accuracy_score(y_lab[va], probs.argmax(axis=1))
        fold_bacc.append(bacc)
        print(f"  Balanced acc: {bacc:.4f}")
        print(classification_report(y_lab[va], probs.argmax(axis=1),
                                    target_names=CLASSES, zero_division=0))

    mean_bacc = np.mean(fold_bacc)
    std_bacc  = np.std(fold_bacc)
    oof_pred  = oof_probs.argmax(axis=1)
    print(f"\nCV Balanced Acc: {mean_bacc:.4f} ± {std_bacc:.4f}")
    print("\nOOF report (all labeled):")
    print(classification_report(y_lab, oof_pred, target_names=CLASSES, zero_division=0))

    # Final model
    print(f"Final model on all {len(lab_ids)} labeled objects...")
    final_model = train_model(Z_lab, y_lab, seed=RANDOM_SEED)
    torch.save(final_model.state_dict(), os.path.join(LATENT_DIR, "clf_multiclass.pt"))

    # Centroids + OOD threshold
    centroids = compute_centroids(Z_lab, y_lab)
    np.savez(os.path.join(LATENT_DIR, "clf_centroids.npz"),
             centroids=centroids, classes=np.array(CLASSES))
    ood_thr, lab_nearest = ood_threshold(Z_lab, y_lab, centroids)
    print(f"\nOOD threshold ({OOD_SIGMA}σ): {ood_thr:.3f}  "
          f"(labeled nearest: mean={lab_nearest.mean():.3f}  std={lab_nearest.std():.3f})")

    # Inference
    probs_all = predict_proba(final_model, Z_all)
    dists_all = centroid_distances(Z_all, centroids)
    dist_near = dists_all.min(axis=1)
    is_ood    = dist_near > ood_thr
    pred_cls  = probs_all.argmax(axis=1)

    print(f"\nPredictions ({len(all_ids):,} objects):")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {int(((~is_ood) & (pred_cls == i)).sum())}")
    print(f"  OOD: {is_ood.sum()} ({100*is_ood.mean():.1f}%)")

    # Save
    pd.DataFrame({
        "diaObjectId" : all_ids,
        "prob_ECL"    : probs_all[:, CLASSES.index("ECL")],
        "prob_LPV"    : probs_all[:, CLASSES.index("LPV")],
        "prob_RR"     : probs_all[:, CLASSES.index("RR")],
        "pred_class"  : [CLASSES[i] for i in pred_cls],
        "pred_prob"   : probs_all.max(axis=1),
        "dist_ECL"    : dists_all[:, CLASSES.index("ECL")],
        "dist_LPV"    : dists_all[:, CLASSES.index("LPV")],
        "dist_RR"     : dists_all[:, CLASSES.index("RR")],
        "dist_nearest": dist_near,
        "is_ood"      : is_ood,
    }).to_csv(os.path.join(RESULTS_DIR, "clf_predictions.csv"), index=False)
    print("Saved → results/clf_predictions.csv")

    lines = [
        "STEP 5b — 3-class Classifier + OOD Detection", "=" * 55, "",
        f"Classes: {CLASSES}",
        "Labeled: " + "  ".join(f"{c}={int((y_lab==i).sum())}" for i, c in enumerate(CLASSES)), "",
        f"CV Balanced Accuracy: {mean_bacc:.4f} ± {std_bacc:.4f}", "",
        "OOF report:", classification_report(y_lab, oof_pred, target_names=CLASSES, zero_division=0),
        f"OOD threshold ({OOD_SIGMA}σ): {ood_thr:.3f}",
        f"  OOD: {is_ood.sum()} / {len(all_ids)} ({100*is_ood.mean():.1f}%)", "",
        "Predictions (non-OOD):",
    ] + [f"  {c}: {int(((~is_ood)&(pred_cls==i)).sum())}" for i, c in enumerate(CLASSES)]
    with open(os.path.join(RESULTS_DIR, "clf_eval.txt"), "w") as f:
        f.write("\n".join(lines))

    plot_umap(Z_all, all_ids, probs_all, dist_near, ood_thr, lab_ids, y_lab,
              os.path.join(RESULTS_DIR, "clf_umap.png"))


if __name__ == "__main__":
    main()
