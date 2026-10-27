"""
STEP 5a — Binary classifier: RR Lyrae vs rest.

Input:  latent/ssvae_latent_vectors.npy + ssvae_latent_index.csv  (all training objects)
        latent/ssvae_labeled.npy + ssvae_labeled_index.csv        (labeled subset)
Output: results/rr_predictions.csv   diaObjectId, prob_rr, pred_rr
        results/rr_eval.txt
        results/rr_umap.png
        latent/rr_binary.pt
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, classification_report)

BASE        = os.path.dirname(os.path.abspath(__file__))
LATENT_DIR  = os.path.join(BASE, "latent")
RESULTS_DIR = os.path.join(BASE, "results")
CATALOG_CSV = os.path.join(BASE, "crossmatch", "catalogue_labeled.csv")

K_FOLDS     = 5
RANDOM_SEED = 42
LATENT_DIM  = 32
HIDDEN      = [128, 64]
DROPOUT     = 0.3
LR          = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS      = 400
PATIENCE    = 50
BATCH_SIZE  = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_data():
    # All training objects
    Z_all   = np.load(os.path.join(LATENT_DIR, "ssvae_latent_vectors.npy"))
    idx_all = pd.read_csv(os.path.join(LATENT_DIR, "ssvae_latent_index.csv"),
                          dtype={"diaObjectId": str})
    all_ids = list(idx_all["diaObjectId"])

    # Labeled subset with class labels
    idx_lab = pd.read_csv(os.path.join(LATENT_DIR, "ssvae_labeled_index.csv"),
                          dtype={"diaObjectId": str})
    Z_lab   = np.load(os.path.join(LATENT_DIR, "ssvae_labeled.npy"))
    lab_ids = list(idx_lab["diaObjectId"])
    labels  = list(idx_lab["label"])

    return Z_all, all_ids, Z_lab, lab_ids, labels


def train_model(X_tr, y_tr, seed=RANDOM_SEED):
    torch.manual_seed(seed)
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True)

    model     = MLP(LATENT_DIM, HIDDEN, DROPOUT).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_loss, patience_cnt, best_state = float("inf"), 0, None
    for epoch in range(1, EPOCHS + 1):
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
def predict(model, Z):
    model.eval()
    return torch.sigmoid(
        model(torch.tensor(Z, dtype=torch.float32).to(DEVICE))
    ).cpu().numpy()


def calibrate_threshold(probs, y):
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1     = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best_i = np.argmax(f1[:-1])
    return float(thresholds[best_i]), precision, recall, thresholds, f1, best_i


def plot_umap(Z_all, all_ids, prob_all, lab_ids, labels, threshold, out_path):
    import umap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("UMAP fit...")
    emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                    random_state=RANDOM_SEED, verbose=False).fit_transform(Z_all)

    id_to_idx = {oid: i for i, oid in enumerate(all_ids)}
    lab_map   = dict(zip(lab_ids, labels))

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    ax = axes[0]
    sc = ax.scatter(emb[:, 0], emb[:, 1], s=2, c=prob_all, cmap="Reds",
                    alpha=0.7, rasterized=True, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="prob_RR")
    ax.set_title("RR Lyrae probability", fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    ax = axes[1]
    pred = (prob_all >= threshold).astype(int)
    ax.scatter(emb[pred == 0, 0], emb[pred == 0, 1],
               s=1, c="#cccccc", alpha=0.3, rasterized=True,
               label=f"non-RR ({(pred==0).sum()})")
    ax.scatter(emb[pred == 1, 0], emb[pred == 1, 1],
               s=4, c="#e63946", alpha=0.8, rasterized=True,
               label=f"RR pred ({(pred==1).sum()})")
    known_rr = [id_to_idx[o] for o in lab_ids if lab_map.get(o) == "RR" and o in id_to_idx]
    if known_rr:
        ax.scatter(emb[known_rr, 0], emb[known_rr, 1],
                   s=40, c="gold", edgecolors="k", linewidths=0.5, zorder=6,
                   label=f"RR Gaia ({len(known_rr)})")
    ax.set_title(f"Binary predictions (thr={threshold:.3f})", fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=8, markerscale=2)

    fig.suptitle("STEP 5a — RR Lyrae binary classifier", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"UMAP → {out_path}")


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== STEP 5a — RR Binary Classifier (5-fold CV) ===\n")
    print(f"Device: {DEVICE}\n")

    Z_all, all_ids, Z_lab, lab_ids, labels = load_data()
    y_lab = (np.array(labels) == "RR").astype(np.float32)
    n_rr  = int(y_lab.sum())
    print(f"All objects  : {len(all_ids):,}")
    print(f"Labeled      : {len(lab_ids)}  (RR={n_rr}, non-RR={len(lab_ids)-n_rr})\n")

    # 5-fold CV
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_aucroc, fold_aucpr = [], []
    oof_probs = np.zeros(len(lab_ids))

    for fold, (tr, va) in enumerate(skf.split(Z_lab, y_lab), 1):
        print(f"Fold {fold}/{K_FOLDS}  train={len(tr)} (RR={int(y_lab[tr].sum())})  "
              f"val={len(va)} (RR={int(y_lab[va].sum())})")
        model         = train_model(Z_lab[tr], y_lab[tr], seed=RANDOM_SEED + fold)
        probs         = predict(model, Z_lab[va])
        oof_probs[va] = probs
        auc_roc = roc_auc_score(y_lab[va], probs)
        auc_pr  = average_precision_score(y_lab[va], probs)
        fold_aucroc.append(auc_roc)
        fold_aucpr.append(auc_pr)
        print(f"  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}")

    mean_roc = np.mean(fold_aucroc); std_roc = np.std(fold_aucroc)
    mean_pr  = np.mean(fold_aucpr);  std_pr  = np.std(fold_aucpr)
    print(f"\nCV AUC-ROC: {mean_roc:.4f} ± {std_roc:.4f}")
    print(f"CV AUC-PR:  {mean_pr:.4f} ± {std_pr:.4f}")

    best_thr, precision, recall, thresholds, f1, best_i = calibrate_threshold(oof_probs, y_lab)
    print(f"\nCalibrated threshold (max F1): {best_thr:.3f}"
          f"  P={precision[best_i]:.3f}  R={recall[best_i]:.3f}  F1={f1[best_i]:.3f}")

    # Final model on all labeled
    print(f"\nFinal model on all {len(lab_ids)} labeled objects...")
    final_model = train_model(Z_lab, y_lab, seed=RANDOM_SEED)
    torch.save(final_model.state_dict(), os.path.join(LATENT_DIR, "rr_binary.pt"))
    np.save(os.path.join(LATENT_DIR, "rr_threshold.npy"), np.array([best_thr]))

    # Inference on all objects
    prob_all = predict(final_model, Z_all)
    pred_all = (prob_all >= best_thr).astype(int)
    print(f"RR predicted (all objects, thr={best_thr:.3f}): {pred_all.sum()}")

    pd.DataFrame({
        "diaObjectId": all_ids,
        "prob_rr": prob_all,
        "pred_rr": pred_all,
    }).to_csv(os.path.join(RESULTS_DIR, "rr_predictions.csv"), index=False)

    # Log
    lines = [
        "STEP 5a — RR Binary Classifier", "=" * 50, "",
        f"Labeled: {len(lab_ids)}  (RR={n_rr}, non-RR={len(lab_ids)-n_rr})", "",
        "5-fold CV:"
    ]
    for i, (r, p) in enumerate(zip(fold_aucroc, fold_aucpr), 1):
        lines.append(f"  Fold {i}: AUC-ROC={r:.4f}  AUC-PR={p:.4f}")
    lines += [
        f"\nMean AUC-ROC: {mean_roc:.4f} ± {std_roc:.4f}",
        f"Mean AUC-PR:  {mean_pr:.4f} ± {std_pr:.4f}",
        f"\nThreshold (max F1): {best_thr:.3f}",
        f"  P={precision[best_i]:.3f}  R={recall[best_i]:.3f}  F1={f1[best_i]:.3f}", "",
        "OOF classification report:",
        classification_report(y_lab.astype(int), (oof_probs >= best_thr).astype(int),
                              target_names=["non-RR", "RR"], zero_division=0),
    ]
    with open(os.path.join(RESULTS_DIR, "rr_eval.txt"), "w") as f:
        f.write("\n".join(lines))

    plot_umap(Z_all, all_ids, prob_all, lab_ids, labels, best_thr,
              os.path.join(RESULTS_DIR, "rr_umap.png"))


if __name__ == "__main__":
    main()
