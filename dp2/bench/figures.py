"""
Paper figures from the saved predictions — the same files report.py reads — into
C.BENCH_DIR/<out>/, each as .pdf and .png:

  results    RR AUC-PR and balanced accuracy on held-out fields, mean ± std over seeds
  nights     the same objects cut to their first n nights, 95% field-bootstrap band
  learning   learning curve, seed 0
  confusion  row-normalised confusion matrices of the seed-averaged probabilities
  umap       held-out representations of fold UMAP_FOLD, by class and by sky field

Colours follow the entity, never its rank: a model keeps its colour in every figure
whichever models are drawn.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix

import config as C
from bench import degrade, protocol, run

# validated categorical palette, fixed order (dataviz reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
          "#e34948"]
MODEL_COLOURS = dict(zip(("hgb", "vae", "transformer", "hgb_classic", "hgb_dmdt", "vae_hgb",
                          "mlp_dmdt"), SERIES))
LABELS = {"hgb": "ML: HGB", "vae": "Generative: SS-VAE", "transformer": "DL: transformer",
          "hgb_classic": "HGB, classical only", "hgb_dmdt": "HGB, VAE input",
          "vae_hgb": "SS-VAE + HGB head", "mlp_dmdt": "VAE encoder, no ELBO"}
INK, INK_2, GRID, OTHER = "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"
# fields, largest first: the largest in neutral grey, the rest in palette slots the classes
# (slots 1-5) do not use, so that no field shares a colour with a class
FIELD_COLOURS = ["#c9c8c3", SERIES[7], SERIES[6], SERIES[5], INK, OTHER]
BLUES = LinearSegmentedColormap.from_list(
    "blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
UMAP_FOLD = 2      # the most fields among the non-bulge folds, and the most RR
N_BOOT = 1000

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.bbox": "tight",
    "lines.linewidth": 2, "lines.markersize": 5})


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


def _scores(y, p, rr):
    return average_precision_score(y == rr, p[:, rr]), balanced_accuracy_score(y, p.argmax(1))


def _boot_ap(pos, s, fields, rng):
    """95% field-bootstrap interval of the AUC-PR of one model."""
    uf = np.unique(fields)
    idx = {f: np.where(fields == f)[0] for f in uf}
    vals = []
    for _ in range(N_BOOT):
        r = np.concatenate([idx[f] for f in rng.choice(uf, len(uf))])
        if pos[r].any():
            vals.append(average_precision_score(pos[r], s[r]))
    return np.percentile(vals, [2.5, 97.5])


def _results(names, oof, d, rows, rr, out):
    fig, axes = plt.subplots(1, 2, figsize=(7, 0.45 * len(names) + 0.8), sharey=True)
    per = {n: np.array([_scores(d.y[rows], p[rows], rr) for p, _ in o.values()])
           for n, o in oof.items()}
    ypos = np.arange(len(names))[::-1]
    for ax, j, title in zip(axes, (0, 1), ("RR AUC-PR", "Balanced accuracy (5 classes)")):
        for yp, n in zip(ypos, names):
            ax.errorbar(per[n][:, j].mean(), yp, xerr=per[n][:, j].std(), fmt="o",
                        color=MODEL_COLOURS[n], ecolor=MODEL_COLOURS[n], capsize=0)
            ax.annotate(f"{per[n][:, j].mean():.3f}", (per[n][:, j].mean(), yp),
                        textcoords="offset points", xytext=(0, 7), ha="center", color=INK_2)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, [LABELS[n] for n in names])
    axes[0].set_ylim(-0.6, len(names) - 0.4)
    _save(fig, out, "results")


def _nights(names, mean, d, rr, out):
    _, deg_rows, deg_n = degrade.load()
    base = np.unique(deg_rows)
    levels = sorted(C.DEG_LEVELS)
    rng = np.random.default_rng(C.RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(levels) + 1)
    for n in names:
        p, q = mean[n]
        sets = [(d.y[deg_rows[deg_n == lev]], q[deg_n == lev], d.fields[deg_rows[deg_n == lev]])
                for lev in levels] + [(d.y[base], p[base], d.fields[base])]
        ap = [average_precision_score(y == rr, s[:, rr]) for y, s, _ in sets]
        band = np.array([_boot_ap(y == rr, s[:, rr], f, rng) for y, s, f in sets])
        ax.fill_between(x, band[:, 0], band[:, 1], color=MODEL_COLOURS[n], alpha=0.15, lw=0)
        ax.plot(x, ap, "o-", color=MODEL_COLOURS[n], label=LABELS[n])
    chance = (d.y[base] == rr).mean()
    ax.axhline(chance, color=INK_2, lw=1, ls="--")
    ax.text(0, chance, " chance", va="bottom", color=INK_2)
    ax.set_xticks(x, [str(lev) for lev in levels] + ["all"])
    ax.set_xlabel(f"first n observing nights (objects with ≥ {C.DEG_MIN_NIGHTS} nights)")
    ax.set_ylabel("RR AUC-PR")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out, "nights")


def _learning(names, d, rows, rr, out):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for n in names:
        lc = run.load_lc(n, d)
        f = sorted(lc)
        sc = np.array([_scores(d.y[rows], lc[k][rows], rr) for k in f])
        for ax, j in zip(axes, (0, 1)):
            ax.plot(np.array(f) * 100, sc[:, j], "o-", color=MODEL_COLOURS[n], label=LABELS[n])
    for ax, title in zip(axes, ("RR AUC-PR", "Balanced accuracy (5 classes)")):
        ax.set_title(title, loc="left")
        ax.set_xlabel("training rows used (%)")
        ax.set_xticks([25, 50, 100])
    axes[0].legend(frameon=False)
    _save(fig, out, "learning")


def _confusion(names, mean, d, rows, out):
    fig, axes = plt.subplots(1, len(names), figsize=(2.4 * len(names), 2.5), squeeze=False,
                             sharey=True)
    k = len(d.cls)
    for ax, n in zip(axes[0], names):
        cm = confusion_matrix(d.y[rows], mean[n][0][rows].argmax(1), labels=range(k),
                              normalize="true")
        ax.imshow(cm, cmap=BLUES, vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 0.55 else INK)
        ax.set_xticks(range(k), d.cls, rotation=45)
        ax.set_yticks(range(k), d.cls)
        ax.tick_params(length=0)
        ax.set_xlabel("predicted")
        ax.set_title(LABELS[n], loc="left")
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0][0].set_ylabel("true (Gaia DR3)")
    _save(fig, out, "confusion")


def _umap(names, d, out):
    import umap
    have = [n for n in names
            if "emb" in np.load(os.path.join(protocol.model_dir(n), f"oof_s0_f{UMAP_FOLD}.npz"))]
    if not have:
        return
    fig, axes = plt.subplots(len(have), 2, figsize=(7, 3.2 * len(have)), squeeze=False)
    for row, n in zip(axes, have):
        f = np.load(os.path.join(protocol.model_dir(n), f"oof_s0_f{UMAP_FOLD}.npz"))
        z = umap.UMAP(random_state=C.RANDOM_SEED).fit_transform(f["emb"])
        y, fields = d.y[f["rows"]], d.fields[f["rows"]]
        uf, size = np.unique(fields, return_counts=True)
        uf = uf[np.argsort(-size)]
        groups = [(c, y == i, SERIES[i]) for i, c in enumerate(d.cls)]
        fgroups = [(f"field {int(v)}", fields == v, FIELD_COLOURS[min(i, len(FIELD_COLOURS) - 1)])
                   for i, v in enumerate(uf)]
        for ax, gs, title in zip(row, (groups, fgroups), ("by class", "by sky field")):
            for lab, m, col in sorted(gs, key=lambda g: -g[1].sum()):   # small groups on top
                ax.scatter(z[m, 0], z[m, 1], s=3, color=col, label=lab, lw=0, alpha=0.8)
            ax.set_title(f"{LABELS[n]}, fold {UMAP_FOLD} held out — {title}", loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.legend(frameon=False, markerscale=3, fontsize=6.5, loc="best")
    _save(fig, out, "umap")


def figures(names, out="figures"):
    out = os.path.join(C.BENCH_DIR, out)
    C.ensure_dirs(out)
    d = protocol.load()
    rows = np.where(d.lab)[0]
    rr = d.cls.index(C.POSITIVE_CLASS)
    n_deg = len(degrade.load()[1])
    oof = {n: run.load_oof(n, d, n_deg) for n in names}
    mean = {n: (np.mean([p for p, _ in o.values()], 0), np.mean([q for _, q in o.values()], 0))
            for n, o in oof.items()}
    _results(names, oof, d, rows, rr, out)
    _nights(names, mean, d, rr, out)
    _learning(names, d, rows, rr, out)
    _confusion(names, mean, d, rows, out)
    _umap(names, d, out)
    print(f"  figures → {out}")
