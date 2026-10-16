'''
"""Part 2 -- score DBSCAN against the ground-truth labels.

DBSCAN never saw the labels. This script compares the clusters it invented
(labels.csv, written by cluster_dbscan.py on the TRAINING split) with the
labels recorded during collection, and writes:

  ../figures/generated/confusion_dbscan.csv    true label x discovered cluster
  ../figures/generated/eps_sweep.csv           clusters/noise/purity vs eps
  ../figures/generated/dbscan_vs_truth.pdf     side-by-side figure
  cluster_names.json                           cluster id -> lighting name
                                               (Part 3 uses it for CLASS_NAMES)

Run cluster_dbscan.py first. Run from inside src/.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from pipeline_common import load_split_scaled, VIEW_LIM

# Must match cluster_dbscan.py
EPS, MIN_SAMPLES = 0.03, 10
SWEEP = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15]

data = load_split_scaled()
X, truth = data["train"], data["y_train"]
labels = np.loadtxt("labels.csv", dtype=int)
assert len(labels) == len(X), "labels.csv is stale: re-run cluster_dbscan.py"

# ---- confusion table: rows = true label, cols = discovered cluster (-1 = noise)
conf = pd.crosstab(pd.Series(truth, name="true label"),
                   pd.Series(labels, name="DBSCAN cluster"))
print("Confusion table (training split):")
print(conf.to_string(), "\n")
conf.to_csv("../figures/generated/confusion_dbscan.csv")

# ---- name each cluster after its majority true label
names = {}
for c in sorted(set(labels) - {-1}):
    col = conf[c]
    names[int(c)] = str(col.idxmax())
    print(f"cluster {c} -> {col.idxmax():12s} purity {col.max() / col.sum():.1%} "
          f"({col.sum()} points)")
noise = int((labels == -1).sum())
print(f"noise: {noise} points ({noise / len(labels):.1%})")
if len(set(names.values())) < len(names):
    print("WARNING: two clusters map to the same label (a class was split).")
missing = set(truth) - set(names.values())
if missing:
    print(f"WARNING: no cluster for {missing} (classes were merged).")
with open("cluster_names.json", "w") as f:
    json.dump(names, f, indent=2)

# ---- eps sweep: how sensitive is the discovery?
rows = []
for e in SWEEP:
    lab = DBSCAN(eps=e, min_samples=MIN_SAMPLES).fit_predict(X)
    k = len(set(lab) - {-1})
    m = lab != -1
    c = pd.crosstab(lab[m], truth[m]) if m.any() else None
    purity = c.max(axis=1).sum() / m.sum() if m.any() else 0.0
    rows.append((e, k, float(np.mean(lab == -1)), float(purity)))
sweep = pd.DataFrame(rows, columns=["eps", "clusters", "noise_frac", "purity"])
print("\neps sweep (min_samples fixed at", MIN_SAMPLES, "):")
print(sweep.to_string(index=False))
sweep.to_csv("../figures/generated/eps_sweep.csv", index=False)

# ---- figure: ground truth vs DBSCAN, same points
MARKERS = ["o", "s", "^", "D", "v", "P"]
GRAYS = ["0.75", "0.45", "0.15", "0.6", "0.3", "0.85"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharex=True, sharey=True)
for i, t in enumerate(sorted(set(truth))):
    p = X[truth == t]
    axes[0].scatter(p[:, 0], p[:, 1], s=6, marker=MARKERS[i], facecolor=GRAYS[i],
                    edgecolor="black", linewidth=0.25, label=t)
for i, c in enumerate(sorted(set(labels) - {-1})):
    p = X[labels == c]
    axes[1].scatter(p[:, 0], p[:, 1], s=6, marker=MARKERS[i % 6], facecolor=GRAYS[i % 6],
                    edgecolor="black", linewidth=0.25,
                    label=f"cluster {c} ({names[int(c)]})")
if noise:
    p = X[labels == -1]
    axes[1].scatter(p[:, 0], p[:, 1], s=9, c="0.45", marker="x", label="noise")
axes[0].set_title("Ground-truth labels", fontsize=9)
axes[1].set_title(f"DBSCAN (eps={EPS}, min_samples={MIN_SAMPLES})", fontsize=9)
for ax in axes:
    ax.set_xlim(*VIEW_LIM); ax.set_ylim(*VIEW_LIM); ax.set_aspect("equal")
    ax.set_xlabel("scaled UV"); ax.legend(fontsize=6, loc="center")
axes[0].set_ylabel(r"scaled $\log_{10}(1+\mathrm{IR})$")
fig.savefig("../figures/generated/dbscan_vs_truth.pdf", bbox_inches="tight")
print("\nwrote confusion_dbscan.csv, eps_sweep.csv, dbscan_vs_truth.pdf, cluster_names.json")
'''

"""Part 2 -- score DBSCAN against the ground-truth labels.

DBSCAN never saw the labels. This script compares the clusters it invented
(labels.csv, written by cluster_dbscan.py on the TRAINING split) with the
labels recorded during collection, and writes:

  ../figures/generated/confusion_dbscan.csv    true label x discovered cluster
  ../figures/generated/eps_sweep.csv           clusters/noise/purity vs eps
  ../figures/generated/dbscan_vs_truth.pdf     side-by-side figure
  cluster_names.json                           cluster id -> lighting name
                                               (Part 3 uses it for CLASS_NAMES)

Run cluster_dbscan.py first. Run from inside src/.
"""
"""Part 2 -- score DBSCAN against the ground-truth labels.

DBSCAN never saw the labels. This script compares the clusters it invented
(labels.csv, written by cluster_dbscan.py on the TRAINING split) with the
labels recorded during collection, and writes:

  ../figures/generated/confusion_dbscan.csv    true label x discovered cluster
  ../figures/generated/eps_sweep.csv           clusters/noise/purity vs eps
  ../figures/generated/dbscan_vs_truth.pdf     side-by-side figure
  cluster_names.json                           cluster id -> lighting name
                                               (Part 3 uses it for CLASS_NAMES)

Run cluster_dbscan.py first. Run from inside src/.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from pipeline_common import load_split_scaled, VIEW_LIM

# Must match cluster_dbscan.py
EPS, MIN_SAMPLES = 0.03, 10
SWEEP = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15]

data = load_split_scaled()
X, truth = data["train"], data["y_train"]
labels = np.loadtxt("labels.csv", dtype=int)
assert len(labels) == len(X), "labels.csv is stale: re-run cluster_dbscan.py"

# ---- confusion table: rows = true label, cols = discovered cluster (-1 = noise)
conf = pd.crosstab(pd.Series(truth, name="true label"),
                   pd.Series(labels, name="DBSCAN cluster"))
print("Confusion table (training split):")
print(conf.to_string(), "\n")
conf.to_csv("../figures/generated/confusion_dbscan.csv")

# ---- name each cluster after its majority true label
names = {}
for c in sorted(set(labels) - {-1}):
    col = conf[c]
    names[int(c)] = str(col.idxmax())
    print(f"cluster {c} -> {col.idxmax():12s} purity {col.max() / col.sum():.1%} "
          f"({col.sum()} points)")
noise = int((labels == -1).sum())
print(f"noise: {noise} points ({noise / len(labels):.1%})")
if len(set(names.values())) < len(names):
    print("WARNING: two clusters map to the same label (a class was split).")
missing = set(truth) - set(names.values())
if missing:
    print(f"WARNING: no cluster for {missing} (classes were merged).")
with open("cluster_names.json", "w") as f:
    json.dump(names, f, indent=2)

# ---- eps sweep: how sensitive is the discovery?
rows = []
for e in SWEEP:
    lab = DBSCAN(eps=e, min_samples=MIN_SAMPLES).fit_predict(X)
    k = len(set(lab) - {-1})
    m = lab != -1
    c = pd.crosstab(lab[m], truth[m]) if m.any() else None
    purity = c.max(axis=1).sum() / m.sum() if m.any() else 0.0
    rows.append((e, k, float(np.mean(lab == -1)), float(purity)))
sweep = pd.DataFrame(rows, columns=["eps", "clusters", "noise_frac", "purity"])
print("\neps sweep (min_samples fixed at", MIN_SAMPLES, "):")
print(sweep.to_string(index=False))
sweep.to_csv("../figures/generated/eps_sweep.csv", index=False)

# ---- figure: ground truth vs DBSCAN, same points
MARKERS = ["o", "s", "^", "D", "v", "P"]
GRAYS = ["0.75", "0.45", "0.15", "0.6", "0.3", "0.85"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharex=True, sharey=True)
for i, t in enumerate(sorted(set(truth))):
    p = X[truth == t]
    axes[0].scatter(p[:, 0], p[:, 1], s=6, marker=MARKERS[i], facecolor=GRAYS[i],
                    edgecolor="black", linewidth=0.25, label=t)
for i, c in enumerate(sorted(set(labels) - {-1})):
    p = X[labels == c]
    axes[1].scatter(p[:, 0], p[:, 1], s=6, marker=MARKERS[i % 6], facecolor=GRAYS[i % 6],
                    edgecolor="black", linewidth=0.25,
                    label=f"cluster {c} ({names[int(c)]})")
if noise:
    p = X[labels == -1]
    axes[1].scatter(p[:, 0], p[:, 1], s=9, c="0.45", marker="x", label="noise")
axes[0].set_title("Ground-truth labels", fontsize=9)
axes[1].set_title(f"DBSCAN (eps={EPS}, min_samples={MIN_SAMPLES})", fontsize=9)
for ax in axes:
    ax.set_xlim(*VIEW_LIM); ax.set_ylim(*VIEW_LIM); ax.set_aspect("equal")
    ax.set_xlabel("scaled UV"); ax.legend(fontsize=6, loc="center")
axes[0].set_ylabel(r"scaled $\log_{10}(1+\mathrm{IR})$")
fig.savefig("../figures/generated/dbscan_vs_truth.pdf", bbox_inches="tight")

# ---- zoomed figure: the clusters are tiny on the full [0,1] square, so show
# an overview plus close-ups of the indoor group and the sun cluster, with the
# GRID_N x GRID_N lookup-grid lines and IR in sensor units on the right axis.
from pipeline_common import GRID_N

lo, hi = data["bounds"]
to_ir = lambda y: 10 ** (y * (hi[1] - lo[1]) + lo[1]) - 1           # scaled -> raw IR
to_y = lambda ir: (np.log10(1 + np.maximum(ir, 0)) - lo[1]) / (hi[1] - lo[1])
to_uv = lambda x: x * (hi[0] - lo[0]) + lo[0]                         # scaled -> raw UV
to_x = lambda uv: (uv - lo[0]) / (hi[0] - lo[0])


COLORS = plt.cm.tab10.colors


def scatter_clusters(ax, s, mask=None, annotate=False):
    mask = np.ones(len(X), bool) if mask is None else mask
    for i, c in enumerate(sorted(set(labels) - {-1})):
        m = (labels == c) & mask
        if not m.any():
            continue
        p = X[m]
        ax.scatter(p[:, 0], p[:, 1], s=s, marker=MARKERS[i % 6], color=COLORS[i],
                   edgecolor="black", linewidth=0.25, label=f"cluster {c}: {names[int(c)]}")
        if annotate:      # name + raw IR range next to the group
            ir = to_ir(p[:, 1])
            ax.annotate(f"{names[int(c)]}\nIR {ir.min():.0f}–{ir.max():.0f}",
                        (p[:, 0].max(), p[:, 1].mean()), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=7,
                        color=COLORS[i])
    if noise and (labels[mask] == -1).any():
        p = X[(labels == -1) & mask]
        ax.scatter(p[:, 0], p[:, 1], s=9, c="0.45", marker="x", label="noise")


def zoom_limits(pts, pad=0.25, min_span=0.02):
    lo_, hi_ = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi_ - lo_, min_span)
    mid = (lo_ + hi_) / 2
    return mid - span * (0.5 + pad), mid + span * (0.5 + pad)


groups = {"indoor (UV = 0)": X[:, 0] < 0.5, "sun": X[:, 0] >= 0.5}
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6),
                         gridspec_kw={"width_ratios": [1, 1, 1]})

# overview with boxes around the zoomed regions
scatter_clusters(axes[0], 4)
axes[0].set_xlim(*VIEW_LIM); axes[0].set_ylim(*VIEW_LIM); axes[0].set_aspect("equal")
axes[0].set_title("Overview (scaled feature space)", fontsize=9)
axes[0].set_xlabel("scaled UV"); axes[0].set_ylabel(r"scaled $\log_{10}(1+\mathrm{IR})$")

for ax, (title, m) in zip(axes[1:], groups.items()):
    (x0, y0), (x1, y1) = zoom_limits(X[m])
    axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor="black", lw=0.8, ls="--"))
    scatter_clusters(ax, 8, mask=m, annotate=True)
    for v in np.arange(GRID_N + 1) / GRID_N:      # lookup-grid cell edges
        ax.axvline(v, color="0.6", lw=0.6, ls=":", zorder=0)
        ax.axhline(v, color="0.6", lw=0.6, ls=":", zorder=0)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_title(f"Zoom: {title}  (dotted = {GRID_N}x{GRID_N} grid)", fontsize=9)
    ax.set_xlabel("scaled UV")
    sec_y = ax.secondary_yaxis("right", functions=(to_ir, to_y))
    sec_y.set_ylabel("IR (raw counts)", fontsize=8)
    sec_x = ax.secondary_xaxis("top", functions=(to_uv, to_x))
    sec_x.set_xlabel("UV (raw ADC)", fontsize=8)
    ax.legend(fontsize=6, loc="best")

fig.tight_layout()
fig.savefig("../figures/generated/dbscan_zoom.pdf", bbox_inches="tight")

# ---- one panel per indoor cluster, each zoomed to its own IR range.
# Every indoor point has UV = 0, so each cluster is a vertical strip: the
# spread you see is IR only. Points are semi-transparent so overlaps show
# as darker regions.
indoor = [c for c in sorted(set(labels) - {-1}) if np.all(X[labels == c][:, 0] < 0.5)]
order = sorted(indoor, key=lambda c: X[labels == c][:, 1].mean())   # dim -> bright
fig, axes = plt.subplots(1, len(order), figsize=(3.0 * len(order), 3.6))
for ax, c in zip(np.atleast_1d(axes), order):
    i = sorted(set(labels) - {-1}).index(c)
    p = X[labels == c]
    ymin, ymax = p[:, 1].min(), p[:, 1].max()
    pad = max(0.15 * (ymax - ymin), 0.002)
    ax.scatter(p[:, 0], p[:, 1], s=14, marker=MARKERS[i % 6], color=COLORS[i],
               edgecolor="black", linewidth=0.25, alpha=0.35)
    for v in np.arange(GRID_N + 1) / GRID_N:      # grid row edges, if any fall inside
        ax.axhline(v, color="0.5", lw=0.8, ls=":", zorder=0)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(-0.01, 0.01)
    ax.set_xticks([0]); ax.set_xticklabels(["0"])
    ax.set_xlabel("scaled UV (all points = 0)")
    ir = to_ir(p[:, 1])
    rows = sorted({min(int(y * GRID_N), GRID_N - 1) for y in p[:, 1]})
    ax.set_title(f"cluster {c}: {names[int(c)]}\nIR {ir.min():.0f}–{ir.max():.0f}, "
                 f"grid row(s) {', '.join(map(str, rows))}", fontsize=9)
    sec_y = ax.secondary_yaxis("right", functions=(to_ir, to_y))
    sec_y.set_ylabel("IR (raw counts)", fontsize=8)
np.atleast_1d(axes)[0].set_ylabel(r"scaled $\log_{10}(1+\mathrm{IR})$")
fig.tight_layout()
fig.savefig("../figures/generated/dbscan_zoom_indoor.pdf", bbox_inches="tight")
print("\nwrote confusion_dbscan.csv, eps_sweep.csv, dbscan_vs_truth.pdf, "
      "dbscan_zoom.pdf, dbscan_zoom_indoor.pdf, cluster_names.json")