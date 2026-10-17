"""Part 5 -- filtering evidence: raw vs EMA-filtered signal, per class.

Reads the same device log as check_agreement.py (it has both raw and
filtered columns) and plots sensor value vs sample index for each labelled
condition: IR on top, UV below. The point to show: the filter removes
transient glitches without erasing the level differences between classes.

Run from inside src/:
    python plot_filter.py ../data/raw/heldout_device_log.csv
Writes ../figures/generated/raw_vs_filtered.pdf
"""
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/heldout_device_log.csv"
COLS = ["rawUV", "rawIR", "uvF", "irF", "label", "pred", "score"]
LINE = re.compile(r"UV raw\s+(-?\d+)\s+filt\s+(-?[\d.]+)\s*\|\s*IR raw\s+(-?\d+)\s+filt\s+(-?[\d.]+)"
                  r"\s*\|\s*label:\s*(\S+)\s*\|\s*detected:\s*(\S+)\s+conf\s+([\d.]+)%")


def load_log(path):
    """Read a Serial Monitor log from inference.ino.

    Understands the labelled lines ("UV raw ... | IR raw ... | label: ... |
    detected: ... conf ...%") and, as a fallback, plain CSV rows
    rawUV,rawIR,uvF,irF,label,pred,score. Status lines ('#') and anything
    else are skipped, so the whole Serial Monitor can be saved as-is.
    """
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = LINE.search(line)
            if m:
                u, uf, i, irf, lab, pred, conf = m.groups()
                rows.append((int(u), int(i), float(uf), float(irf), lab, pred, float(conf) / 100))
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 7:
                try:
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]),
                                 parts[4], parts[5], float(parts[6])))
                except ValueError:
                    pass          # header line or garbage
    return pd.DataFrame(rows, columns=COLS)

log = load_log(path)
log = log[~log.label.isin(["-", "", "nan"])]
labels = list(dict.fromkeys(log.label))           # in recording order

fig, axes = plt.subplots(2, len(labels), figsize=(3.2 * len(labels), 4.8),
                         squeeze=False)
for j, lab in enumerate(labels):
    g = log[log.label == lab].reset_index(drop=True)
    for i, (raw, filt, name) in enumerate([("rawIR", "irF", "IR"), ("rawUV", "uvF", "UV")]):
        ax = axes[i, j]
        ax.plot(g.index, g[raw], color="0.7", lw=0.8, label="raw")
        ax.plot(g.index, g[filt], color="black", lw=1.2, label=r"EMA, $\alpha=0.2$")
        if i == 0:
            ax.set_title(lab, fontsize=9)
        if j == 0:
            ax.set_ylabel(f"{name} (counts)")
        if i == 1:
            ax.set_xlabel("sample index (~10 Hz)")
        ax.tick_params(labelsize=7)
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig("../figures/generated/raw_vs_filtered.pdf", bbox_inches="tight")
print(f"wrote ../figures/generated/raw_vs_filtered.pdf ({len(labels)} conditions)")

# numbers for the paper: how much the filter reduces sample-to-sample jitter
print(f"\n{'condition':12s} {'signal':6s} {'raw std':>9s} {'filtered std':>13s}")
for lab in labels:
    g = log[log.label == lab]
    for raw, filt, name in [("rawIR", "irF", "IR"), ("rawUV", "uvF", "UV")]:
        print(f"{lab:12s} {name:6s} {g[raw].diff().std():9.2f} {g[filt].diff().std():13.2f}")
print("(std of sample-to-sample change: lower = smoother)")
