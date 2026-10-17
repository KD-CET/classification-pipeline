"""Part 4 -- validate the device on a FRESH held-out batch.

Input: a serial log saved from firmware/inference/inference.ino, with rows
    rawUV,rawIR,uvF,irF,label,pred,score
(lines starting with '#' are ignored).

Answers two questions:
  1. Device-vs-desktop agreement: feed the SAME filtered readings (uvF, irF)
     through the Python lookup; does it give the same answer as the ESP32?
     Should be ~100%; anything else means a scaling/rounding mismatch.
  2. Held-out accuracy: does the device's answer match the lighting you
     said it was (the label column)? This batch was never used for training.

Run from inside src/:
    python check_agreement.py ../data/raw/heldout_device_log.csv
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from pipeline_common import (load_split_scaled, scale_features, to_features,
                             cell_of, quantize, dequantize, REJECT_TAU)

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

# ---- the desktop model: same tables, same bounds as model.h
bounds = load_split_scaled()["bounds"]
surfaces = np.load("surfaces.npz")
classes = sorted(int(k[1:]) for k in surfaces.files)
tables = {c: quantize(surfaces[f"c{c}"]) for c in classes}
names = json.load(open("cluster_names.json"))
class_names = [names[str(c)] for c in classes]


def desktop_predict(uv, ir):
    p, _ = scale_features(to_features([uv], [ir])[0], bounds)
    r, c = cell_of(p)
    v = np.array([dequantize(tables[k][r, c]) for k in classes])
    return "UNKNOWN" if v.max() < REJECT_TAU else class_names[int(np.argmax(v))]


log["desktop"] = [desktop_predict(u, i) for u, i in zip(log.uvF, log.irF)]

# ---- 1. agreement (all rows, labelled or not)
agree = (log.pred == log.desktop).mean()
print(f"rows: {len(log)}")
print(f"device-vs-desktop agreement: {100 * agree:.2f}%  "
      f"({(log.pred != log.desktop).sum()} mismatches)")
if agree < 1:
    print("\nmismatched rows (check scaling, log10, rounding):")
    print(log.loc[log.pred != log.desktop, ["uvF", "irF", "pred", "desktop"]].head(10))

# ---- 2. held-out accuracy (only rows you tagged with L:<label>)
lab = log[~log.label.isin(["-", "", "nan"])].copy()
if len(lab):
    norm = {n.lower(): n for n in class_names}
    lab["truth"] = lab.label.str.lower().map(norm).fillna(lab.label)
    acc = (lab.pred == lab.truth).mean()
    unk = (lab.pred == "UNKNOWN").mean()
    print(f"\nheld-out rows: {len(lab)}")
    print(f"held-out accuracy (device vs your label): {100 * acc:.1f}%   "
          f"UNKNOWN: {100 * unk:.1f}%")
    conf = pd.crosstab(lab.truth, lab.pred, rownames=["true"], colnames=["device"])
    print(conf.to_string())
    conf.to_csv("../figures/generated/confusion_heldout_device.csv")
    print("per class:")
    for t, g in lab.groupby("truth"):
        print(f"  {t:12s} n={len(g):4d}  correct {100 * (g.pred == t).mean():5.1f}%  "
              f"unknown {100 * (g.pred == 'UNKNOWN').mean():5.1f}%")
else:
    print("\nno labelled rows: use L:<label> on the device to tag the held-out batch")
