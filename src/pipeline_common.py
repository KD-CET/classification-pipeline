"""Shared setup: the dataset, feature scaling, the lookup grid, and byte
storage. Defined once here so every stage agrees on all four.

On-Device Classification lab: same API as the Lab 03 version, but the
dataset is now the real labelled sensor CSV instead of the synthetic shapes.

Feature transform (the ESP32 must apply exactly the same thing):
    x1 = UV                 GUVA-S12SD, ESP32 ADC (0-4095), EMA-filtered
    x2 = log10(1 + IR)      TSL2591 IR channel (0-65535), EMA-filtered
IR spans ~130 (shade) to ~32,000 (sun). On a linear scale the three indoor
classes would share one grid cell; the log spreads them out.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

SEED = 20250903

# Flash cost is classes * GRID_N**2 bytes, so this is a memory decision.
GRID_N = 16

# Below this, the winning class is too weak to trust: answer "unknown".
REJECT_TAU = 0.12

# Plot axis limits: the [0,1] feature space padded 10% each side.
VIEW_LIM = (-0.1, 1.1)

# Three-way split, exactly as in Lab 03.
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.60, 0.20, 0.20

# --- real data -------------------------------------------------------------
DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "data_complete.csv"
UV_COL, IR_COL, LABEL_COL = "uv_sensor", "IR_sensor", "label"

# Rows dropped at the start of each class: the EMA filter is still settling
# there (and the fluorescent tube was still warming up).
TRIM_ROWS = 25


def to_features(uv, ir):
    """Raw sensor readings -> feature vector. Identical on the ESP32."""
    return np.column_stack([np.asarray(uv, float),
                            np.log10(1.0 + np.asarray(ir, float))])


def load_labelled(csv_path=None):
    """Features AND ground-truth labels. Labels are only used to SCORE
    DBSCAN afterwards; clustering and KDE never see them."""
    df = pd.read_csv(csv_path or DATA_CSV)
    df = df[df.groupby(LABEL_COL).cumcount() >= TRIM_ROWS].reset_index(drop=True)
    return to_features(df[UV_COL], df[IR_COL]), df[LABEL_COL].to_numpy()


def load_dataset(csv_path=None):
    """Feature matrix only (no labels) -- what DBSCAN is allowed to see."""
    return load_labelled(csv_path)[0]


def scale_features(X, bounds=None):
    """Min-max each feature onto [0, 1]. Returns (X_scaled, bounds).

    Pass the returned bounds back to scale a query point the same way --
    they are part of the shipped model, not a training detail.
    """
    lo, hi = bounds if bounds is not None else (X.min(axis=0), X.max(axis=0))
    return (X - lo) / (hi - lo), (lo, hi)


def split_data(X, y=None, seed=SEED):
    """Split into (train, val, test), 60/20/20, fixed seed. If labels y are
    given they are split with the SAME shuffle and returned as a second tuple."""
    if y is None:
        train, rest = train_test_split(X, train_size=TRAIN_FRAC, random_state=seed)
        val, test = train_test_split(rest, test_size=0.5, random_state=seed)
        return train, val, test
    Xtr, Xrest, ytr, yrest = train_test_split(X, y, train_size=TRAIN_FRAC,
                                              random_state=seed, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xrest, yrest, test_size=0.5,
                                          random_state=seed, stratify=yrest)
    return (Xtr, Xva, Xte), (ytr, yva, yte)


def load_split_scaled(seed=SEED):
    """Load, split, and scale every subset by TRAIN's bounds.

    Returns the same keys as Lab 03 ("train", "val", "test", "bounds") plus
    the ground-truth labels of each split ("y_train", "y_val", "y_test"),
    which only the scoring script reads.
    """
    X, y = load_labelled()
    (train, val, test), (ytr, yva, yte) = split_data(X, y, seed)
    train_s, bounds = scale_features(train)
    val_s, _ = scale_features(val, bounds)
    test_s, _ = scale_features(test, bounds)
    return {"train": train_s, "val": val_s, "test": test_s, "bounds": bounds,
            "y_train": ytr, "y_val": yva, "y_test": yte}


def make_grid():
    """Cell centres of the GRID_N x GRID_N grid covering [0, 1]^2."""
    centres = (np.arange(GRID_N) + 0.5) / GRID_N
    xx, yy = np.meshgrid(centres, centres)
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def cell_of(point):
    """The (row, col) cell holding a scaled point."""
    i = np.clip((np.asarray(point) * GRID_N).astype(int), 0, GRID_N - 1)
    return int(i[1]), int(i[0])


def fit_density(points, bandwidth):
    """Fit a Gaussian KDE. score_samples() returns LOG density."""
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    return kde.fit(points)


def evaluate_on_grid(kde, grid_points, xx):
    """Evaluate a fitted KDE at every cell centre."""
    return np.exp(kde.score_samples(grid_points)).reshape(xx.shape)


def normalise_peak(surface):
    """Scale a density map so its own peak is 1.0."""
    peak = surface.max()
    return surface / peak if peak > 0 else surface


def quantize(surface_01):
    """Store a peak-normalised map as one byte per cell."""
    return np.round(np.clip(surface_01, 0.0, 1.0) * 255).astype(np.uint8)


def dequantize(codes):
    """Read stored bytes back as values in [0, 1]."""
    return codes.astype(float) / 255.0