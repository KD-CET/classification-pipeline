"""Part4 skeleton -- YOU write the lookup.

The tables, the grid, and the scaling are done.
Fill in the two TODOs, then check against classify_point.py.
"""

import numpy as np
from pipeline_common import (
    load_split_scaled,
    scale_features,
    quantize,
    dequantize,
    GRID_N,
    REJECT_TAU
)

bounds = load_split_scaled()["bounds"]  # training bounds ship with the model

surfaces = np.load("surfaces.npz")

classes = sorted(
    int(k[1:]) for k in surfaces.files
)

# The shipped model. Your code may read only these and 'bounds'.
tables = {
    c: quantize(surfaces[f"c{c}"])
    for c in classes
}


def classify(raw_point):
    """Class label for a RAW point, or None for "unknown"."""

    point, _ = scale_features(
        np.asarray(raw_point),
        bounds
    )

    # TODO 1: turn 'point' into a grid cell.
    col = np.clip(
        int(point[0] * GRID_N),
        0,
        GRID_N - 1
    )

    row = np.clip(
        int(point[1] * GRID_N),
        0,
        GRID_N - 1
    )

    # TODO 2: read tables[c][row, col] for each class,
    # convert with dequantize(), and return the class
    # with the largest value.

    scores = {}

    for c in classes:
        scores[c] = dequantize(tables[c][row, col])

    # Find the class with the largest density.
    best_class = max(scores, key=scores.get)
    best_value = scores[best_class]

    # Reject the point if the winning value is below the threshold.
    if best_value < REJECT_TAU:
        return None

    return best_class


if __name__ == "__main__":
    # One raw point on each shape, then the ring's hollow centre (unknown).
    for q in (
        [0.0, -1.8],
        [0.0, 5.0],
        [6.8, 1.6],
        [4.8, 1.6]
    ):
        print(q, "->", classify(q))