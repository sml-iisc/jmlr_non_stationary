import numpy as np


def schedule_value(start, kind, remaining, floor):
    # SB3 reports progress from one down to zero.
    remaining = min(1.0, max(0.0, remaining))

    if kind == "constant":
        return start

    if kind == "linear":
        return start * (floor + (1 - floor) * remaining)

    if kind == "cosine":
        return start * (floor + (1 - floor) * 0.5 * (1 - np.cos(np.pi * remaining)))

    raise ValueError(kind)
