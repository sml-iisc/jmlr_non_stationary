import math


def schedule_value(start, kind, progress, floor):
    progress = min(1.0, max(0.0, progress))
    fraction = progress if kind == "linear" else (1 - math.cos(math.pi * progress)) / 2

    return start if kind == "constant" else start * (floor + (1 - floor) * fraction)
