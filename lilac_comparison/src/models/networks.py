from torch import nn


def mlp(inputs, sizes, outputs):
    layers = []

    for width in sizes:
        layers.extend([nn.Linear(inputs, width), nn.ReLU()])
        inputs = width

    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


def cache_average(old, count, entries):
    """Correct cumulative mean, including episode zero and repeated task draws."""
    m = len(entries)

    return (old * count + entries.sum(axis=0)) / (count + m), count + m
