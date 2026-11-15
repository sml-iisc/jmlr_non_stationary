import numpy as np


def sample_coupled(uniform, baseline, weights, tau, mass):
    """All t probabilities depend ONLY on events before t."""
    decay = np.exp(-1 / tau)
    jump = mass * (1 - decay)
    events = np.zeros(uniform.shape, dtype=bool)
    probabilities = np.empty(uniform.shape, dtype=np.float64)
    probabilities[0] = baseline
    excitation = np.zeros(len(baseline))

    for second in range(1, len(uniform)):
        active = np.flatnonzero(events[second - 1])
        excitation *= decay

        if len(active):
            excitation += jump * weights[:, active].sum(axis=1)

        probabilities[second] = 1 - (1 - baseline) * np.exp(-excitation)
        events[second] = uniform[second] < probabilities[second]

    return events, probabilities
