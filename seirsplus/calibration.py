from typing import List, Tuple

import numpy as np


def gen_vl_distrbution(
    bounds: List[Tuple],
    lambdas: List[float],
    cap: float,
    num_samples: int,
) -> np.array:
    r"""Generate log10 viral load samples from a piecewise linear model.

    Args:
        bounds: A 3-element list of tuples of the form (min, max) for each slope
            representing the piecewise linear function.
        lambdas: A 2-element list of floats representing the rate of transitions
            between exposed, pre-symptomatic, and symptomatic states.
        cap: A float representing the maximum log10 viral load.
        num_samples: The number of log10 viral load samples to generate.

    Returns:
        A 1-d array of size `num_samples` representing the log10 viral load samples.
    """
    # sample slopes
    slopes = [
        np.random.uniform(bounds[i][0], bounds[i][1], (num_samples, 1))
        for i in range(len(bounds))
    ]
    slopes = np.concatenate(slopes, axis=1)  # shape: (num_samples, 3)

    # sample durations in the exposed and pre-symptomatic states
    durations = [
        np.random.exponential(1 / lambd, (num_samples, 1)) for lambd in lambdas
    ]
    durations = np.concatenate(durations, axis=1)  # shape: (num_samples, 2)

    # compute the peak viral loads
    peaks = np.clip(np.sum(slopes[:, :2] * durations, axis=1), a_min=None, a_max=cap)

    # compute the duration of infection (positive viral load) and sample timestamps
    taus = -1 * peaks / slopes[:, 2] + np.sum(durations, axis=1)
    time_stamps = np.random.uniform(0, taus, num_samples)

    # compute the viral loads corresponding to the sampled timestamps
    viral_loads = np.clip(
        np.clip(time_stamps, a_min=None, a_max=durations[:, 0]) * slopes[:, 0],
        a_min=None,
        a_max=cap,
    )
    viral_loads = np.clip(
        np.clip(time_stamps - durations[:, 0], a_min=0, a_max=durations[:, 1])
        * slopes[:, 1]
        + viral_loads,
        a_min=None,
        a_max=cap,
    )
    viral_loads = (
        np.clip(time_stamps - np.sum(durations[:, :2], axis=1), a_min=0, a_max=None)
        * slopes[:, 2]
        + viral_loads
    )
    return (viral_loads, time_stamps, peaks)
