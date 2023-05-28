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


# TODO: later we can adapt this model to update the VL mechanism in ViralModel
def _get_vl_with_plateau(
    critical_time_points: List[float],
    peak_plateau_height: float,
    tail_plateau_height: float,
    sample_time: float,
):
    r"""Sample one log10 VL at a given time under the VL-with-plateau model.
    
    Args:
        critical_time_points: A list of 4 floats reprensenting the time points 
            at which VL starts the peak plateau, starts decaying from the peak plateau, 
            enters the tail plateau, and drops to 0 from the tail plateau. # TODO: double check
        peak_plateau_height: A float representing the height of the peak plateau.
        tail_plateau_height: A float representing the height of the tail plateau.
        sample_time: A float representing the time at which to sample the VL.
    Returns:
        A float representing the log10 VL at the given time.
    """

    start_peak_time, start_decay_time, start_tail_time, end_tail_time = critical_time_points

    vl = 0

    if sample_time < start_peak_time:
        vl = peak_plateau_height / start_peak_time * sample_time
    elif sample_time < start_decay_time:
        vl = peak_plateau_height
    elif sample_time < start_tail_time:
        vl = peak_plateau_height - (peak_plateau_height - tail_plateau_height) / (start_tail_time - start_decay_time) * (sample_time - start_decay_time)
    elif sample_time < end_tail_time:
        vl = tail_plateau_height
    else:
        vl = 0
    
    return float(vl)


def gen_vl_distribution_with_plateau(
    critical_time_points_bounds: List[Tuple[float]],
    peak_plateau_height_bounds: Tuple[float], # TODO: keep this fixed? or vary
    tail_plateau_height: float,
    num_samples: int,
    noise: float = 0.5
):
    r"""Generate log10 viral load samples from a piecewise linear model with plateau.

    Args:
        critical_time_points_bounds: A 4-element list of tuples of the form 
            (min, max) for each time point at which VL starts the peak plateau, 
            starts decaying from the peak plateau, enters the tail plateau, 
            and drops to 0 from the tail plateau. 
        peak_plateau_height_bounds: A tuple of the form (min, max) for the
            height of the peak plateau.
        tail_plateau_height: A float representing the height of the tail plateau.
        num_samples: The number of log10 viral load samples to generate.
        noise: A float representing the standard deviation of the Gaussian noise
            added to the sampled viral loads.

    Returns:
        A 1-d array of size `num_samples` representing the log10 viral load samples.
    """


    # sample critical time points
    critical_time_points_l = [
        np.random.uniform(
        critical_time_points_bounds[i][0], 
        critical_time_points_bounds[i][1],
        (num_samples, 1)
        )
        for i in range(len(critical_time_points_bounds))
    ]
    critical_time_points_l = np.concatenate(critical_time_points_l, axis=1)  # shape: (num_samples, 3)

    # sample the peak plateau heights
    peak_plateau_height_l = np.random.uniform(
        peak_plateau_height_bounds[0], 
        peak_plateau_height_bounds[1], 
        (num_samples, 1)
    )

    time_stamps = np.random.uniform(0, critical_time_points_l[:, -1], num_samples)

    sampled_vls = []

    for i in range(num_samples):
        _vl = _get_vl_with_plateau(
            critical_time_points_l[i], 
            peak_plateau_height_l[i], 
            tail_plateau_height,
            time_stamps[i]
        )
        _vl += np.random.normal(0, noise)
        sampled_vls.append(_vl)
    
    return sampled_vls, time_stamps


