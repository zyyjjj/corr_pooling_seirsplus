import os
import random
import sys
from typing import Optional

import numpy as np
import yaml

from seirsplus.networks import (
    generate_demographic_contact_network,
    household_country_data,
)
from seirsplus.sim_loops_pooled_test import SimulationRunner
from seirsplus.viral_model import ViralExtSEIRNetworkModel


def run_simulation(
    seed: int,
    pop_size: int,
    init_prev: float,
    horizon: int,
    num_groups: int,
    pool_size: int,
    beta: float,
    sigma: float,
    lamda: float,
    gamma: float,
    pooling_strategy: str,
    output_path: Optional[str] = None,
    save_results: Optional[bool] = True,
    **kwargs,
):
    """Run simulation.

    Args:
        seed: The random seed for the simulation.
        pop_size: The population size.
        init_prev: The initial prevalence.
        horizon: The time horizon of the simulation.
        num_groups: Number of screening groups to split the population into.
        pool_size: Size of the pooled tests in one-stage group testing.
        beta, sigma, lamda, gamma: The parameters of the SEIR+ model.
        pooling_strategy: The pooling strategy to use. Must be one of "naive"
            and "correlated".
        output_path: The directory to save the simulation results to.
        save_results: Whether to save the simulation results, default True.

    Returns:
        An instance of `SimulationRunner`.
    """
    # set manual seed
    random.seed(seed)
    np.random.seed(seed)

    # generate social network graph
    demographic_graphs, _, _ = generate_demographic_contact_network(
        N=pop_size,
        demographic_data=household_country_data("US"),
        distancing_scales=[0.7],
        isolation_groups=[],
    )
    G = demographic_graphs["baseline"]

    # initiate SEIR+ model
    init_exposed = int(init_prev * pop_size)
    model = ViralExtSEIRNetworkModel(
        G=G,
        beta=beta,
        sigma=sigma,
        lamda=lamda,
        gamma=gamma,
        initE=init_exposed,
    )

    # initiate simulation runner
    sim = SimulationRunner(
        model=model,
        pooling_strategy=pooling_strategy,
        T=horizon,
        num_groups=num_groups,
        pool_size=pool_size,
        seed=seed,
        output_path=output_path,
        save_results=save_results,
    )

    # run simulation
    sim.run_simulation()
    return sim


if __name__ == "__main__":
    kwargs = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    num_seeds = 1
    path = (
        f"../results/US_N={kwargs['pop_size']}_"
        f"p={kwargs['init_prev']}_T={kwargs['horizon']}/"
    )
    for seed in range(num_seeds):
        for pooling_strategy in ["naive", "correlated"]:
            output_path = os.path.join(path, pooling_strategy)
            _ = run_simulation(
                seed=seed,
                pooling_strategy=pooling_strategy,
                output_path=output_path,
                **kwargs,
            )
