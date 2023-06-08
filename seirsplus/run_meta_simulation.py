import os
import random
import sys
from typing import Optional
import argparse
import itertools
from multiprocessing.pool import Pool


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
    # assign higher weights to inter-household edges
    for e in G.edges():
        if "weight" not in G[e[0]][e[1]]:
            G[e[0]][e[1]]["weight"] = 10**10

    # initiate SEIR+ model
    init_exposed = int(init_prev * pop_size)
    model = ViralExtSEIRNetworkModel(
        G=G,
        beta=beta,
        sigma=sigma,
        lamda=lamda,
        gamma=gamma,
        initE=init_exposed,
        seed=seed,
        transition_mode="time_in_state"
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
        max_dt=0.01
    )

    # run simulation
    sim.run_simulation()
    return sim


def parse(arg_list):
    # experiment-running params -- read from command line input
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_to_vary", type = str, nargs = "+", default = ["init_prev"])
    parser.add_argument("--num_seeds", type = int)

    args = parser.parse_args(arg_list)

    return args.params_to_vary, args.num_seeds


def run_simulation_wrapper(seed, kwargs):
    for pooling_strategy in ["naive", "correlated"]:
        kwargs_ = kwargs.copy()
        kwargs_["output_path"] = os.path.join(kwargs["output_path"], pooling_strategy)
        _ = run_simulation(
            seed=seed,
            pooling_strategy=pooling_strategy,
            **kwargs_,
        )


if __name__ == "__main__":

    kwargs = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)

    params_to_vary, num_seeds = parse(sys.argv[2:])

    param_values = {}

    for param in [
        "init_prev", "num_groups", "pool_size",
        "pop_size", "horizon", "beta", "sigma", "lamda", "gamma", # typically don't change
    ]:
        if param in params_to_vary:
            param_values[param] = kwargs[param] # a list
        else:
            param_values[param] = [kwargs[param+"_default"]] # 1-element list
    
    all_param_configs = [dict(zip(param_values.keys(), x)) for x in itertools.product(*param_values.values())]

    for param_config in all_param_configs:
        path = "../results/US"
        for param in [
            "pop_size", "init_prev", "num_groups", "pool_size", "horizon", 
            "beta", "sigma", "lamda", "gamma"
        ]:
            path += f"_{param}={param_config[param]}"
        path += "/"
        param_config["output_path"] = path

    with Pool() as pool:
        pool.starmap(
            run_simulation_wrapper, 
            [(seed, param_config) for seed in range(num_seeds) for param_config in all_param_configs]
        )
