# sim loop for tti sim with pooled tests
from __future__ import division

import math
import os
import pickle
import random
import time
from typing import Any, Dict, Optional
import copy

import numpy as np
from networkx import Graph
from seirsplus.assignment import embed_nodes, get_equal_sized_clusters
from seirsplus.pooled_test import OneStageGroupTesting
from seirsplus.viral_model import ViralExtSEIRNetworkModel


class SimulationRunner:
    """Runner class for SEIRS+ simulation with pooled testing."""

    def __init__(
        self,
        model: ViralExtSEIRNetworkModel,
        pooling_strategy: str,
        T: int,
        num_groups: int,
        pool_size: int,
        LoD: int,
        seed: int,
        save_results: bool = True,
        output_path: Optional[str] = None,
        verbose: bool = False,
        max_dt: Optional[float] = None,
        community_size: int = None,
        dilute: str = "average",
    ):
        r"""Initialize the simulation runner.

        Args:
            model: A `ViralExtSEIRNetworkModel` object.
            T: Duration (in days) of the simulation.
            num_groups: Number of testing groups to split the population into.
            pool_size: Size of the pooled tests in one-stage group testing.
            seed: The random seed for the simulation.
            save_results: Whether to save the simulation results, default True.
            output_path: The directory to save the simulation results to.
            verbose: Whether to print the simulation progress, default False.
            max_dt: Maximum allowed time-between-transition; if time till next
                transition exceeds max_dt, advance the model by max_dt without
                executing any transition. If None, defaults to T. 
            community_size: Size of the community, used in weak-correlation scenario.

        Returns:
            None.
        """
        # set manual seed
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        # simulation setup
        self.model = model

        self.T = T
        self.max_dt = max_dt
        self.num_groups = num_groups
        self.pool_size = pool_size
        self.community_size = community_size if community_size else 2*self.pool_size
        self.pcr_params={
            "V_sample": 1,
            "c_1": 1/10,
            "xi": 1/2,
            "c_2": 1,
            "LoD": LoD,
            "dilute": dilute
        }
        self.screening_groups = self.get_groups(
            graph=self.model.G_weighted,
            cluster_size=math.ceil(self.model.numNodes / self.num_groups),
        )
        if pooling_strategy not in ["naive", "correlated", "correlated_weak"]:
            raise NotImplementedError(
                f"Pooling strategy {pooling_strategy} not implemented."
            )
        self.pooling_strategy = pooling_strategy
        self.save_results = save_results
        self.output_path = output_path
        if self.save_results:
            if self.output_path is None:
                raise ValueError("Please specify an output path to save the results.")
            os.makedirs(self.output_path, exist_ok=True)
        self.verbose = verbose

        self.isolation_states = [
            model.Q_S,
            model.Q_E,
            model.Q_pre,
            model.Q_sym,
            model.Q_asym,
            model.Q_R,
        ]

        # list of dicts where each dict logs the diagnostics for one day of screening
        self.daily_results = []
        # list of dicts where each dict logs the day, the total number recovered, and the cumulative test performance so far
        self.overall_results = []

        if verbose:
            print(
                f"Running simulation with seed {self.seed} "
                f"for strategy {self.pooling_strategy}..."
            )

    def get_groups(self, graph: Graph, cluster_size: int) -> Dict[int, Any]:
        """Get the screening groups or pools for the simulation.

        Args:
            graph: The networkx graph object.
            cluster_size: The size of groups or pools into which we split the population.
        """
        embedding, node2vec_model = embed_nodes(graph)
        clusters = get_equal_sized_clusters(
            X=embedding,
            model=node2vec_model,
            graph=graph,
            cluster_size=cluster_size,
        )  # dict of node_id to cluster id
        cluster_ids = list(set(clusters.values()))  # unique list of cluster ids
        groups = {
            i: [x for x, v in clusters.items() if v == i] for i in cluster_ids
        }  # dict of cluster ids as the keys and the node ids as the values
        return groups

    def run_screening_one_day(
        self, screening_group_id: int, dayOfNextIntervention: int
    ):
        """
        Test non-isolated individuals in one screening_group on one day.

        Assumptions:
        - There is no upper bound on the amount of tests per day.
        - No symptomatic testing or contact tracing
        - Assumes 100% test compliance

        Args:
            screening_group_id: ID of the group to be screened on this day

        Returns / Saves:
            test results
            statistics of test procedure, e.g., FNR and test consumption
        """

        if self.verbose:
            print(
                f"Running screening for group {screening_group_id} "
                f"on day {dayOfNextIntervention}..."
            )

        nodeStates = self.model.X.flatten()

        # test those in the screening_group and *not isolated*
        screening_group = self.screening_groups[screening_group_id]  # list of IDs
        screening_group = [
            x for x in screening_group if nodeStates[x] not in self.isolation_states
        ]

        np_random_state = np.random.get_state()
        random_state = random.getstate()

        # divide individuals in screening group into pools according to pooling strategy
        # store pooling result and viral loads in nested lists
        if self.pooling_strategy == "correlated":
            pools = self.get_groups(
                self.model.G_weighted.subgraph(screening_group),
                cluster_size=self.pool_size,
            )
            pools = [v for _, v in pools.items()]  # a list of lists
        elif self.pooling_strategy == "correlated_weak":
            community_pools = self.get_groups(
                self.model.G_weighted.subgraph(screening_group),
                cluster_size=self.community_size
            )
            community_pools = [v for _, v in community_pools.items()]
            pools = []
            for community_pool in community_pools:
                random.shuffle(community_pool)
                pools += [
                    community_pool[i : i + self.pool_size]
                    for i in range(0, len(community_pool), self.pool_size)
                ]                
        elif self.pooling_strategy == "naive":
            random.shuffle(screening_group)
            pools = [
                screening_group[i : i + self.pool_size]
                for i in range(0, len(screening_group), self.pool_size)
            ]
        viral_loads = [
            [int(10 ** self.model.current_VL[x]) for x in pool] for pool in pools
        ]
        viral_loads_in_positive_pools = [
            [
                (x, np.round(self.model.infection_start_times[x],2), np.round(self.model.current_VL[x],2)) 
                for x in pool if self.model.current_VL[x]>0
            ] for pool,vl in zip(pools,viral_loads) if max(vl)>0
        ]
        print("Viral loads in positive pools: ", viral_loads_in_positive_pools)
        group_testing = OneStageGroupTesting(
            ids=pools, 
            viral_loads=viral_loads,
            pcr_params=self.pcr_params
        )
        test_results, diagnostics = group_testing.run_one_stage_group_testing()

        np.random.set_state(np_random_state)
        random.setstate(random_state)

        num_susceptible_neighbors_of_identified = 0
        num_susceptible_neighbors_of_unidentified = 0

        # pass test_results to update isolation status in self.model
        for pool_idx, pool in enumerate(pools):
            for individual_idx, individual in enumerate(pool):
                # get the number of susceptible neighbors
                neighbors = list(self.model.G[individual].keys())
                num_susceptible_neighbors = sum(self.model.X[nb]==1 for nb in neighbors)
                if test_results[pool_idx][individual_idx] == 1:
                    self.model.set_positive(individual, True)
                    self.model.set_isolation(individual, True)
                    self.model.isolation_start_times[individual] = self.model.t
                    num_susceptible_neighbors_of_identified += num_susceptible_neighbors # TODO: use set() to prevent double counting
                else:
                    if viral_loads[pool_idx][individual_idx] > 0:
                        num_susceptible_neighbors_of_unidentified += num_susceptible_neighbors

        self.daily_results.append(diagnostics)

        performance = (
            self.get_cumulative_test_performance()
        )  # cumulative test performance
        performance["day"] = dayOfNextIntervention
        # performance["cumRecovered"] = self.model.total_num_recovered(self.model.tidx)
        performance["cumRecovered"] = self.model.numR[self.model.tidx] \
            + self.model.numQ_R[self.model.tidx]
        performance["cumInfections"] = (
            self.model.numR[self.model.tidx]
            + self.model.numQ_R[self.model.tidx]
            + self.model.numE[self.model.tidx]
            + self.model.numI_pre[self.model.tidx]
            + self.model.numI_sym[self.model.tidx]
            + self.model.numI_asym[self.model.tidx]
            + self.model.numH[self.model.tidx]
            + self.model.numQ_E[self.model.tidx]
            + self.model.numQ_pre[self.model.tidx]
            + self.model.numQ_sym[self.model.tidx]
            + self.model.numQ_asym[self.model.tidx]
        )
        performance["numActiveInfections"] = (
            self.model.numE[self.model.tidx]
            + self.model.numI_pre[self.model.tidx]
            + self.model.numI_sym[self.model.tidx]
            + self.model.numI_asym[self.model.tidx]
            + self.model.numH[self.model.tidx]
        )
        performance["numQuarantinedInfections"] = (
            self.model.numQ_E[self.model.tidx]
            + self.model.numQ_pre[self.model.tidx]
            + self.model.numQ_sym[self.model.tidx]
            + self.model.numQ_asym[self.model.tidx]
        )
        performance["mean_num_positives_in_positive_pool"] = np.mean(diagnostics["num_positives_per_positive_pool"])
        performance["mean_num_identifiable_positives_in_positive_pool"] = np.mean(diagnostics["num_identifiable_positives_per_positive_pool"])
        performance["median_num_positives_in_positive_pool"] = np.median(diagnostics["num_positives_per_positive_pool"])
        performance["median_num_identifiable_positives_in_positive_pool"] = np.median(diagnostics["num_identifiable_positives_per_positive_pool"])
        performance["daily_sensitivity"] = np.divide(diagnostics["num_identified"], diagnostics["num_positives"])
        performance["daily_effective_efficiency"] = np.divide(diagnostics["num_identified"], diagnostics["num_tests"])
        performance["daily_effective_followup_efficiency"] = np.divide(diagnostics["num_identified"], (diagnostics["num_tests"] - len(test_results)))
        performance["num_susceptible_neighbors_of_identified_positives"] = num_susceptible_neighbors_of_identified
        performance["num_susceptible_neighbors_of_unidentified_positives"] = num_susceptible_neighbors_of_unidentified
        performance["VL_in_positive_pools"] = viral_loads_in_positive_pools

        self.overall_results.append(performance)

        if self.save_results:
            with open(
                os.path.join(self.output_path, f"results_{self.seed}.pickle"), "wb"
            ) as f:
                pickle.dump(self.overall_results, f)

    def run_simulation(self):
        r"""
        Run screening for the full duration self.T.
        """

        dayOfNextIntervention = 0
        self.model.tmax = self.T

        # TODO [P1]: if time % 10 == 0, save snapshot of current viral loads

        while True:
            # first run a model iteration, i.e., one transition
            running = self.model.run_iteration(max_dt = self.max_dt)

            if running == False:
                break

            # make sure we don't skip any days due to the transition
            # implement testing at the start of each day, on 0, 1, ..., T-1,
            while dayOfNextIntervention <= int(self.model.t):
                group_id = dayOfNextIntervention % self.num_groups
                self.run_screening_one_day(group_id, dayOfNextIntervention)
                if self.verbose:
                    print(
                        "Screening day: ",
                        dayOfNextIntervention,
                        " self.model.t: ",
                        self.model.t,
                    )
                dayOfNextIntervention += 1

    def get_cumulative_test_performance(self) -> Dict[str, float]:
        r"""
        Compute the sensitivity and test consumption of a testing strategy over time.

        self.daily_results is a list of dicts where each dict contains results of
        group testing conducted on one screening group on one day.
        Keys are 'sensitivity', 'num_tests, 'num_positives', 'num_identified'.

        Returns:
            cum_positives_screened: cumulative number of positive samples screened so far
            cum_positives_identified: cumulative number of positive samples identified in the screening so far
            cum_sensitivity: cumulative sensitivity of the tests (cum_num_identified / cum_num_positives)
            cum_num_tests: cumulative number of PCR tests consumed so far
        """

        cum_positives_screened = sum(
            [result["num_positives"] for result in self.daily_results]
        )

        cum_positives_identified = sum(
            [result["num_identified"] for result in self.daily_results]
        )

        if cum_positives_screened > 0:
            cum_sensitivity = cum_positives_identified / cum_positives_screened
        else:
            cum_sensitivity = float("nan")

        cum_num_tests = sum([result["num_tests"] for result in self.daily_results])
        # TODO: also need to retrive number of positives per pool
        # [result["num_positives"] for result in self.daily_results] -- this is wrong, not per-pool

        return {
            "cum_positives_screened": cum_positives_screened,
            "cum_positives_identified": cum_positives_identified,
            "cum_sensitivity": cum_sensitivity,
            "cum_num_tests": cum_num_tests,
        }
