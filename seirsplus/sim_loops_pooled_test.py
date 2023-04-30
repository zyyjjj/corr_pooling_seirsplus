# sim loop for tti sim with pooled tests
from __future__ import division

import math
import os
import pickle
import random
import time
from typing import Any, Dict, Optional

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
        seed: int,
        save_results: bool = True,
        output_path: Optional[str] = None,
        verbose: bool = False,
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
        self.num_groups = num_groups
        self.pool_size = pool_size
        self.screening_groups = self.get_groups(
            graph=self.model.G,
            cluster_size=math.ceil(self.model.numNodes / self.num_groups),
        )
        if pooling_strategy not in ["naive", "correlated"]:
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
            print("Running simulation with seed ", self.seed, " for strategy ", \
                self.pooling_strategy, "...")

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
            print("Running screening for group", screening_group_id, "on day", dayOfNextIntervention, "...")

        nodeStates = self.model.X.flatten()

        # test those in the screening_group and *not isolated*
        screening_group = self.screening_groups[screening_group_id]  # list of IDs
        screening_group = [
            x for x in screening_group if nodeStates[x] not in self.isolation_states
        ]

        # update VL for everyone except self.model.transitionNode
        self.model.update_VL(nodes_to_exclude=[self.model.transitionNode])

        # divide individuals in screening group into pools according to pooling strategy
        # store pooling result and viral loads in nested lists
        if self.pooling_strategy == "correlated":
            pools = self.get_groups(
                graph=self.model.G.subgraph(screening_group),
                cluster_size=self.pool_size,
            )
            pools = [v for _, v in pools.items()]  # a list of lists
        elif self.pooling_strategy == "naive":
            random.shuffle(screening_group)
            pools = [
                screening_group[i : i + self.pool_size]
                for i in range(0, len(screening_group), self.pool_size)
            ]
        viral_loads = [
            [int(10 ** self.model.current_VL[x]) for x in pool] for pool in pools
        ]
        group_testing = OneStageGroupTesting(ids=pools, viral_loads=viral_loads)
        test_results, diagnostics = group_testing.run_one_stage_group_testing()

        # pass test_results to update isolation status in self.model
        for pool_idx, pool in enumerate(pools):
            for individual_idx, individual in enumerate(pool):
                if test_results[pool_idx][individual_idx] == 1:
                    self.model.set_positive(individual, True)
                    self.model.set_isolation(individual, True)

        self.daily_results.append(diagnostics)

        performance = (
            self.get_cumulative_test_performance()
        )  # cumulative test performance
        performance["day"] = dayOfNextIntervention
        performance["cumRecovered"] = np.max(self.model.total_num_recovered())  

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
            running = self.model.run_iteration()

            if running == False:
                break

            # make sure we don't skip any days due to the transition
            # implement testing at the start of each day, on 0, 1, ..., T-1,
            while dayOfNextIntervention <= int(self.model.t):
                group_id = dayOfNextIntervention % self.num_groups
                self.run_screening_one_day(group_id, dayOfNextIntervention)
                dayOfNextIntervention += 1

    def get_cumulative_test_performance(self) -> Dict[str, float]:
        r"""
        Compute the sensitivity and test consumption of a testing strategy over time.

        self.daily_results is a list of dicts where each dict contains results of
        group testing conducted on one screening group on one day.
        Keys are 'sensitivity', 'num_tests, 'num_positives', 'num_identified'.

        Returns:
            cum_num_positives: cumulative number of positive samples screened so far
            cum_num_identified: cumulative number of positive samples identified in the screening so far
            cum_sensitivity: cumulative sensitivity of the tests (cum_num_identified / cum_num_positives)
            cum_num_tests: cumulative number of PCR tests consumed so far
        """

        cum_num_positives = sum(
            [result["num_positives"] for result in self.daily_results]
        )

        cum_num_identified = sum(
            [result["num_identified"] for result in self.daily_results]
        )

        if cum_num_positives > 0:
            cum_sensitivity = cum_num_identified / cum_num_positives
        else:
            cum_sensitivity = float("nan")

        cum_num_tests = sum([result["num_tests"] for result in self.daily_results])

        return {
            "cum_num_positives": cum_num_positives,
            "cum_num_identified": cum_num_identified,
            "cum_sensitivity": cum_sensitivity,
            "cum_num_tests": cum_num_tests,
        }
