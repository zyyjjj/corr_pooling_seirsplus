# sim loop for tti sim with pooled tests
from __future__ import division
from typing import Optional
import random
import time
import pickle
from networkx import Graph
import numpy as np

from pooled_test import OneStageGroupTesting
from viral_model import ViralExtSEIRNetworkModel
from assignment import embed_nodes, get_equal_sized_clusters

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
        output_path: Optional[str] = None,
    ):
        r"""Initialize the simulation runner. 
        
        Args:
            model: A `ViralExtSEIRNetworkModel` object.
            T: Duration (in days) of the simulation.
            num_groups: Number of testing groups to split the population into.
            pool_size: Size of the pooled tests in one-stage group testing.
            seed: The random seed for the simulation. 
            output_path: The directory to save the simulation results to.
            
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
            cluster_size=self.num_groups # TODO: this is also not consistent -- use self.model.numNodes // self.num_groups + 1?
        )
        if pooling_strategy not in ['naive', 'correlated']:
            raise NotImplementedError(f"Pooling strategy {pooling_strategy} not implemented.")
        self.pooling_strategy = pooling_strategy
        self.output_path = output_path

        self.isolation_states = [
            model.Q_S, 
            model.Q_E, 
            model.Q_pre, 
            model.Q_sym, 
            model.Q_asym, 
            model.Q_R
        ]
        
        # initialize results
        self.results = {} # TODO


    def get_groups(self, graph: Graph, cluster_size: int):
        """Get the screening groups or pools for the simulation.
        
        Args:
            graph: The networkx graph object.
            cluster_size: The size of groups or pools to split the population into. 
        """
        embedding, node2vec_model = embed_nodes(graph)
        clusters = get_equal_sized_clusters(
            X=embedding, 
            model=node2vec_model,
            graph=graph,
            cluster_size=cluster_size,
        )  # dict of node_id to cluster id
        cluster_ids = list(set(clusters.values())) # unique list of cluster ids
        groups = {
            i: [x for x,v in clusters.items() if v == i] 
            for i in cluster_ids
        }  # dict of cluster ids as the keys and the node ids as the values
        return groups


    def run_screening_one_day(self, screening_group_id: int):
        """
        Test non-isolated individuals in one screening_group during one day.

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

        nodeStates = self.model.X.flatten()

        # test those in the screening_group and *not isolated*
        screening_group = self.screening_assignment[screening_group_id] # list of IDs
        screening_group = [
            x for x in screening_group 
            if nodeStates[x] not in self.isolation_states
        ]
            
        # update VL for everyone except self.model.transitionNode
        self.model.update_VL(nodes_to_exclude=[self.transitionNode]) 
        
        # divide individuals in screening group into pools according to pooling strategy
        # store pooling result and viral loads in nested lists
        if self.pooling_strategy == 'correlated':
            pools = self.get_groups(
                graph=self.model.G(screening_group), 
                cluster_size=self.pool_size
            )
            pools = [v for _, v in pools.items()] # a list of lists 
        elif self.pooling_strategy == 'naive':
            random.shuffle(screening_group)
            pools = [
                screening_group[i: i + self.pool_size] 
                for i in range(0, len(screening_group), self.pool_size)
            ]
        viral_loads = [[self.model.current_VL[x] for x in pool] for pool in pools]
        group_testing = OneStageGroupTesting(ids=pools, viral_loads=viral_loads)
        test_results, diagnostics = group_testing.run_one_stage_group_testing(seed=self.seed)
        
        # TODO: pass test_results to update isolation status in self.model
        for pool_idx, pool in enumerate(pools):
            for individual_idx, individual in enumerate(pool):
                if test_results[pool_idx][individual_idx] == 1:
                    self.model.set_positive(individual, True)
                    self.model.set_isolation(individual, True)
        
                    # TODO: when we isolate someone through testing
                    # make sure to change their state in model from X to QX

        # TODO: save diagnostics in self.results, key is day, val is diagnostics dict
        # TODO: save self.results to self.output_path
        


    def run_simulation(self):
        r"""
        Run screening for the full duration self.T.
        """

        dayOfLastIntervention = 0
        self.model.tmax  = self.T
        running = True
        
        while running:

            # first run a model iteration, i.e., one transition
            running = self.model.run_iteration()

            # make sure we don't skip any days due to the transition
            # implement testing on each day
            while dayOfLastIntervention <= int(self.model.t):
                cadenceDayNumber = int(dayOfLastIntervention % self.cadence_cycle_length)
                screening_group_id = self.day_to_screening_group[cadenceDayNumber]
                self.run_screening_one_day(screening_group_id)
                dayOfLastIntervention += 1
            
            if dayOfLastIntervention > self.T:
                running = False


# sketch of overall loop
# this will be in a separate .py file I think

def run_full_loop(T):

    # create model class

    # generate screening assignment groups

    # create screening class passing in (model, screening_assignment, T)
    # call screening.run_screening_full()
