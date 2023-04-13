# sim loop for tti sim with pooled tests
from __future__ import division
from typing import Optional

import pickle
import time
import random

import numpy as np

from pooled_test import OneStageGroupTesting
from viral_model import ViralExtSEIRNetworkModel
from assignment import embed_nodes, get_equal_sized_clusters

class SimulationRunner:
    """Runner class for SEIRS+ simulation with pooled testing."""
    def __init__(
        self,
        model: ViralExtSEIRNetworkModel, 
        T: int,
        num_groups: int,
        pool_size: int,
        seed: int,
        max_dt: Optional[int] = None,
        output_path: Optional[str] = None,
    ):
        r"""Initialize the simulation runner. 
        
        Args:
            model: A `ViralExtSEIRNetworkModel` object.
            T: Duration (in days) of the simulation.
            num_groups: Number of testing groups to split the population into.
            pool_size: Size of the pooled tests in one-stage group testing.
            seed: The random seed for the simulation. 
            max_dt: seemingly useless model parameter that I prefer not to touch # TODO: update this
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
            cluster_size=self.num_groups
        )
        self.max_dt = max_dt
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
        """Get the screening groups for the simulation."""
        embedding, node2vec_model = embed_nodes(graph)
        clusters = get_equal_sized_clusters(
            X=embedding, 
            model=node2vec_model,
            graph=graph,
            cluster_size=cluster_size,
        )  # dict of node_id to cluster id
        groups = {
            i: [x for x,v in clusters.items() if v == i] 
            for i in range(cluster_size)
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
            
        self.model.update_VL(nodes_to_exclude=[self.transitionNode]) # TODO: to implement; update VL for everyone except self.model.transitionNode
        self.model.update_beta_given_VL(nodes_to_exclude=[self.transitionNode]) # TODO: to implement

        # TODO: assign screening_group to individual pools, 
        
        # return a nested list called `screening_group_pools`
        # also fetch the viral loads and put in nested list `screening_group_VL`
        # individual_pools = assign(screening_group, self.model.VL)
        pools = self.get_pools(
            graph=self.model.G(screening_group), 
            cluster_size=self.pool_size
        )
        pools = [v for k,v in pools.items()] # a list of lists
        viral_loads = [[self.model.VL[x] for x in pool] for pool in pools]
        group_testing = OneStageGroupTesting(ids=pools, viral_loads=viral_loads)
        test_results, diagnostics = group_testing.run_one_stage_group_testing(seed=self.seed)
        
        # TODO: then call group_testing
        # group_testing = OneStageGroupTesting(ids = screening_group_pools, viral_loads = screening_group_VL)
        # test_results, diagnostics = group_testing.run_one_stage_group_testing(seed=self.seed)
        
        # TODO: pass test_results to update isolation status in self.model
            # model.set_positive(node, True)
            # model.set_isolation(node, True)
        
        # TODO: when we isolate someone through testing
        # make sure to change their state in model from X to QX

        # TODO: save diagnostics in self.results
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
            running = self.model.run_iteration(max_dt=self.max_dt) # NOTE: max_dt is None by default, in which case max_dt is set to model.tmax=T
            
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
