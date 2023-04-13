# sim loop for tti sim with pooled tests
from __future__ import division
from typing import Dict, Tuple, List

import pickle
import time

import numpy
from pooled_test import OneStageGroupTesting
from viral_model import ViralExtSEIRNetworkModel

# from screening_assignment import assign # TODO: in progress, add when finished


class SimulationRunner:
    """Runner class for SEIRS+ simulation with pooled testing."""
    def __init__(
        self,
        model: ViralExtSEIRNetworkModel, 
        T: int, 
        screening_assignment: Dict[Tuple[List]],  # TODO: Jiayue to provide this
        seed: int,
        max_dt: int = None,
        cadence_cycle_length: int = 28, 
        output_path: str = None,
    ):
        r"""Initialize the simulation runner. 
        
        Args:
            model: a `ViralExtSEIRNetworkModel` 
            T: duration (days) of the simulation
            screening_assignment: dictionary that specifies, for each screening group,
                the days in the cadence cyle on which the group is screened,
                and the IDs of the individuals in the group. For example,
                an assignment splitting 35 people evenly across 7 groups may look like
                    {
                        0: ([0,7,14,21], [0,1,2,3,4]),
                        1: ([1,8,15,22], [5,6,7,8,9]), ...
                        6: ([6,13,20,27], [30,31,32,33,34])
                    }
            seed: random seed
            max_dt: seemingly useless model parameter that I prefer not to touch
            cadence_cycle: default 4 weeks
            
        Returns: 
            None.
        """

        self.model = model
        self.T = T
        self.screening_assignment = screening_assignment
        self.seed = seed
        self.max_dt = max_dt
        self.cadence_cycle_length = cadence_cycle_length
        self.output_path = output_path

        self.day_to_screening_group = {}
        for group_id, screening_info in screening_assignment: 
            for day in screening_info[0]: 
                self.day_to_screening_group[day] = group_id

        self.isolation_states = [model.Q_S, model.Q_E, model.Q_pre, model.Q_sym, model.Q_asym, model.Q_R]

        self.results = {} # TODO


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
        screening_group = self.screening_assignment[screening_group_id][1] # list of IDs
        for test_subject in screening_group:
            if nodeStates[test_subject] in self.isolation_states:
                screening_group.remove(test_subject)
            
        self.model.update_VL(nodes_to_exclude = [self.transitionNode]) # TODO: to implement; update VL for everyone except self.model.transitionNode
        self.model.update_beta_given_VL(n) # TODO: to implement

        # TODO: assign screening_group to individual pools, 
        # return a nested list called `screening_group_pools`
        # also fetch the viral loads and put in nested list `screening_group_VL`
        # individual_pools = assign(screening_group, self.model.VL)

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
