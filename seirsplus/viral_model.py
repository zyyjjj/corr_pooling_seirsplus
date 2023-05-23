from typing import List

import numpy

from seirsplus.models import ExtSEIRSNetworkModel

# initial log10 viral load by state
# TODO: later enable sampling from a distribution
INIT_VL_BY_STATE = {
    1: -1, # S
    2: 0, # E, by Larremore et al. 2021
    3: 6, # I_pre, taking the average of E and I_sym/I_asym
    4: 9, # I_sym, to start with
    5: 9, # I_asym, to start with, though could consider increasing
    6: 6, # H, to start with, keep same as R, though could consider increasing
    7: 6, # R, by Larremore et al. 2021
    8: -1, # F, ?
    11: 0, # Q_S
    12: 3, # Q_E
    13: 6, # Q_pre
    14: 9, # Q_sym
    15: 9, # Q_asym
    17: 6 # Q_R
}
"""
what Larremore et la. 2021 says about viral load progression
(t_0, 3), where t_0 ~ Unif[2.5, 3.5]
(t_peak, V_peak), where t_peak - t_0 ~ 0.5 + Gamma(1.5), capped at 3; V_peak ~ Unif[7, 11]
(t_f, 6), where t_f - t_peak ~ Unif[4, 9]
t_symptoms − t_peak ∼ unif[0, 3] -- symptom onset happens after VL peaks
let's assume that VL is increasing in state I_pre and decreasing in states I_sym and I_asym
"""

# slope of log10 VL progression in different states -- TODO: think how to represent 0 viral load? use -1
# TODO: later enable sampling from a distribution
# TODO: to discuss (1) floor at 0? (2) ceiling at some max value? (3) int() when passing into group testing?
VL_SLOPES = { 
    1: 0., # S
    2: 1., # E
    3: 1., # I_pre
    4: -1., # I_sym, ?
    5: -1., # I_asym?
    6: -1., # H
    7: -1., # R
    8: 0, # F
    11: 0., # Q_S
    12: 1., # Q_E
    13: 1., # Q_pre
    14: -1., # Q_sym
    15: -1., # Q_asym
    17: -1. # Q_R
}
"""
The slopes and the initial state values should be consistent in the way that roughly
initial VL at S1 + slope at S1 * (average time between S1 and S2) = initial VL at S2
avg time between two states should be inverse of the transition rate
"""

class ViralExtSEIRNetworkModel(ExtSEIRSNetworkModel):
    r"""
    A class to simulate the extended SEIRS Stochastic Network Model, where
    each node has a viral load that is dynamically updated as time progresses 
    and they transition through different states. Viral loads are represented
    as log-10 values.
     
    Additional params compared to ExtSEIRSNetworkModel:
        init_VL: dict, initial log10 viral load in each state
        VL_slopes: dict, slope of log10 viral load progression in each state
    
    Additional variables compared to ExtSEIRSNetworkModel:
        self.current_VL: numpy array of size self.numNodes,
            tracks the current log10 viral load of each node
        self.current_state_init_VL: numpy array, 
            tracks the log10 viral load of each node at the start of the current state
        self.transitionNode: int, tracks the node that is currently undergoing
            transition. At daily screenings, our SimulationRunner computes the 
            updated viral load of all nodes eligible to participate in testing. 
            To avoid conflicting updates, we prevent the node currently undergoing 
            transition from being updated by the SimulationRunner.
    """

    def __init__(self, G, beta, sigma, lamda, gamma, 
                    init_VL_by_state = INIT_VL_BY_STATE, VL_slopes = VL_SLOPES, VL_ceiling = 12,
                    gamma_asym=None, eta=0, gamma_H=None, mu_H=0, alpha=1.0, xi=0, mu_0=0, nu=0, a=0, h=0, f=0, p=0,             
                    beta_local=None, beta_asym=None, beta_asym_local=None, beta_pairwise_mode='infected', delta=None, delta_pairwise_mode=None,
                    G_Q=None, beta_Q=None, beta_Q_local=None, sigma_Q=None, lamda_Q=None, eta_Q=None, gamma_Q_sym=None, gamma_Q_asym=None, alpha_Q=None, delta_Q=None,
                    theta_S=0, theta_E=0, theta_pre=0, theta_sym=0, theta_asym=0, phi_S=0, phi_E=0, phi_pre=0, phi_sym=0, phi_asym=0,    
                    psi_S=0, psi_E=1, psi_pre=1, psi_sym=1, psi_asym=1, q=0, isolation_time=14,
                    initE=0, initI_pre=0, initI_sym=0, initI_asym=0, initH=0, initR=0, initF=0,        
                    initQ_S=0, initQ_E=0, initQ_pre=0, initQ_sym=0, initQ_asym=0, initQ_R=0,
                    o=0, prevalence_ext=0,
                    transition_mode='exponential_rates', node_groups=None, store_Xseries=False, seed=None):

        super().__init__(G, beta, sigma, lamda, gamma, 
                    gamma_asym=gamma_asym, eta=eta, gamma_H=gamma_H, mu_H=mu_H, alpha=alpha, xi=xi, mu_0=mu_0, nu=nu, a=a, h=h, f=f, p=p,             
                    beta_local=beta_local, beta_asym=beta_asym, beta_asym_local=beta_asym_local, beta_pairwise_mode=beta_pairwise_mode, delta=delta, delta_pairwise_mode=delta_pairwise_mode,
                    G_Q=G_Q, beta_Q=beta_Q, beta_Q_local=beta_Q_local, sigma_Q=sigma_Q, lamda_Q=lamda_Q, eta_Q=eta_Q, gamma_Q_sym=gamma_Q_sym, gamma_Q_asym=gamma_Q_asym, alpha_Q=alpha_Q, delta_Q=delta_Q,
                    theta_S=theta_S, theta_E=theta_E, theta_pre=theta_pre, theta_sym=theta_sym, theta_asym=theta_asym, phi_S=phi_S, phi_E=phi_E, phi_pre=phi_pre, phi_sym=phi_sym, phi_asym=phi_asym,    
                    psi_S=psi_S, psi_E=psi_E, psi_pre=psi_pre, psi_sym=psi_sym, psi_asym=psi_asym, q=q, isolation_time=isolation_time,
                    initE=initE, initI_pre=initI_pre, initI_sym=initI_sym, initI_asym=initI_asym, initH=initH, initR=initR, initF=initF,        
                    initQ_S=initQ_S, initQ_E=initQ_E, initQ_pre=initQ_pre, initQ_sym=initQ_sym, initQ_asym=initQ_asym, initQ_R=initQ_R,
                    o=o, prevalence_ext=prevalence_ext,
                    transition_mode=transition_mode, node_groups=node_groups, store_Xseries=store_Xseries, seed=seed)
        
        # node currently undergoing a transition; "locked" and does not participate in screening
        # TODO: come back to this when integrating into SimulationRunner
        self.transitionNode = None 

        self.init_VL_by_state = init_VL_by_state
        self.VL_slopes = VL_slopes
        self.VL_ceiling = VL_ceiling
        self.current_VL = -numpy.ones(self.numNodes) # TODO: should all be -1
        self.VL_over_time = {
            "time_points": [],
            "VL_time_series": [[] for _ in range(self.numNodes)]
        }
        self.peak_VLs = []
        self.time_in_pre_state = []
        # VL value at the beginning of the current state
        self.current_state_init_VL = -numpy.ones(self.numNodes)

        self.initialize_VL()
    
    def save_VL_timeseries(self):
        r"""
        Record each node's viral load at the transitions throughout the simulation.
        """
        
        self.VL_over_time["time_points"].append(self.t)
        for node in range(self.numNodes):
            self.VL_over_time["VL_time_series"][node].append(self.current_VL[node])

    def initialize_VL(self):
        r"""
        Initialize the log10 viral load of each node at the start of the simulation.
        This function should be called in __init__().

        The rationale for keeping this separate from update_VL() is the following:
        if someone starts from an infectious state whose previous state also 
            has positive viral load, 
            then the initialized all-0 self.current_state_init_VL doesn't make sense;
            instead we should generate a positive hypothetical last_state_log10_VL value
            e.g., we sample the initial VL from some distribution like Brault's GMM
        This logic is a bit too messy to implement in update_VL. So let's keep it separate.
        In other words, update_VL would not involve any kind of sampling of the starting VL
        """

        # for each node, get / sample initial viral load
        for node, state in enumerate(self.X):
            self.current_VL[node] = self.init_VL_by_state[state[0]]
            self.current_state_init_VL[node] = self.init_VL_by_state[state[0]]
        
        self.save_VL_timeseries()


    def update_VL(
        self,
        nodes_to_include: List = None,
        nodes_to_exclude: List = None
        ):

        r"""
        Update the log10 viral load of each node depending on their current state (self.X), 
        and the time they have been in their current state (self.timer_state)

        This function is called 
        - every time a state transition happens (model.run_one_iteration()),
        - and every time a screening takes place (run_tti_one_day()).

        The overall logic of updating the VL: 
        - the VL at the start of the current state defines a "baseline value"
        - the current state maps to a slope of increase/decrease, either deterministic or sampled
            (in initial stages, should be increasing; if recovering, should be decreasing)
        - update goes like: new_VL = baseline_VL + slope * timer_state[i]
        - this also works for new state transitions, since timer_state is reset to 0 
            and the new viral load is set to the init viral load of the new state.

        Args:
            nodes_to_include: list of nodes to update VL for; default to everyone
            nodes_to_exclude: list of nodes to not update VL for; default to empty
        """

        # TODO: if the transition type is different, e.g., Isym to R or H
        # the slope could be different; we choose to ignore that for now

        if nodes_to_include is None:
            nodes_to_include = list(range(self.numNodes))
        if nodes_to_exclude == None:
            nodes_to_exclude = []
        
        nodes_to_update = list(set(nodes_to_include) - set(nodes_to_exclude))

        for node in nodes_to_update:
            state = self.X[node][0]
            
            new_VL_val = self.current_state_init_VL[node] + self.VL_slopes[state] * self.timer_state[node]  # TODO: change this line

            self.current_VL[node] = max(-1, new_VL_val) 
            self.current_VL[node] = min(self.current_VL[node], self.VL_ceiling)
        
        self.save_VL_timeseries()


    def run_iteration(self, max_dt=None):

        max_dt = self.tmax if max_dt is None else max_dt

        if(self.tidx >= len(self.tseries)-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate 2 random numbers uniformly distributed in (0,1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r1 = numpy.random.rand()
        r2 = numpy.random.rand()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calculate propensities
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities, transitionTypes = self.calc_propensities()

        if(propensities.sum() > 0): # NOTE: transition only happens if someone has the propensity to do so, not according to discrete time steps

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calculate alpha
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            propensities_flat   = propensities.ravel(order='F') # flatten in column-major order, p_f[num_Nodes*transitionTypeIdx+nodeIdx]=propensity[nodeIdx, transitionTypeIdx]
            cumsum              = propensities_flat.cumsum()
            alpha               = propensities_flat.sum() # NOTE: rate of the most immediate next transition among all

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute the time until the next event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            tau = (1/alpha)*numpy.log(float(1/r1)) # TODO: understand why log(1/r1) here

            # print(f"Before transition time update, self.t: {self.t}, tau: {tau}")

            if(tau > max_dt):
                # If the time to next event exceeds the max allowed interval,
                # advance the system time by the max allowed interval,
                # but do not execute any events (recalculate Gillespie interval/event next iteration)
                self.t += max_dt
                self.timer_state += max_dt # NOTE: timer_state records how long each node has been in its current state
                # Update testing and isolation timers/statuses
                isolatedNodes = numpy.argwhere((self.X==self.Q_S)|(self.X==self.Q_E)|(self.X==self.Q_pre)|(self.X==self.Q_sym)|(self.X==self.Q_asym)|(self.X==self.Q_R))[:,0].flatten()
                self.timer_isolation[isolatedNodes] = self.timer_isolation[isolatedNodes] + max_dt
                nodesExitingIsolation = numpy.argwhere(self.timer_isolation >= self.isolationTime)
                for isoNode in nodesExitingIsolation:
                    self.set_isolation(node=isoNode, isolate=False)
                # return without any further event execution
                return True
            else:
                self.t += tau
                self.timer_state += tau
            
            # print(f"After transition time update, self.t: {self.t}")

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute which event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # NOTE: cumsum has size (number of types of transitions) * (number of nodes)
            transitionIdx   = numpy.searchsorted(cumsum,r2*alpha) # NOTE: randomly pick a transition (specific to a node and a transition type)
            transitionNode  = transitionIdx % self.numNodes 
            transitionType  = transitionTypes[ int(transitionIdx/self.numNodes) ]
            
            # save the node currently undergoing a transition, to lock when running screening
            self.transitionNode = transitionNode 

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform updates triggered by rate propensities:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.update_VL()

            assert(self.X[transitionNode] == self.transitions[transitionType]['currentState'] and self.X[transitionNode]!=self.F), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+str(transitionType)+"."
            self.X[transitionNode] = self.transitions[transitionType]['newState']

            self.save_VL_timeseries()
            if transitionType in ("IPREtoISYM", "IPREtoIASYM", "QPREtoQSYM", "QPREtoQASYM"):
                self.time_in_pre_state.append(self.timer_state[transitionNode][0])

            self.testedInCurrentState[transitionNode] = False
            self.timer_state[transitionNode] = 0.0 # reset timer, since transitionNode is in a new state

            # directly update VL to the initial VL level of the new state
            # self.current_VL[transitionNode] = self.init_VL_by_state[self.X[transitionNode][0]]
            if(transitionType == 'StoE' or transitionType == 'QStoQE'):
                self.current_state_init_VL[transitionNode] = self.init_VL_by_state[2] # 2 for state E
                self.current_VL[transitionNode] = self.init_VL_by_state[2]
            else:      
                self.current_state_init_VL[transitionNode] = self.current_VL[transitionNode]

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # Save information about infection events when they occur:
            if(transitionType == 'StoE' or transitionType == 'QStoQE'):
                transitionNode_GNbrs  = list(self.G[transitionNode].keys())
                transitionNode_GQNbrs = list(self.G_Q[transitionNode].keys())
                self.infectionsLog.append({ 't':                            self.t,
                                            'infected_node':                transitionNode,
                                            'infection_type':               transitionType,
                                            'infected_node_degree':         self.degree[transitionNode],
                                            'local_contact_nodes':          transitionNode_GNbrs,
                                            'local_contact_node_states':    self.X[transitionNode_GNbrs].flatten(),
                                            'isolation_contact_nodes':      transitionNode_GQNbrs,
                                            'isolation_contact_node_states':self.X[transitionNode_GQNbrs].flatten() })
            if transitionType in ("IPREtoISYM", "IPREtoIASYM", "QPREtoQSYM", "QPREtoQASYM"):
                self.peak_VLs.append(self.current_VL[transitionNode])

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if(transitionType in ['EtoQE', 'IPREtoQPRE', 'ISYMtoQSYM', 'IASYMtoQASYM', 'ISYMtoH']):
                self.set_positive(node=transitionNode, positive=True)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        else:

            tau = 0.01
            self.t += tau
            self.timer_state += tau

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.tidx += 1
        
        self.tseries[self.tidx]     = self.t
        self.numS[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.S), a_min=0, a_max=self.numNodes)
        self.numE[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.E), a_min=0, a_max=self.numNodes)
        self.numI_pre[self.tidx]    = numpy.clip(numpy.count_nonzero(self.X==self.I_pre), a_min=0, a_max=self.numNodes)
        self.numI_sym[self.tidx]    = numpy.clip(numpy.count_nonzero(self.X==self.I_sym), a_min=0, a_max=self.numNodes)
        self.numI_asym[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.I_asym), a_min=0, a_max=self.numNodes)
        self.numH[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.H), a_min=0, a_max=self.numNodes)
        self.numR[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.R), a_min=0, a_max=self.numNodes)
        self.numF[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.F), a_min=0, a_max=self.numNodes)
        self.numQ_S[self.tidx]      = numpy.clip(numpy.count_nonzero(self.X==self.Q_S), a_min=0, a_max=self.numNodes)
        self.numQ_E[self.tidx]      = numpy.clip(numpy.count_nonzero(self.X==self.Q_E), a_min=0, a_max=self.numNodes)
        self.numQ_pre[self.tidx]    = numpy.clip(numpy.count_nonzero(self.X==self.Q_pre), a_min=0, a_max=self.numNodes)
        self.numQ_sym[self.tidx]    = numpy.clip(numpy.count_nonzero(self.X==self.Q_sym), a_min=0, a_max=self.numNodes)
        self.numQ_asym[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.Q_asym), a_min=0, a_max=self.numNodes)
        self.numQ_R[self.tidx]      = numpy.clip(numpy.count_nonzero(self.X==self.Q_R), a_min=0, a_max=self.numNodes)
        self.numTested[self.tidx]   = numpy.clip(numpy.count_nonzero(self.tested), a_min=0, a_max=self.numNodes)
        self.numPositive[self.tidx] = numpy.clip(numpy.count_nonzero(self.positive), a_min=0, a_max=self.numNodes)
        
        self.N[self.tidx]           = numpy.clip((self.numNodes - self.numF[self.tidx]), a_min=0, a_max=self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update testing and isolation statuses
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        isolatedNodes = numpy.argwhere((self.X==self.Q_S)|(self.X==self.Q_E)|(self.X==self.Q_pre)|(self.X==self.Q_sym)|(self.X==self.Q_asym)|(self.X==self.Q_R))[:,0].flatten()
        self.timer_isolation[isolatedNodes] = self.timer_isolation[isolatedNodes] + tau

        nodesExitingIsolation = numpy.argwhere(self.timer_isolation >= self.isolationTime)
        for isoNode in nodesExitingIsolation:
            self.set_isolation(node=isoNode, isolate=False)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.store_Xseries):
            self.Xseries[self.tidx,:] = self.X.T

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI_pre'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_pre)
                self.nodeGroupData[groupName]['numI_sym'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_sym)
                self.nodeGroupData[groupName]['numI_asym'][self.tidx]   = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_asym)
                self.nodeGroupData[groupName]['numH'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.H)
                self.nodeGroupData[groupName]['numR'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numF'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.F)
                self.nodeGroupData[groupName]['numQ_S'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_S)
                self.nodeGroupData[groupName]['numQ_E'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_E)
                self.nodeGroupData[groupName]['numQ_pre'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_pre)
                self.nodeGroupData[groupName]['numQ_sym'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_sym)
                self.nodeGroupData[groupName]['numQ_asym'][self.tidx]   = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_asym)
                self.nodeGroupData[groupName]['numQ_R'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_R)
                self.nodeGroupData[groupName]['N'][self.tidx]           = numpy.clip((self.nodeGroupData[groupName]['numS'][0] + self.nodeGroupData[groupName]['numE'][0] + self.nodeGroupData[groupName]['numI'][0] + self.nodeGroupData[groupName]['numQ_E'][0] + self.nodeGroupData[groupName]['numQ_I'][0] + self.nodeGroupData[groupName]['numR'][0]), a_min=0, a_max=self.numNodes)
                self.nodeGroupData[groupName]['numTested'][self.tidx]   = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.tested)
                self.nodeGroupData[groupName]['numPositive'][self.tidx] = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.positive)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Terminate if tmax reached or num infections is 0:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.t >= self.tmax or (self.total_num_infected(self.tidx) < 1 and self.total_num_isolated(self.tidx) < 1)):
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        

        return True