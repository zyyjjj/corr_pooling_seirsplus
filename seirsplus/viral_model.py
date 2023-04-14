import numpy as np
from models import ExtSEIRSNetworkModel
from typing import List

# initial viral load by state
# TODO: later enable sampling from a distribution
# either way, look into literature (Cleary / Brault) to decide values
INIT_VL_BY_STATE = {
    "S": 0,
    "E": 3, # by Larremore et al. 2021
    "I_pre": 6, # taking the average of E and I_sym/I_asym
    "I_sym": 9, # to start with
    "I_asym": 9, # to start with, though could consider increasing
    "H": 6, # to start with, keep same as R, though could consider increasing
    "R": 6, # by Larremore et al. 2021
    "F": 0, # ?
    "Q_S": 0,
    "Q_E": 3,
    "Q_pre": 6,
    "Q_sym": 9,
    "Q_asym": 9,
    "Q_R": 6
}

# what Larremore et la. 2021 says about viral load progression
# (t_0, 3), where t_0 ~ Unif[2.5, 3.5]
# (t_peak, V_peak), where t_peak - t_0 ~ 0.5 + Gamma(1.5), capped at 3; V_peak ~ Unif[7, 11]
# (t_f, 6), where t_f - t_peak ~ Unif[4, 9]
# t_symptoms − t_peak ∼ unif[0, 3] -- symptom onset happens after VL peaks
# let's assume that VL is increasing in state I_pre and decreasing in states I_sym and I_asym



# slope of log10 VL progression
# key is transitionType, keep consistent with existing notation
# TODO: later enable sampling from a distribution
# NOTE: the slopes for state 'X' and state 'QX' can be assumed the same

# these values are preliminary values, could consider improving later
# TODO: also make sure to floor self.VL at 0; should I ceiling it at a max value?
VL_SLOPES = { 
    "S": 0.,
    "E": 1., 
    "I_pre": 1., 
    "I_sym": -1., # ?
    "I_asym": -1., # ?
    "H": -1.,
    "R": -1., 
    "F": 0, 
    "Q_S": 0., 
    "Q_E": 1., 
    "Q_pre": 1., 
    "Q_sym": -1., 
    "Q_asym": -1., 
    "Q_R": -1. 
}
# The slopes and the initial state values should be consistent in the way that 
# initial VL at S1 + slope at S1 * (average time between S1 and S2) = initial VL at S2
# avg time between two states should be inverse of the transition rate
# how did they set transition rate in the model?
# TODO: later also implement max_vl to prevent someone's VL from growing infinitely big

class ViralExtSEIRNetworkModel(ExtSEIRSNetworkModel):
    # to add VL and time-varying transmissibility
    # :
    # 1. have a viral load array tracking every person's VL over time (cts or discrete?)
    # 2. have beta (transmission rate) depend on VL

    def __init__(self, G, beta, sigma, lamda, gamma, 
                    init_VL = INIT_VL_BY_STATE, VL_slopes = VL_SLOPES,
                    gamma_asym=None, eta=0, gamma_H=None, mu_H=0, alpha=1.0, xi=0, mu_0=0, nu=0, a=0, h=0, f=0, p=0,             
                    beta_local=None, beta_asym=None, beta_asym_local=None, beta_pairwise_mode='infected', delta=None, delta_pairwise_mode=None,
                    G_Q=None, beta_Q=None, beta_Q_local=None, sigma_Q=None, lamda_Q=None, eta_Q=None, gamma_Q_sym=None, gamma_Q_asym=None, alpha_Q=None, delta_Q=None,
                    theta_S=0, theta_E=0, theta_pre=0, theta_sym=0, theta_asym=0, phi_S=0, phi_E=0, phi_pre=0, phi_sym=0, phi_asym=0,    
                    psi_S=0, psi_E=1, psi_pre=1, psi_sym=1, psi_asym=1, q=0, isolation_time=14,
                    initE=0, initI_pre=0, initI_sym=0, initI_asym=0, initH=0, initR=0, initF=0,        
                    initQ_S=0, initQ_E=0, initQ_pre=0, initQ_sym=0, initQ_asym=0, initQ_R=0,
                    o=0, prevalence_ext=0,
                    transition_mode='exponential_rates', node_groups=None, store_Xseries=False, seed=None):
        r"""
        Initialize the model class. 
        How it's different from ExtSEIRNetworkModel: # TODO: add more documentation
        """

        super().__init__() 

        # node currently undergoing a transition; "locked" and does not participate in screening
        self.transitionNode = None 

        self.init_VL = init_VL
        self.VL_slopes = VL_slopes
        self.current_VL = np.zeros(self.numNodes)

        # VL value at the beginning of the current state
        self.current_state_init_VL = np.zeros(self.numNodes)

        self.initialize_VL_and_beta()
    

    def initialize_VL_and_beta(self):
        r"""
        Initialize the log10 viral load of each node at the start of the simulation.
        This function should be called in __init__().

        The rationale for keeping this separate from update_VL() is the following:
        if someone starts from an infectious state whose previous state also should have positive viral load, 
            then the initialized all-0 self.current_state_init_VL doesn't make sense;
            instead we should generate a positive hypothetical last_state_log10_VL value
            e.g., we sample the initial VL from some distribution like Brault's GMM
        This logic is a bit too messy to implement in update_VL. So let's keep it separate.
        In other words, update_VL would not involve any kind of sampling of the starting VL
        """

        # for each node, get / sample initial viral load
        for node, state in enumerate(self.X):
            self.current_VL[node] = self.init_VL[state]
            self.current_state_init_VL[node] = self.init_VL[state]

        # TODO: set beta according to viral load
        # self.update_beta_given_VL()


    def update_VL(
        self,
        nodes_to_include: List = None,
        nodes_to_exclude: List = None
        ):

        r"""
        Update the log10 viral load of each node depending on their current state (self.X), 
        and the time they have been in their current state (self.timer_state)

        This function should be called every time a state transition happens (model.run_one_iteration()),
        and every time a screening takes place (run_tti_one_day()).

        The overall logic of updating the VL: 
        - the VL at the end of last state defines a "baseline value"
        - the current state maps to a slope of increase/decrease, either deterministic or sampled
            (in initial stages, should be increasing; if recovering, should be decreasing)
        - update goes like: new_VL = baseline_VL + slope * timer_state[i]
        - if a state transition occurs, update self.current_state_init_VL (here? or in run_one_iteration?)

        Args:
            nodes_to_include: list of nodes to update VL for; 
            nodes_to_exclude: list of nodes to not update VL for;
                for both of them, if None, assume update VL for everyone 
        """

        # Option 1:
        # update VL and beta for everybody at every individual transition (not much added value)
        # update VL for everybody (except self.transitionNode) on integer days

        # Option 2:
        # or we just update VL and beta for the perosn undergoing the transition
        # and update VL for everybody (except self.transitionNode) on integer days

        # TODO: if the transition type is different, e.g., Isym to R or H
        # the slope could be different; we choose to ignore that for now

        # 
        # TODO: if this function is called for screening
        # where a node's transition happens after screening but the state is already updated, how do we deal with that?
        # one simple fix is to exclude this node from participating in screening at all
        # like "lock" in multiprocessing
        # to achieve this we need to save the self.transitionNode in run_iteration()

        if nodes_to_include is None:
            nodes_to_include = list(range(self.numNodes))
        if nodes_to_exclude = None:
            nodes_to_exclude = []
        
        nodes_to_update = nodes_to_include - nodes_to_exclude # TODO: check

        for node in nodes_to_update:
            state = self.X[node]
            new_VL_val = self.current_state_init_VL[node] + self.VL_slopes[state] * self.timer_state[node]
            self.current_VL[node] = max(0, new_VL_val) # TODO: is this necessary? because it's log it's ok to have negatives? 


    def update_beta_given_VL(
        self, 
        nodes_to_include: List = None,
        nodes_to_exclude: List = None
        ):
        """
        Update the transmission rates (both self.beta and self.beta_Q) given the current viral load.
        The logic is TBD

        This function should be called every time after update_VL() is called
        or maybe combine the two into one function

        Args:
            nodes_to_include: list of nodes to update VL for; 
            nodes_to_exclude: list of nodes to not update VL for;
                for both of them, if None, assume update VL for everyone 
        """

        # TODO: think through what logic to use
        # to start with, have beta proportional to log10 VL
        
        # the default method for initializing beta: R0 / infectiousPeriod, 
        # where infectiousPeriod = presymptomatic period + symptomatic period

        # do we have to include it?
        # If we don't, the correlation in infection status is induced by network structure only
        # If we do, the correlation is induced by network structure + prop-to-VL transmission intensity

        pass


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
            assert(self.X[transitionNode] == self.transitions[transitionType]['currentState'] and self.X[transitionNode]!=self.F), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+str(transitionType)+"."
            self.X[transitionNode] = self.transitions[transitionType]['newState']

            self.testedInCurrentState[transitionNode] = False
            self.timer_state[transitionNode] = 0.0 # reset timer, since transitionNode is in a new state

            # directly update VL to the initial VL level of the new state
            # TODO: check this
            self.current_VL[transitionNode] = self.init_VL[self.X[transitionNode]]
            self.current_state_init_VL[transitionNode] = self.current_VL[transitionNode]
            # self.update_beta_given_VL(nodes_to_include = [transitionNode])

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