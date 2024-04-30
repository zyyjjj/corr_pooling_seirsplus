from typing import List
from collections import defaultdict
import numpy

from seirsplus.models import ExtSEIRSNetworkModel

# range for uniform sampling of key parameters for temporal VL progression
# VL_PARAMS = {
#     "symptomatic": {
#         "start_peak": (1, 3), # I_pre
#         "end_peak": (4, 6), # I_sym
#         "start_tail": (11.5, 13.5), # I_sym
#         "end_tail": (15, 17), # R
#         # "peak_height": (7.5, 10),
#         # "tail_height": (3.5, 6)
#         "peak_height": (7, 7),
#         "tail_height": (3, 3.5)
#     },
#     "asymptomatic": {
#         "start_peak": (1, 3), # I_pre
#         "end_peak": (4, 6), # I_asym
#         "start_tail": (11, 11), # I_asym
#         "end_tail": (11, 11), # R
#         # "peak_height": (5, 7),
#         # "tail_height": (3, 3.5)
#         "peak_height": (7, 7),
#         "tail_height": (3, 3.5)
#     }
# }

VL_PARAMS = {
    "symptomatic": {
        # "start_peak": (2.19, 5.26),
        "half_peak": (1,1),
        "start_peak": (3,5),
        "dt_peak": (1, 3), 
        "dt_decay": (7, 10), 
        "dt_tail": (5, 6), 
        "peak_height": (7, 7),
        # "tail_height": (4, 4)
        # "peak_height": (6.5, 6.5),
        "tail_height": (3, 3)
    },
    "asymptomatic": {
        # "start_peak": (2.19, 5.26), 
        "half_peak": (1,1),
        "start_peak": (3,5),
        "dt_peak": (1, 3), 
        "dt_decay": (7, 10), 
        "dt_tail": (5, 6), 
        "peak_height": (7, 7),
        # "tail_height": (4, 4)
        # "peak_height": (6.5, 6.5),
        "tail_height": (3, 3)
    }
}

# VL_PARAMS = {
#     "symptomatic": {
#         "start_peak": (1, 1), # I_pre
#         "end_peak": (10, 10), # I_sym
#         "start_tail": (11, 11), # I_sym
#         "end_tail": (15, 15), # R
#         # "peak_height": (3, 5),
#         "peak_height": (4,6), 
#         "tail_height": (-1, -1)
#     },
#     "asymptomatic": {
#         "start_peak": (1, 1), # I_pre
#         "end_peak": (10, 10), # I_sym
#         "start_tail": (11, 11), # I_sym
#         "end_tail": (15, 15), # R
#         # "peak_height": (3, 5),
#         "peak_height": (4,6), 
#         "tail_height": (-1, -1)
#     }
# }


class ViralExtSEIRNetworkModel(ExtSEIRSNetworkModel):
    r"""
    A class to simulate the extended SEIRS Stochastic Network Model, where
    each node has a viral load that is dynamically updated as time progresses 
    and they transition through different states. Viral loads are represented
    as log-10 values.
     
    Additional params compared to ExtSEIRSNetworkModel:
        VL_params: dict of VL progression parameters for each node
    
    Additional variables compared to ExtSEIRSNetworkModel:
        self.current_VL: numpy array of size self.numNodes,
            tracks the current log10 viral load of each node
        self.transitionNode: int, tracks the node that is currently undergoing
            transition. At daily screenings, our SimulationRunner computes the 
            updated viral load of all nodes eligible to participate in testing. 
            To avoid conflicting updates, we prevent the node currently undergoing 
            transition from being updated by the SimulationRunner.
    """

    def __init__(self, G, G_weighted, beta, sigma, lamda, gamma, households_dict,
                    VL_params = VL_PARAMS,
                    gamma_asym=None, eta=0, gamma_H=None, mu_H=0, alpha=1.0, xi=0, mu_0=0, nu=0, a=0, h=0, f=0, p=0,             
                    beta_local=None, beta_asym=None, beta_asym_local=None, beta_pairwise_mode='infected', delta=None, delta_pairwise_mode=None,
                    G_Q=None, beta_Q=None, beta_Q_local=None, sigma_Q=None, lamda_Q=None, eta_Q=None, gamma_Q_sym=None, gamma_Q_asym=None, alpha_Q=None, delta_Q=None,
                    theta_S=0, theta_E=0, theta_pre=0, theta_sym=0, theta_asym=0, phi_S=0, phi_E=0, phi_pre=0, phi_sym=0, phi_asym=0,    
                    psi_S=0, psi_E=1, psi_pre=1, psi_sym=1, psi_asym=1, q=0, isolation_time=14,
                    initE=0, initI_pre=0, initI_sym=0, initI_asym=0, initH=0, initR=0, initF=0,        
                    initQ_S=0, initQ_E=0, initQ_pre=0, initQ_sym=0, initQ_asym=0, initQ_R=0,
                    o=0, prevalence_ext=0,
                    transition_mode='exponential_rates', node_groups=None, store_Xseries=False, seed=None, verbose=0):

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
        
        self.transitionNode = None 

        self.G_weighted = G_weighted
        self.households_dict = households_dict

        self.current_VL = -numpy.ones(self.numNodes)
        self.VL_params = VL_params
        self.VL_over_time = {
            "time_points": [],
            "VL_time_series": [[] for _ in range(self.numNodes)]
        }
        self.peak_VLs = []
        self.time_in_pre_state = []
        self.transitions_log = []

        # start time of infections; initialize as large numbers for those never infected
        self.infection_start_times = numpy.ones(self.numNodes)*100000
        self.isolation_start_times = numpy.ones(self.numNodes)*100000

        # assign (a)symptomatic nodes; this is independent from built-in parameter `a`
        symptomatic_rv = numpy.random.rand(self.numNodes, 1)
        self.symptomatic_by_node = symptomatic_rv > 0.3 

        # draw VL progression params for each node
        self.VL_params_by_node = {}
        for node in range(self.numNodes):
            key = "symptomatic" if self.symptomatic_by_node[node] else "asymptomatic"
            critical_time_points = [numpy.random.uniform(bounds[0], bounds[1])
                    for bounds in list(self.VL_params[key].values())[:5]]
            # if we are sampling time intervals rather than time points, convert to time points
            for i in range(1,5):
                critical_time_points[i] += critical_time_points[i-1]
            peak_plateau_height = numpy.random.uniform(
                self.VL_params[key]["peak_height"][0], self.VL_params[key]["peak_height"][1]
            )
            tail_height = numpy.random.uniform(
                self.VL_params[key]["tail_height"][0], self.VL_params[key]["tail_height"][1]
            )
            self.VL_params_by_node[node] = {
                "critical_time_points": critical_time_points,
                "peak_plateau_height": peak_plateau_height,
                "tail_height": tail_height
            }
            # update and override transition rate parameters
            self.sigma[node] = 1 / critical_time_points[0] # E to Ipre
            self.lamda[node] = 1 / (critical_time_points[2]-critical_time_points[0]) # Ipre to Isym
            self.gamma[node] = 1 / (critical_time_points[4]-critical_time_points[2]) # Isym to R
        
        self.initialize_VL()

        self.verbose = verbose

        self.sec_infs_household = defaultdict(int)
        self.sec_infs_non_household = defaultdict(int)

        self.individual_history = defaultdict(lambda: defaultdict(list))
        self.blame_history = defaultdict(list)

                    
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
        """

        for node, state in enumerate(self.X):
            if state[0] in (2, 12): # E, Q_E:
                self.current_VL[node] = 0
                self.infection_start_times[node] = 0
            elif state[0] in (3, 13): # I_pre, Q_pre
                self.current_VL[node] = self.VL_params_by_node[node]["tail_height"]
                self.infection_start_times[node] = - self.VL_params_by_node[node]["critical_time_points"][0] 
            elif state[0] in (4, 14): # I_sym, Q_sym
                self.current_VL[node] = self.VL_params_by_node[node]["peak_plateau_height"]
                self.infection_start_times[node] = - self.VL_params_by_node[node]["critical_time_points"][2] 

        self.save_VL_timeseries()


    def update_VL(
        self,
        nodes_to_include: List = None,
        nodes_to_exclude: List = None
        ):
        r"""
        Update the log10 viral load of each node depending on the time since 
        their infection started. The viral load of each node, if infected,
        progresses over time following a piecewise linear function with parameters
        specified in self.VL_params_by_node.
        This function is called every time a state transition happens (model.run_iteration()).

        Args:
            nodes_to_include: list of nodes to update VL for; default to everyone
            nodes_to_exclude: list of nodes to not update VL for; default to empty
        """

        if nodes_to_include is None:
            nodes_to_include = list(range(self.numNodes))
        if nodes_to_exclude == None:
            nodes_to_exclude = []
        
        nodes_to_update = list(set(nodes_to_include) - set(nodes_to_exclude))

        for node in nodes_to_update:
            half_peak_time, start_peak_time, end_peak_time, start_tail_time, end_tail_time = self.VL_params_by_node[node]["critical_time_points"]
            peak_plateau_height = self.VL_params_by_node[node]["peak_plateau_height"]
            tail_height = self.VL_params_by_node[node]["tail_height"]

            # if node is not infected, self.t - self.infection_start_times[node] is negative
            if self.infection_start_times[node] < self.t:
                sample_time = self.t - self.infection_start_times[node]
                if sample_time < half_peak_time:
                    vl = tail_height / half_peak_time * sample_time
                elif sample_time < start_peak_time:
                    vl = tail_height + (peak_plateau_height - tail_height) / (start_peak_time - half_peak_time) * (sample_time - half_peak_time)
                elif sample_time < end_peak_time:
                    vl = peak_plateau_height
                elif sample_time < start_tail_time:
                    vl = peak_plateau_height - (peak_plateau_height - tail_height) / (start_tail_time - end_peak_time) * (sample_time - end_peak_time)
                elif sample_time < end_tail_time:
                    vl = tail_height
                else:
                    vl = -1
                
                self.current_VL[node] = vl


    def assign_infector_credit(self, infected):

        neighbors = list(self.G._adj[infected].keys())
        household_neighbors = set(self.households_dict[infected]) - set([infected])
        non_household_neighbors = list(set(neighbors) - set(household_neighbors))

        print(f"infected: {infected}, transmissionTerms_I: {self.transmissionTerms_I[infected]}, transmissionTerms_Q: {self.transmissionTerms_Q[infected]}, household_neighbors: {household_neighbors}, non_household_neighbors: {non_household_neighbors}")
        print(f"Household member states: {[self.X[j][0] for j in household_neighbors]}")
        print(f"Non-household member states: {[self.X[j][0] for j in non_household_neighbors]}")


        if len(neighbors) == 0:
            return
        
        total_contribution = {}
        total_contribution_Q = {}
        
        for j in household_neighbors:
            if self.X[j] in (self.I_pre, self.I_sym, self.I_asym):
                contribution = self.A[infected, j]**2 * self.beta[j] / self.transmissionTerms_I[infected]
                self.sec_infs_household[j] += contribution.item()
                total_contribution[j] = round(contribution.item(), 3)
            elif self.X[j] in (self.Q_pre, self.Q_sym, self.Q_asym):
                contribution = self.A[infected, j]**2 * numpy.divide(
                    self.beta_Q[j], self.transmissionTerms_Q[infected], 
                    out=numpy.array([0.]), where=self.transmissionTerms_Q[infected]!=0)
                total_contribution_Q[j] = round(contribution.item(), 3)
        
        for j in non_household_neighbors:
            if self.X[j] in (self.I_pre, self.I_sym, self.I_asym):
                contribution = self.A[infected, j]**2 * self.beta[j] / self.transmissionTerms_I[infected]
                self.sec_infs_non_household[j] += contribution.item()
                total_contribution[j] = round(contribution.item(), 3)
            elif self.X[j] in (self.Q_pre, self.Q_sym, self.Q_asym):
                contribution = self.A[infected, j]**2 * numpy.divide(
                    self.beta_Q[j], self.transmissionTerms_Q[infected], 
                    out=numpy.array([0.]), where=self.transmissionTerms_Q[infected]!=0)
                total_contribution_Q[j] = round(contribution.item(), 3)
        
        # log infected info in individual history
        self.individual_history[infected]["infected"].append(
            {
                "time": self.t,
                "state": self.X[infected],
                "VL": self.current_VL[infected],
                "household_members_states": [(j, self.X[j][0]) for j in household_neighbors], 
                "non_household_members_states": [(j, self.X[j][0]) for j in non_household_neighbors],
            }
        )

        # assign blame to possible infectors
        for contribution_log in [total_contribution, total_contribution_Q]:
            for j, c in contribution_log.items(): 
                self.blame_history[j].append(
                    {
                        "time": self.t,
                        "state": self.X[j][0],
                        "VL": self.current_VL[j],
                        "infected_node": infected,
                        "contribution": c,
                        "edge_weight": self.A[infected, j]
                    }
                )
        

        print(f"Infected node {infected} got contribution from infectious contacts {total_contribution} and quarantined contacts {total_contribution_Q}")


    def run_iteration(self, max_dt=None):
        r"""
        Run one iteration of the model. 
        - If sum of all nodes' propensities is 0, advance the time by 0.01 and do 
            not run any state transitions.
        - If sum of all nodes' propensities > 0, but the time until the next 
            event exceeds `max_dt`, do not run any state transitions except
            for releasing nodes who have been isolated for `self.isolation_time`
            from isolation.
        - If sum of all nodes' propensities >0 and the time until the next 
            event does not exceed `max_dt`, execute the state transition and 
            save the current VL and transition information.
        """

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

        if self.verbose >= 2:
            print("calling model.run_iteration(), time: ", self.t)

        if self.verbose >= 2:
            print("    Nodes with transition propensities:")
            for i, prop in enumerate(propensities):
                if sum(prop)>0:
                    p_list = []
                    for i_p, p in enumerate(prop):
                        if p>0:
                            p_list.append((transitionTypes[i_p], p))
                    print(f"        node{i}, in state {self.X[i]}, propensity {p_list}")
            print(
                "    propensities.sum(): ", propensities.sum())

        if(propensities.sum() > 0): # NOTE: transition only happens if someone has the propensity to do so, not according to discrete time steps

            if self.verbose >= 2:
                print("    propensities sum to >0")

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calculate alpha
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            propensities_flat   = propensities.ravel(order='F') # flatten in column-major order, p_f[num_Nodes*transitionTypeIdx+nodeIdx]=propensity[nodeIdx, transitionTypeIdx]
            cumsum              = propensities_flat.cumsum()
            alpha               = propensities_flat.sum() # NOTE: rate of the most immediate next transition among all

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute the time until the next event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            tau = (1/alpha)*numpy.log(float(1/r1)) # convert uniform to exponential RV
            if self.verbose >= 2:
                print("    tau: ", tau)

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
                    self.individual_history[isoNode]["exited_isolated"].append(
                        {
                            "time": self.t,
                            "state": self.X[isoNode][0],
                            "VL": self.current_VL[isoNode]
                        }
                    )
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

            if self.verbose >= 1:
                print("    Nodes with transition propensities:")
                for i, prop in enumerate(propensities):
                    if i == transitionNode:
                        p_list = []
                        for i_p, p in enumerate(prop):
                            if p>0:
                                p_list.append((transitionTypes[i_p], p))
                        print(f"        node{i}, in state {self.X[i]}, propensity {p_list}")
                    
            if self.verbose >= 1:
                if transitionType in ("EtoIPRE", "QEtoQPRE"):
                    print(f"-- node {transitionNode} is transitioning {transitionType} at time {self.t} with timer_state: {self.timer_state[transitionNode]}; 1/sigma: {1.0/self.sigma[transitionNode]}; VL: {self.current_VL[transitionNode]}")
                    if self.timer_state[transitionNode] < 1.0/self.sigma[transitionNode]:
                        print(f"** ERROR01: timer_state is less than 1/sigma for node {self.transitionNode} at time {self.t} for transition {transitionType}")
                        # print(f"propensity: {propensities[transitionNode, transitionTypes.index(transitionType)]}, sum of all propensities: {alpha}")
                if transitionType in ("IPREtoISYM", "IPREtoIASYM", "QPREtoQSYM", "QPREtoQASYM"):
                    print(f"-- node {transitionNode} is transitioning {transitionType} at time {self.t} with timer_state: {self.timer_state[transitionNode]}; 1/lamda: {1.0/self.lamda[transitionNode]}; VL: {self.current_VL[transitionNode]}")
                    if self.timer_state[transitionNode] < 1.0/self.lamda[transitionNode]:
                        print(f"** ERROR01: timer_state is less than 1/lamda for node {self.transitionNode} at time {self.t} for transition {transitionType}")
                        # print(f"propensity: {propensities[transitionNode, transitionTypes.index(transitionType)]}, sum of all propensities: {alpha}")
                if transitionType in ("ISYMtoR", "QSYMtoR"):
                    print(f"-- node {transitionNode} is transitioning {transitionType} at time {self.t} with timer_state: {self.timer_state[transitionNode]}; 1/gamma: {1.0/self.gamma[transitionNode]}; VL: {self.current_VL[transitionNode]}")
                    if self.timer_state[transitionNode] < 1.0/self.gamma[transitionNode]:
                        print(f"** ERROR01: timer_state is less than 1/gamma for node {self.transitionNode} at time {self.t} for transition {transitionType}")
                        # print(f"propensity: {propensities[transitionNode, transitionTypes.index(transitionType)]}, sum of all propensities: {alpha}")
                    if self.current_VL[self.transitionNode] < 0:
                        if self.symptomatic_by_node[self.transitionNode]:
                            print(f"** ERROR02: VL is negative for node {self.transitionNode} at time {self.t} for transition {transitionType}, current timer_state: {self.timer_state[self.transitionNode]}, current VL: {self.current_VL[self.transitionNode]}")
                        # print(f"** ERROR02: VL is negative for node {self.transitionNode} at time {self.t} for transition {transitionType}, current timer_state: {self.timer_state[self.transitionNode]}")


            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform updates triggered by rate propensities:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.update_VL()

            assert(self.X[transitionNode] == self.transitions[transitionType]['currentState'] and self.X[transitionNode]!=self.F), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+str(transitionType)+"."
            self.X[transitionNode] = self.transitions[transitionType]['newState']

            if transitionType in ("IPREtoISYM", "IPREtoIASYM", "QPREtoQSYM", "QPREtoQASYM"):
                self.time_in_pre_state.append(self.timer_state[transitionNode][0])

            self.testedInCurrentState[transitionNode] = False
            self.timer_state[transitionNode] = 0.0 # reset timer, since transitionNode is in a new state

            if(transitionType == 'StoE' or transitionType == 'QStoQE'):
                self.current_VL[transitionNode] = 0

            self.save_VL_timeseries()

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # Save information about infection events when they occur:
            if(transitionType == 'StoE' or transitionType == 'QStoQE'):
                transitionNode_GNbrs  = list(self.G[transitionNode].keys())
                transitionNode_GQNbrs = list(self.G_Q[transitionNode].keys())
                self.infectionsLog.append({ 't':                            self.t,
                                            'infected_node':                transitionNode,
                                            'infection_type':               transitionType,
                                            'infected_node_degree':         self.degree[transitionNode],
                                            'household_members':            self.households_dict[transitionNode],
                                            'household_members_states':     self.X[self.households_dict[transitionNode]].flatten()
                                            # 'local_contact_nodes':          transitionNode_GNbrs,
                                            # 'local_contact_node_states':    self.X[transitionNode_GNbrs].flatten(),
                                            # 'isolation_contact_nodes':      transitionNode_GQNbrs,
                                            # 'isolation_contact_node_states':self.X[transitionNode_GQNbrs].flatten() 
                                        })
                self.infection_start_times[transitionNode] = self.t
                self.assign_infector_credit(transitionNode)
            if transitionType in ("IPREtoISYM", "IPREtoIASYM", "QPREtoQSYM", "QPREtoQASYM"):
                self.peak_VLs.append(self.current_VL[transitionNode])

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if(transitionType in ['EtoQE', 'IPREtoQPRE', 'ISYMtoQSYM', 'IASYMtoQASYM', 'ISYMtoH']):
                self.set_positive(node=transitionNode, positive=True)
            
            transition_info_tmp = {"t": self.t, "transitionNode": self.transitionNode, "transitionNodeVL": self.current_VL[self.transitionNode], "transitionType": transitionType}
            if self.verbose >= 1:
                print(transition_info_tmp)
                print(f"propensity: {propensities[transitionNode, transitionTypes.index(transitionType)]}, sum of all propensities: {alpha}")

            self.transitions_log.append(transition_info_tmp)

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
            self.individual_history[isoNode]["exited_isolated"].append(
                {
                    "time": self.t,
                    "state": self.X[isoNode][0],
                    "VL": self.current_VL[isoNode]
                }
            )

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