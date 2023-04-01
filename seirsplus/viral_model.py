import numpy as np
from models import ExtSEIRSNetworkModel

# initial viral load by state
# TODO: do we make this constant, or a distribution?
# either way, look into literature (Cleary / Brault) to decide values
INIT_VL_BY_STATE = {
    "S": 0,
    "E": 0,
    "I_pre": 0,
    "I_sym": 0,
    "I_asym": 0,
    "H": 0,
    "R": 0,
    "F": 0,
    "Q_S": 0,
    "Q_E": 0,
    "Q_pre": 0,
    "Q_sym": 0,
    "Q_asym": 0,
    "Q_R": 0
}

# slope of log10 VL progression
# key is transitionType, keep consistent with existing notation
# TODO: constant or distribution? figure out values
VL_SLOPES = { 
    'StoE': 0.1,
    'StoQS': 0.1,
    'EtoIPRE': 0.1,
    'EtoQE': 0.1,
    'IPREtoISYM': 0.1,
    'IPREtoIASYM': 0.1,
    'IPREtoQPRE': 0.1,
    'ISYMtoH': 0.1,
    'ISYMtoR': 0.1,
    'ISYMtoQSYM': 0.1,
    'IASYMtoR': 0.1,
    'IASYMtoQASYM': 0.1,
    'HtoR': 0.1,
    'HtoF': 0.1,
    'RtoS': 0.1,
    'QStoQE': 0.1,
    'QEtoQPRE': 0.1,
    'QPREtoQSYM': 0.1,
    'QPREtoQASYM': 0.1,
    'QSYMtoH': 0.1,
    'QSYMtoQR': 0.1,
    'QASYMtoQR': 0.1,
    '_toS': 0.1,
}

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
        How it's different from ExtSEIRNetworkModel: # TODO
        """

        super().__init__()

        self.init_VL = init_VL
        self.VL_slopes = VL_slopes
        self.current_VL = np.zeros(self.numNodes)
        # VL value at end of last state; updated only when a node undergoes a transition
        self.last_state_VL = np.zeros(self.numNodes)

        self.initialize_VL_and_beta()
    

    def initialize_VL_and_beta(self):
        r"""
        Initialize the log10 viral load of each node at the start of the simulation.
        This function should be called in __init__().

        The rationale for keeping this separate from update_VL() is the following:
        if someone starts from an infectious state whose previous state also should have positive viral load, 
            then the initialized all-0 self.last_state_VL doesn't make sense;
            instead we should generate a positive hypothetical last_state_log10_VL value
            e.g., we sample the initial VL from some distribution like Brault's GMM
        This logic is a bit too messy to implement in update_VL. So let's keep it separate.
        In other words, update_VL would not involve any kind of sampling of the starting VL
        """

        # for each node, get / sample initial viral load
        for node, state in enumerate(self.X):
            self.current_VL[node] = self.init_VL[state]
            # TODO: enable sampling later

        # TODO: set beta according to viral load
        self.update_beta_given_VL()



    def update_VL(self):

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
        - if a state transition occurs, update self.last_state_VL (here? or in run_one_iteration?)
        """


        # TODO: think through: if the transition type is different, e.g., Isym to R or H
        # the slope could be different

        # and if this function is called for screening
        # where a node's transition happens after screening but the state is already updated, how do we deal with that?

        pass


    def update_beta_given_VL(self):
        """
        Update the transmission rates given the current viral load.
        The logic is TBD

        This function should be called every time after update_VL() is called
        or maybe combine the two into one function
        """

        # TODO: think through what logic to use

        pass




    def run_iteration(self, max_dt=None):
        # return super().run_iteration(max_dt)? Maybe not
        # TODO: copy over all the steps
        # and update VL and beta at the end
        # update self.last_state_VL must happen here, I think

        # there seems to be no other cleaner way to do this