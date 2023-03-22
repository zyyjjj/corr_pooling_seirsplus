import numpy as np
from models import ExtSEIRSNetworkModel


class ViralExtSEIRNetworkModel(ExtSEIRSNetworkModel):
    # to add VL and time-varying transmissibility
    # TODO s:
    # 1. have a viral load array tracking every person's VL over time (cts or discrete?)
    # 2. have beta (transmission rate) depend on VL

    def __init__(self, G, beta, sigma, lamda, gamma, 
                    gamma_asym=None, eta=0, gamma_H=None, mu_H=0, alpha=1.0, xi=0, mu_0=0, nu=0, a=0, h=0, f=0, p=0,             
                    beta_local=None, beta_asym=None, beta_asym_local=None, beta_pairwise_mode='infected', delta=None, delta_pairwise_mode=None,
                    G_Q=None, beta_Q=None, beta_Q_local=None, sigma_Q=None, lamda_Q=None, eta_Q=None, gamma_Q_sym=None, gamma_Q_asym=None, alpha_Q=None, delta_Q=None,
                    theta_S=0, theta_E=0, theta_pre=0, theta_sym=0, theta_asym=0, phi_S=0, phi_E=0, phi_pre=0, phi_sym=0, phi_asym=0,    
                    psi_S=0, psi_E=1, psi_pre=1, psi_sym=1, psi_asym=1, q=0, isolation_time=14,
                    initE=0, initI_pre=0, initI_sym=0, initI_asym=0, initH=0, initR=0, initF=0,        
                    initQ_S=0, initQ_E=0, initQ_pre=0, initQ_sym=0, initQ_asym=0, initQ_R=0,
                    o=0, prevalence_ext=0,
                    transition_mode='exponential_rates', node_groups=None, store_Xseries=False, seed=None):


        super().__init__()

        self.log10VL = np.zeros(self.numNodes)
        self.update_VL()

    def update_VL(self):

        """
        Update the log10 viral load of each node depending on their current state (self.X), 
        and the time they have been in their current state (self.timer_state)

        This function should be called in __init__() and every time a state transition happens (model.run_one_iteration()),
        and every time a screening takes place (run_tti_one_day()).

        Seems like we should also keep an array of VL value at end of last state
        This array is updated only when a node undergoes a transition
        When initializing, for the infected, we sample the initial VL from some distribution like Brault's GMM

        The overall logic of updating the VL: 
        - the VL at the end of last state defines a "baseline value"
        - the current state maps to a slope of increase/decrease, either deterministic or sampled
            (in initial stages, should be increasing; if recovering, should be decreasing)
        - update goes like: new_VL = baseline_VL + slope * timer_state[i]
        """

        pass


    def update_beta_given_VL(self):
        """
        Update the transmission rates given the current viral load.
        The logic is TBD

        This function should be called every time after update_VL() is called
        """

        # TODO: think through what logic to use

        pass



