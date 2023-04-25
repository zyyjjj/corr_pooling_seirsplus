"""
Sketch code for running full loop, as exemplified in test_sim_runner.ipynb

(Using a yaml file?)
set population size N
set INIT_EXPOSED = init_prevalence * N
set simulation time T
set output path to be sth like "../results/US_N=10000_p=0.01_T=100"

for seed in seeds:
    generate graph using generate_demographic_contact_network()
    initialize ViralExtSEIRNetworkModel
    for strategy in ["NP", "CP"]:
        intialize SimulationRunner
        SimulationRunner.run_simulation()
        SimulationRunner.get_performance() # maybe save these 
"""

"""
Then, need code for loading and aggregating the experiment results.
"""
