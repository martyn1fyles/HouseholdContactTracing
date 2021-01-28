from multiprocessing import Pool
import sys
sys.path.append("..")
import os
sys.path.append(os.getcwd())
import household_contact_tracing.BranchingProcessSimulation as hct
import pandas as pd
import pickle
import numpy as np
import numpy.random as npr
import itertools as it

# Simulation Script Purpose:
# This script is used to compute the growth rate of the epidemic without contact tracing
# or any household contact tracing across all social distancing possibilities. Uses the
# calibration dictionary to tune itself across all infection reporting probabilities.

# Epidemics have exponential growth, don't make this too large as this is difficult for
# computers to handle
days_to_simulate = 20

# Importing the calibration dictionary
with open('Data/Calibration/hazard_rate_detection_prob_pairs.pickle', 'rb') as handle:
    pairs_dict = pickle.load(handle)

# Setting up the parameter grid
global_reduction_in_contacts = np.linspace(0, 0.9)
infection_reporting_prob_range = [0.1, 0.2, 0.3, 0.4, 0.5]
parameter_grid = it.product(global_reduction_in_contacts, infection_reporting_prob_range)

# Function to set up and run the simulation
def run_simulation(pars):

    # Unpack the parameters
    global_contact_reduction, infection_reporting_prob = pars

    # Get the corresponding hazard rate scale from the calibration dictionary
    haz_rate_scale = pairs_dict[infection_reporting_prob]

    # Configure the simulation
    simulation = hct.household_sim_contact_tracing(haz_rate_scale=haz_rate_scale,
                                                   household_haz_rate_scale=0.832824527,   
                                                   contact_tracing_success_prob=0,
                                                   contact_trace_delay_par=0,
                                                   overdispersion=0.36,
                                                   infection_reporting_prob=0, # By setting this to 0 there are no household quarantines
                                                   contact_trace=False,
                                                   reduce_contacts_by=global_contact_reduction,
                                                   starting_infections=5000)

    simulation.run_simulation(days_to_simulate)

    parameters = [
        infection_reporting_prob,
        global_contact_reduction
    ]
    return(parameters + simulation.inf_counts)

# Setting up to record outputs
param_names = [
    "infection_reporting_prob",
    "global_contact_reduction"
]

# Setting up column names for recording the output
col_names = param_names + [str(i) for i in range(days_to_simulate)]
col_names_dict = {}
for i in range(len(col_names)):
    col_names_dict.update({i: col_names[i]})

# Pass the parameter grid to the multiprocessing pool to record the output
if __name__ == '__main__':
    with Pool() as p:
        results = p.map(run_simulation, parameter_grid)
        results = pd.DataFrame(results)
        results = results.rename(columns=col_names_dict)
        results.to_excel("Data/Reference/growth_rates_no_tracing_no_household_quarantine.xlsx")
