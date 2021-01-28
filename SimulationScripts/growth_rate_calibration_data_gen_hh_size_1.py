from multiprocessing import Pool
import household_contact_tracing.BranchingProcessSimulation_hhsize1 as hct
import pandas as pd
import numpy as np
import numpy.random as npr
import itertools as it

# Runs the model over a hypergrid of hazard_rate_scales (which control infectiousness)
# and infection reporting probs, which also impact the growth of the epidemic.
# The household_sar should be calibrated prior to running the simulation, this is done
# using hte calibration notebook since it not too computationally difficult.

# Epidemics have exponential growth, don't make this too large.
days_to_simulate = 20

# Range of parameters to iterate over
hazard_rate_range = np.linspace(0.2, 0.3, 5)
infection_reporting_probs = [0.1] #[0.1, 0.2, 0.3, 0.4, 0.5]

# Create a grid of parameters
parameter_grid = it.product(hazard_rate_range, infection_reporting_probs)

# Function to set up and run the simulation


def run_simulation(pars):
    haz_rate_scale, infection_reporting_prob = pars

    simulation = hct.household_sim_contact_tracing_hh_size_1(haz_rate_scale=haz_rate_scale,
                                                   household_haz_rate_scale=0,
                                                   contact_tracing_success_prob=0,
                                                   contact_trace_delay_par=0,
                                                   overdispersion=0.36,
                                                   infection_reporting_prob=infection_reporting_prob,
                                                   contact_trace=False,
                                                   reduce_contacts_by=0,
                                                   starting_infections=1000)

    simulation.run_simulation(days_to_simulate)

    parameters = [
        haz_rate_scale,
        infection_reporting_prob
    ]
    return(parameters + simulation.inf_counts)

# Setting up to record outputs
param_names = [
    "haz_rate_scale",
    "infection_reporting_prob"
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
        results.to_excel("Data/Calibration/growth_rates_hhSize1.xlsx")
