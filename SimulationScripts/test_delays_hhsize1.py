from multiprocessing import Pool
import sys
sys.path.append("..")
import os
sys.path.append(os.getcwd())
import household_contact_tracing.BranchingProcessSimulation_hhsize1 as hct
import pandas as pd
import pickle
import numpy.random as npr
from datetime import datetime

# npr.seed(1)

repeats = 100
days_to_simulate = 25
starting_infections = 5000

# Importing the calibration dictionary
with open('Data/Calibration/hazard_rate_detection_prob_pairs_hhsize1.pickle', 'rb') as handle:
    pairs_dict = pickle.load(handle)

def run_simulation(repeat):

    print(f'starting simulation {input} at {datetime.now()}')

    npr.seed(None)

    infection_reporting_prob = npr.choice([0.1, 0.2, 0.3, 0.4, 0.5])

    haz_rate_scale = pairs_dict[infection_reporting_prob]

    contact_tracing_success_prob = npr.uniform(0.7, 0.95)

    contact_trace_delay_par = npr.uniform(1.5, 2.5)

    reduce_contacts_by = npr.uniform(0.0, 0.9)

    test_delay_mean = npr.uniform(1, 2)

    do_2_step = False

    prob_has_trace_app = npr.uniform(0, 0.5)

    simulation = hct.household_sim_contact_tracing(haz_rate_scale=haz_rate_scale,
                                                     household_haz_rate_scale=0.77729,
                                                     contact_tracing_success_prob=contact_tracing_success_prob,
                                                     contact_trace_delay_par=contact_trace_delay_par,
                                                     overdispersion=0.36,
                                                     infection_reporting_prob=infection_reporting_prob,
                                                     contact_trace=True,
                                                     reduce_contacts_by=reduce_contacts_by,
                                                     do_2_step=do_2_step,
                                                     test_before_propagate_tracing=True,
                                                     test_delay_mean=test_delay_mean,
                                                     prob_has_trace_app=prob_has_trace_app,
                                                     starting_infections=starting_infections)

    simulation.run_simulation(days_to_simulate)

    parameters = [
        haz_rate_scale,
        infection_reporting_prob,
        contact_tracing_success_prob,
        contact_trace_delay_par,
        reduce_contacts_by,
        do_2_step,
        prob_has_trace_app,
        test_delay_mean
    ]
    return(parameters + simulation.inf_counts)

param_names = [
    "hazard_rate_scale",
    "infection_reporting_prob",
    "contact_tracing_success_prob",
    "contact_trace_delay_par",
    "global_contact_reduction",
    "two_step_tracing",
    "prob_has_trace_app",
    "test_delay_mean"
]

col_names = param_names + [str(i) for i in range(days_to_simulate)]
col_names_dict = {}
for i in range(len(col_names)):
    col_names_dict.update({i: col_names[i]})

if __name__ == '__main__':
    with Pool() as p:
        results = p.map(run_simulation, range(repeats))
        results = pd.DataFrame(results)
        results = results.rename(columns=col_names_dict)
        results.to_excel("Data/Simulation Results/growth_rates_with_test_delays_hh_size1_IDP_0.1_0.5_no2step.xlsx")
