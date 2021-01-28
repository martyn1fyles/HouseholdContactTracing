from multiprocessing import Pool
import sys
sys.path.append("..")
import os
sys.path.append(os.getcwd())
import household_contact_tracing.BranchingProcessSimulation as hct
import pandas as pd
import numpy.random as npr
import pickle

# npr.seed(1)

repeats = 100
days_to_simulate = 25
starting_infections = 5000

# Importing the calibration dictionary
with open('Data/Calibration/hazard_rate_detection_prob_pairs.pickle', 'rb') as handle:
    pairs_dict = pickle.load(handle)

def run_simulation(repeat):

    infection_reporting_prob = npr.choice([0.1, 0.2, 0.3, 0.4, 0.5])

    haz_rate_scale = pairs_dict[infection_reporting_prob]

    contact_tracing_success_prob = npr.uniform(0.7, 0.95)

    contact_trace_delay_par = npr.uniform(1.5, 2.5)

    reduce_contacts_by = npr.uniform(0.0, 0.9)

    do_2_step = npr.choice([True, False])

    prob_has_trace_app = npr.uniform(0, 0.5)
    
    hh_prob_will_take_up_isolation = npr.uniform(0.5, 1)

    prob_household_prop_not_adhere = npr.uniform(0.0, 0.5)

    prob_not_adhere = npr.uniform(0.01, 0.05)

    simulation = hct.household_sim_contact_tracing(haz_rate_scale=haz_rate_scale,
                                                    household_haz_rate_scale=0.832824527,
                                                     contact_tracing_success_prob=contact_tracing_success_prob,
                                                     contact_trace_delay_par=contact_trace_delay_par,
                                                     overdispersion=0.36,
                                                     infection_reporting_prob=infection_reporting_prob,
                                                     contact_trace=True,
                                                     reduce_contacts_by=reduce_contacts_by,
                                                     do_2_step=do_2_step,
                                                     test_before_propagate_tracing=False,
                                                     prob_has_trace_app=prob_has_trace_app,
                                                     starting_infections=starting_infections,
                                                     hh_prob_will_take_up_isolation=hh_prob_will_take_up_isolation,
                                                     hh_prob_propensity_to_leave_isolation=prob_household_prop_not_adhere,
                                                     leave_isolation_prob=prob_not_adhere)

    simulation.run_simulation(days_to_simulate)

    parameters = [
        haz_rate_scale,
        infection_reporting_prob,
        contact_tracing_success_prob,
        contact_trace_delay_par,
        reduce_contacts_by,
        do_2_step,
        prob_has_trace_app,
        hh_prob_will_take_up_isolation,
        prob_household_prop_not_adhere,
        prob_not_adhere
    ]
    return(parameters + simulation.inf_counts)

param_names = [
    "hazard_rate_scale",
    "infection_reporting_prob",
    "contact_tracing_success_prob",
    "contact_trace_delay_par",
    "reduce_contacts_by",
    "two_step_tracing",
    "prob_has_trace_app",
    "hh_prob_will_take_up_isolation",
    "prob_household_prop_not_adhere",
    "prob_not_adhere"
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
        results.to_excel("Data/Simulation Results/growth rates no test delays adherence.xlsx")
