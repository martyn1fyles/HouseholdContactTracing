
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:45:26 2020

@author: LizFearon
"""

from multiprocessing import Pool
import sys
sys.path.append("../..")
import household_contact_tracing.BranchingProcessSimulation as hct
import pandas as pd
import numpy.random as npr
import pickle

# Import the calibration dictionary
handle = open('Data/Calibration/hazard_rate_detection_prob_pairs.pickle', 'rb')
pairs_dict = pickle.load(handle)

# Simulation configuration
repeats = 5000
days_to_simulate = 365
starting_infections = 1

def run_simulation(repeat):
    
    # Infection detection probability and hazard rate scale need to be linked
    detect_prob_haz_scale_pairs = pairs_dict

    # choose a detection prob
    infection_reporting_prob = npr.choice([0.1, 0.2, 0.3, 0.4, 0.5])

    # get the corresponding hazard rate scale
    hazard_rate_scale = detect_prob_haz_scale_pairs[infection_reporting_prob]

    # Decide if baseline scenario
    contact_trace = True

    # varying the contact tracing success probabilities
    contact_tracing_success_prob = npr.uniform(0.7, 0.95)

    # vary the speed of contact tracing
    contact_trace_delay_par = npr.uniform(1.5, 2.5)

    # whether to do two step tracing or not
    do_2_step = npr.choice([True, False])

    # vary the proportion having the contact tracing app
    prob_has_trace_app = npr.uniform(0, 0.5)
    
    # adherence parameters
    # Probability that a household will take up isolation if traced
    hh_prob_will_take_up_isolation = npr.uniform(0.5, 1)
    
    # hh_prob_propensity_to_leave_isolation = npr.uniform(0, 0.5)
    hh_prob_propensity_to_leave_isolation = 0
                 
    # leave_isolation_prob = npr.uniform(0.01, 0.05)
    leave_isolation_prob = 0

    # baseline scenario
    # global_contact_reduction = 0
    # Scenario A
    # reduce_contacts_by=(0.68, 0.83, 0.83, 0.821, 0.846, 0.836)
    # Scenario B
    # reduce_contacts_by=(0.638, 0.786, 0.76, 0.733, 0.765, 0.755)
    # Scenario C
    # reduce_contacts_by=(0.628, 0.76, 0.685, 0.632, 0.668, 0.668)
    #Scenario D
    # reduce_contacts_by=(0.561, 0.698, 0.61, 0.543, 0.589, 0.577)
    # Scenario E
    reduce_contacts_by = (0.413, 0.544, 0.393, 0.278, 0.348, 0.315)

    simulation = model.household_sim_contact_tracing(haz_rate_scale=hazard_rate_scale,
                                                     contact_tracing_success_prob=contact_tracing_success_prob,
                                                     contact_trace_delay_par=contact_trace_delay_par,
                                                     overdispersion=0.36,
                                                     infection_reporting_prob=infection_reporting_prob,
                                                     contact_trace=contact_trace,
                                                     reduce_contacts_by=reduce_contacts_by,
                                                     do_2_step=do_2_step,
                                                     test_before_propagate_tracing=False,
                                                     prob_has_trace_app=prob_has_trace_app,
                                                     hh_prob_will_take_up_isolation=hh_prob_will_take_up_isolation,
                                                     hh_prob_propensity_to_leave_isolation=hh_prob_propensity_to_leave_isolation,
                                                     leave_isolation_prob=leave_isolation_prob,
                                                     starting_infections=starting_infections)

    simulation.run_simulation(days_to_simulate, stop_when_5000_infections = True)

    parameters = [
        hazard_rate_scale,
        infection_reporting_prob,
        contact_tracing_success_prob,
        contact_trace_delay_par,
        reduce_contacts_by,
        do_2_step,
        prob_has_trace_app,
        hh_prob_will_take_up_isolation,
        hh_prob_propensity_to_leave_isolation,
        leave_isolation_prob
    ]
    
    return(parameters + [simulation.end_reason, simulation.day_extinct] + simulation.inf_counts)  # + simulation.day_ext)

param_names = [
    "hazard_rate_scale",
    "infection_reporting_prob",
    "contact_tracing_success_prob",
    "contact_trace_delay_par",
    "global_contact_reduction",
    "two_step_tracing",
    "prob_has_trace_app",
    "hh_prob_will_take_up_isolation",
    "hh_prob_propensity_to_leave_isolation",
    "leave_isolation_prob"
]

simulation_names = [
    "end_reason",
    "extinction_time"
]


col_names = param_names + simulation_names + [str(i) for i in range(days_to_simulate)]
col_names_dict = {}
for i in range(len(col_names)):
    col_names_dict.update({i: col_names[i]})
    
#for i, name in enumerate(col_names):
#    col_names_dict[i] = name

if __name__ == '__main__':
    with Pool() as p:
        results = p.map(run_simulation, range(repeats))
        results = pd.DataFrame(results)
        results = results.rename(columns=col_names_dict)
        results.to_excel("../Data/Extinction Times/Lockdown Relaxations/adherence_TEST.xlsx")
