# This is a version of the model where we overide some aspects so that household have a set size of 1

from household_contact_tracing.BranchingProcessSimulation import household_sim_contact_tracing

class household_sim_contact_tracing_hh_size_1(household_sim_contact_tracing):

    def __init__(
        self,
        haz_rate_scale: float,
        contact_tracing_success_prob: float,
        contact_trace_delay_par: float,
        overdispersion: float,
        infection_reporting_prob: float,
        contact_trace: bool,
        household_haz_rate_scale: bool,
        do_2_step=False,
        backwards_trace=True,
        reduce_contacts_by=0,
        prob_has_trace_app=0,
        hh_propensity_to_use_trace_app=1,
        test_delay_mean=1.52,
        test_before_propagate_tracing=True,
        starting_infections=1,
        hh_prob_will_take_up_isolation=1,
        hh_prob_propensity_to_leave_isolation=0,
        leave_isolation_prob=0
    ):

        super().__init__(
            haz_rate_scale=haz_rate_scale,
            contact_tracing_success_prob=contact_tracing_success_prob,
            contact_trace_delay_par=contact_trace_delay_par,
            overdispersion=overdispersion,
            infection_reporting_prob=infection_reporting_prob,
            contact_trace=contact_trace,
            household_haz_rate_scale=household_haz_rate_scale,
            do_2_step=do_2_step,
            backwards_trace=backwards_trace,
            reduce_contacts_by=reduce_contacts_by,
            prob_has_trace_app=prob_has_trace_app,
            hh_propensity_to_use_trace_app=hh_propensity_to_use_trace_app,
            test_delay_mean=test_delay_mean,
            test_before_propagate_tracing=test_before_propagate_tracing,
            starting_infections=starting_infections,
            hh_prob_will_take_up_isolation=hh_prob_will_take_up_isolation,
            hh_prob_propensity_to_leave_isolation=hh_prob_propensity_to_leave_isolation,
            leave_isolation_prob=leave_isolation_prob
        )

        self.local_contact_probs = [0]
        self.total_contact_means = [11.73887]
            
    
    def size_of_household(self):
        """Creates household that have a fixed size of 1. This effectively moves the branching process
        to an individual level branching process.
        """
        return 1

