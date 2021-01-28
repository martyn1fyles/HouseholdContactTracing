import household_contact_tracing.BranchingProcessSimulation as hct    # The code to test
import pytest
# generate coverage report using:
# py.test test --cov-report xml:cov.xml --cov household_contact_tracing
# in the terminal

@pytest.fixture
def simple_model():
    simulation = hct.uk_model(
        haz_rate_scale=0.2,
        household_haz_rate_scale=0.8,
        overdispersion=0.32,
        contact_tracing_success_prob=1,
        contact_trace_delay_par=0,
        infection_reporting_prob=0.2,
        contact_trace=True,
        reduce_contacts_by=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
        starting_infections=1,
        test_before_propagate_tracing=False,
        prob_has_trace_app=1,
        hh_propensity_to_use_trace_app=1)
    return simulation


def test_new_infection(simple_model):

    simulation = simple_model
    
    simulation.new_household(new_household_number = 2, generation =2, infected_by=1, infected_by_node = 1)
    simulation.new_infection(node_count = 2, generation = 2, household_id = 2, serial_interval = 2, infecting_node = simulation.nodes.node(1))
    simulation.new_infection(node_count = 3, generation = 2, household_id = 1, serial_interval = 2, infecting_node = simulation.nodes.node(1))

    assert simulation.nodes.node(1).locally_infected == False
    assert simulation.nodes.node(2).locally_infected == False
    assert simulation.nodes.node(3).locally_infected == True
    assert simulation.nodes.node(3).propagated_contact_tracing == False

def test_attempt_contact_trace_of_household(simple_model):

    simulation = simple_model

    simulation.new_household(new_household_number = 2, generation =2, infected_by=1, infected_by_node = 1)
    simulation.new_infection(node_count = 2, generation = 2, household_id = 2, serial_interval = 2, infecting_node = simulation.nodes.node(1))
    simulation.new_infection(node_count = 3, generation = 2, household_id = 1, serial_interval = 2, infecting_node = simulation.nodes.node(1))

    assert simulation.nodes.node(1).locally_infected == False
    assert simulation.nodes.node(2).locally_infected == False
    assert simulation.nodes.node(3).locally_infected == True
    assert simulation.nodes.node(3).propagated_contact_tracing == False


@pytest.fixture
def no_app_tracing_model():
    """Returns a simple model where no member of the population has the contact tracing app

    Contact tracing success is guaranteed.
    """

    model = hct.uk_model(
        haz_rate_scale=0.2,
        household_haz_rate_scale=0.8,
        overdispersion=0.32,
        contact_tracing_success_prob=1,
        contact_trace_delay_par=50,
        infection_reporting_prob=0.2,
        contact_trace=True,
        reduce_contacts_by=0.1,
        starting_infections=1,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0,
        hh_propensity_to_use_trace_app=0)

    return model

@pytest.fixture
def only_app_tracing_model():
    """Returns a simple model where every member of the population has the contact tracing app 

    Tracing through the conventional route is guaranteed to fail - it is ONLY app based tracing
    """
    model = hct.uk_model(
        haz_rate_scale=0.2,
        household_haz_rate_scale=0.8,
        overdispersion=0.32,
        contact_tracing_success_prob=0,
        contact_trace_delay_par=100,
        infection_reporting_prob=0.2,
        contact_trace=True,
        reduce_contacts_by=0.1,
        starting_infections=1,
        test_before_propagate_tracing=False,
        prob_has_trace_app=1,
        hh_propensity_to_use_trace_app=1)

    return model

def test_attempt_contact_trace_of_household_non_app_traced(no_app_tracing_model):
    
    model = no_app_tracing_model

    # create a new outside household infection
    model.new_outside_household_infection(infecting_node = model.nodes.node(1), serial_interval = 0)

    # propagate a contact tracing attempt
    model.attempt_contact_trace_of_household(house_to = model.houses.household(2), 
        house_from = model.houses.household(1),
        days_since_contact_occurred = 0)

    # get the household which launched the tracing attempt
    tracing_household = model.houses.household(1)

    # check that we correctly record which household was successfully contact traced
    assert tracing_household.contact_traced_household_ids == [2]

    # get the household which has just been contact traced and check it has the right tracing index
    traced_household = model.houses.household(2)
    assert traced_household.contact_tracing_index == 1 

    # check that the traced household has the correct time until contact traced
    # and that the traced household records where it is being contact traced from
    assert traced_household.time_until_contact_traced > 10
    assert traced_household.being_contact_traced_from == 1

    assert model.nodes.G.edges[1,2]['colour'] == model.contact_traced_edge_between_house


def test_attempt_contact_trace_of_household_app_traced(only_app_tracing_model):

    model = only_app_tracing_model

    # create a new outside household infection
    model.new_outside_household_infection(infecting_node = model.nodes.node(1), serial_interval = 0)

    # propagate a contact tracing attempt
    model.attempt_contact_trace_of_household(house_to = model.houses.household(2), 
        house_from = model.houses.household(1),
        days_since_contact_occurred = 0)

    # get the household which launched the tracing attempt
    tracing_household = model.houses.household(1)

        # check that we correctly record which household was successfully contact traced
    assert tracing_household.contact_traced_household_ids == [2]

    # get the household which has just been contact traced and check it has the right tracing index
    traced_household = model.houses.household(2)
    assert traced_household.contact_tracing_index == 1 

    # check that the traced household has the correct time until contact traced
    # and that the traced household records where it is being contact traced from
    assert traced_household.time_until_contact_traced == 0
    assert traced_household.being_contact_traced_from == 1

    assert model.nodes.G.edges[1,2]['colour'] == model.app_traced_edge

@pytest.fixture
def high_recall_decay_model():
    """Creates a model where every tracing attempt succeeds with probability 1, but there is a high recall decay.
    """
    model = hct.uk_model(
        haz_rate_scale=0.2,
        household_haz_rate_scale=0.8,
        overdispersion=0.32,
        contact_tracing_success_prob=1,
        contact_trace_delay_par=100,
        infection_reporting_prob=0.2,
        contact_trace=True,
        reduce_contacts_by=0.1,
        starting_infections=1,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0,
        hh_propensity_to_use_trace_app=0,
        recall_probability_fall_off=0.5)

    return model

def test_recall_decay_0_days(high_recall_decay_model):
    """Tests that the model is unaffected by recall if the contact occurred 0 days in the past.
    """
    model = high_recall_decay_model

    # create a new outside household infection
    model.new_outside_household_infection(infecting_node = model.nodes.node(1), serial_interval = 0)

    # propagate a contact tracing attempt
    model.attempt_contact_trace_of_household(house_to = model.houses.household(2), 
        house_from = model.houses.household(1),
        days_since_contact_occurred = 0)

    # get the household which launched the tracing attempt
    tracing_household = model.houses.household(1)

        # check that we correctly record which household was successfully contact traced
    assert tracing_household.contact_traced_household_ids == [2]

    # get the household which has just been contact traced and check it has the right tracing index
    traced_household = model.houses.household(2)
    assert traced_household.contact_tracing_index == 1 

    # check that the traced household has the correct time until contact traced
    # and that the traced household records where it is being contact traced from
    assert traced_household.time_until_contact_traced > 10
    assert traced_household.being_contact_traced_from == 1

def test_recall_decay_probability_1(high_recall_decay_model):
    """Tests that for a contact that occurred a really long time ago, there is almost no chance of
    the contact being recalled.
    """

    model = high_recall_decay_model

    # create a new outside household infection
    model.new_outside_household_infection(infecting_node = model.nodes.node(1), serial_interval = 0)

    # it has been 1000 days since contact occurred, so the probability of recall should be 0
    # propagate a contact tracing attempt
    model.attempt_contact_trace_of_household(house_to = model.houses.household(2), 
        house_from = model.houses.household(1),
        days_since_contact_occurred = 10000)

    # get the household which launched the tracing attempt
    traced_household = model.houses.household(2)
    assert traced_household.time_until_contact_traced == float('Inf')

def test_propagate_contact_tracing_backwards(no_app_tracing_model):
    
    model = no_app_tracing_model

    model.new_outside_household_infection(infecting_node = model.nodes.node(1), serial_interval = 0)

    model.propagate_contact_tracing(model.nodes.node(2))

    assert model.houses.household(1).being_contact_traced_from == 2

def test_propagate_contact_tracing_forwards(no_app_tracing_model):

    model = no_app_tracing_model

    model.new_outside_household_infection(infecting_node = model.nodes.node(1), serial_interval = 0)

    model.propagate_contact_tracing(model.nodes.node(2))
    
    assert model.houses.household(1).being_contact_traced_from == 2

def test_propagate_contact_tracing_time_limits_lower(no_app_tracing_model):
    """Check that contact that occurred over 2 days priors to symptom onset are not traced
    """
    model = no_app_tracing_model

    # contact occurred at time = 0
    model.new_outside_household_infection(infecting_node = model.nodes.node(1), serial_interval = 0)

    # the infecting node has symptom onset at time 5
    # therefore the contact occurred 5 days in the past
    model.nodes.node(1).symptom_onset = 5

    # set the time to 5 (not sure if this will make a difference)
    model.time = 5

    model.propagate_contact_tracing(model.nodes.node(1))

    # check that the household is not being contact traced
    assert model.houses.household(1).being_contact_traced_from is None

def test_propagate_contact_tracing_time_limits_upper(no_app_tracing_model):
    """Check that contact that occurred over 5 days after symptom onset are not traced
    """
    model = no_app_tracing_model

    # the infecting node has symptom onset at time 5
    # therefore the contact occurred 5 days in the past
    model.nodes.node(1).symptom_onset = 2

    # set the time to 5 (not sure if this will make a difference)
    model.time = 10    
    # contact occurred at time = 0
    model.new_outside_household_infection(infecting_node = model.nodes.node(1), serial_interval = 10)

    model.propagate_contact_tracing(model.nodes.node(1))

    # check that the household is not being contact traced
    assert model.houses.household(1).being_contact_traced_from is None

def test_propagate_contact_tracing_node_attribute_updates(no_app_tracing_model):
    
    model = no_app_tracing_model

    model.new_outside_household_infection(infecting_node = model.nodes.node(1), serial_interval = 0)

    model.propagate_contact_tracing(node = model.nodes.node(1))

    assert model.nodes.node(1).propagated_contact_tracing

    assert model.nodes.node(1).time_propagated_tracing == model.time


def test_increment_contact_tracing_isolate_households(no_app_tracing_model):
    
    model = no_app_tracing_model

    model.nodes.node(1).contact_traced = True
    model.nodes.node(1).symptom_onset_time = 0
    model.increment_contact_tracing()
    assert model.houses.household(1).isolated == True

def test_increment_contact_tracing_propagation_probable_infections_need_test(no_app_tracing_model):
    model = no_app_tracing_model

    # set up a household to be traced
    model.new_outside_household_infection(infecting_node = model.nodes.node(1), serial_interval = 0)

    # isolate the household, this should produce an isolation time of 0
    model.isolate_household(model.nodes.node(1).household())

    # set the nodes symptom onset time and testing delay to 0
    # so it is a confirmed infection at time 0. with positive test time of 0
    model.nodes.node(1).testing_delay = 0
    model.nodes.node(1).symptom_onset_time = 0

    model.increment_contact_tracing()

    assert model.nodes.node(1).propagated_contact_tracing

def test_increment_contact_tracing_propagation_probable_infections_do_not_need_test(no_app_tracing_model):

    model = no_app_tracing_model

    # set up a household to be traced
    model.new_outside_household_infection(infecting_node = model.nodes.node(1), serial_interval = 0)

    # isolate the household, this should produce an isolation time of 0
    model.isolate_household(model.nodes.node(1).household())

    # set the nodes symptom onset time and testing delay to 0
    # so it is a confirmed infection at time 0. with positive test time of 0
    model.nodes.node(1).testing_delay = 1
    model.nodes.node(1).symptom_onset_time = 0

    model.increment_contact_tracing()

    assert not model.nodes.node(1).propagated_contact_tracing

def test_increment_contact_tracing_propagation_probable_infections_do_not_need_test(no_app_tracing_model):

    model = no_app_tracing_model

    # set up a household to be traced
    model.new_outside_household_infection(infecting_node = model.nodes.node(1), serial_interval = 0)

    # isolate the household, this should produce an isolation time of 0
    model.isolate_household(model.nodes.node(1).household())

    # set the nodes symptom onset time and testing delay to 0
    # so it is a confirmed infection at time 0. with positive test time of 0
    model.nodes.node(1).testing_delay = 1
    model.nodes.node(1).symptom_onset_time = 0

    model.time = 1

    model.increment_contact_tracing()

    assert model.nodes.node(1).propagated_contact_tracing
