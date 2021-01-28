import household_contact_tracing.BranchingProcessSimulation as hct    # The code to test
import numpy as np
import pytest

# generate coverage report using:
# pytest --cov=. --cov-report xml:cov.xml
# make sure you have pytest-cov installed
# in the terminal

# test_model = hct.household_sim_contact_tracing(
#     haz_rate_scale=0.805,
#     household_haz_rate_scale=0.5,
#     contact_tracing_success_prob=0.66,
#     contact_trace_delay_par=2,
#     overdispersion=0.36,
#     infection_reporting_prob=0.8,
#     contact_trace=True,
#     do_2_step=True,
#     test_before_propagate_tracing=True
# )

@pytest.fixture
def basic_model():
    return hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        household_haz_rate_scale=0.8,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        do_2_step=True,
        test_before_propagate_tracing=True
    )

@pytest.fixture
def app_based_tracing_model():
    return hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        household_haz_rate_scale=0.8,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        prob_has_trace_app=1)

def test_incubation_period(basic_model):
    """Asserts that the mean and var of the incubation period are in approximately the right range
    """

    incubation_periods = np.array([
        basic_model.incubation_period()
        for i in range(10000)])

    # True mean = 4.83
    assert incubation_periods.mean() < 5
    assert incubation_periods.mean() > 4.5
    # True var = 7.7
    assert incubation_periods.var() > 7.5
    assert incubation_periods.var() < 8.5


def test_testing_delays(basic_model):
    """Asserts that the testing delay distributions is in approximately the right range
    """

    test_delays = np.array([
        basic_model.testing_delay()
        for _ in range(10000)])

    # True mean = 1.52 (default)
    assert test_delays.mean() < 1.75
    assert test_delays.mean() > 1.25
    # True var = 1.11
    assert test_delays.var() > 0.8
    assert test_delays.var() < 1.5


def test_reporting_delay(basic_model):
    """Asserts that the reporting delay is in approximately the right range
    """
    reporting_delays = np.array([
        basic_model.reporting_delay()
        for _ in range(10000)])

    # True mean = 2.68
    assert reporting_delays.mean() < 3
    assert reporting_delays.mean() > 2
    # True var = 2.38^2 = 5.66 ish
    assert reporting_delays.var() > 4
    assert reporting_delays.var() < 7


def test_contact_trace_delay(basic_model):
    """Asserts that the contact tracing delay is in approximately the right range
    """

    assert basic_model.contact_trace_delay(True) == 0

    trace_delays = np.array(
        [
            basic_model.contact_trace_delay(False)
            for i in range(1000)
        ]
    )

    assert trace_delays.mean() < 2.5
    assert trace_delays.mean() > 1.5


def test_new_household(basic_model):
    """Starts a model and adds a new household. Checks that the household is created with all the attributes
    as intended.
    """

    model = basic_model

    model.time = 100

    model.new_household(new_household_number=10,
                        generation=5,
                        infected_by=6,
                        infected_by_node=3)

    house = model.houses.household(10)

    assert house.size in [1, 2, 3, 4, 5, 6]
    assert house.time == 100
    assert house.size - 1 == house.susceptibles
    assert house.generation == 5
    assert house.infected_by_id == 6
    assert house.infected_by_node == 3


def test_get_edge_between_household(basic_model):
    """Tests logic that is used to find edges between households.

    Starts a model, adds two households where household 2 is infected by household 1, therefore there should be an edge
    between the two households. Tests that this edge is there, and that the logic to find it works.
    """

    model = basic_model

    # household 1
    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    # infection 1
    model.new_infection(
        node_count=1,
        generation=1,
        household_id=1)

    # household 2
    model.new_household(
        new_household_number=2,
        generation=2,
        infected_by=1,
        infected_by_node=1)

    # infection 2
    model.new_infection(
        node_count=2,
        generation=2,
        household_id=2)

    # add an edge between the infections
    model.nodes.G.add_edge(1, 2)

    house1 = model.houses.household(1)
    house2 = model.houses.household(2)
    assert model.get_edge_between_household(house1, house2) == (1, 2)


def test_is_app_traced(app_based_tracing_model):
    """Tests that the logic used to determine whether an edge is traced by an app is working.

    Initiates a model where every node has the contact tracing application. Creates two households, and two infections,
    one in each household. Passes the edge between the two nodes to the function and check that it returns true, since
    both nodes have the contact tracing application.
    """

    model = app_based_tracing_model

    # household 1
    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    # infection 1
    model.new_infection(
        node_count=1,
        generation=1,
        household_id=1)

    # household 2
    model.new_household(
        new_household_number=2,
        generation=2,
        infected_by=1,
        infected_by_node=1)

    # infection 2
    model.new_infection(
        node_count=2,
        generation=2,
        household_id=2)

    # add an edge between the infections
    model.nodes.G.add_edge(1, 2)
    assert model.is_edge_app_traced((1, 2))


def test_new_outside_household_infection(basic_model):
    """Tests the convenience function for creating an outside household infection.

    Sets up a model with a starting infection and household, uses the new_out_side_household_infection function
    and checks that the correct attributes are updated and the edge is there on the network 
    """

    model = basic_model

    # household 1
    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    node1 = model.nodes.node(1)

    # infection 1
    model.new_infection(
        node_count=1,
        generation=1,
        household_id=1)

    model.new_outside_household_infection(
        infecting_node=node1,
        serial_interval=1
    )

    assert model.house_count == 2
    assert node1.spread_to == [2]
    assert model.nodes.G.has_edge(1, 2)


def test_within_household_infection(basic_model):

    model = basic_model

    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    model.new_infection(
        node_count=1,
        generation=1,
        household_id=1)

    node1 = model.nodes.node(1)
    house = model.houses.household(1)
    house.house_size = 2
    house.susceptibles = 1

    model.new_within_household_infection(
        infecting_node=node1,
        serial_interval=10)

    node2 = model.nodes.node(2)

    assert house.susceptibles == 0
    assert node1.spread_to == [2]
    assert node2.household_id == 1
    assert node2.serial_interval == 10
    assert node2.generation == 2
    assert model.nodes.G.edges[1, 2]["colour"] == "black"
    assert house.within_house_edges == [(1, 2)]


def test_perform_recoveries(basic_model):

    model = basic_model

    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    model.new_infection(
        node_count=1,
        generation=1,
        household_id=1)

    node1 = model.nodes.node(1)
    node1.recovery_time = 0
    model.perform_recoveries()
    assert node1.recovered is True


def test_colour_edges_between_houses(basic_model):

    model = basic_model

    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    node1 = model.nodes.node(1)

    model.new_infection(
        node_count=1,
        generation=1,
        household_id=1)

    model.node_count = 2

    model.new_outside_household_infection(
        infecting_node=node1,
        serial_interval=10
    )

    house1 = model.houses.household(1)
    house2 = model.houses.household(2)
    model.colour_node_edges_between_houses(house1, house2, "yellow")
    assert model.nodes.G.edges[1, 2]["colour"] == "yellow"


def test_overide_testing_delay():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        household_haz_rate_scale=0.8,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        test_before_propagate_tracing=False)

    assert model.testing_delay() == 0


def test_hh_prob_leave_iso_default():
    """Checks that the default value of the propensity to leave isolation is 0
    """
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        household_haz_rate_scale=0.8,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        test_before_propagate_tracing=False)
    assert model.hh_propensity_to_leave_isolation() == False


def test_hh_prob_leave_iso():
    """Tests that the values are correctly assigned
    """
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        household_haz_rate_scale=0.8,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        test_before_propagate_tracing=False,
        hh_prob_propensity_to_leave_isolation=1)
    assert model.hh_propensity_to_leave_isolation() == True


def test_hh_has_propensity_attr():
    """Checks that some houses have the propensity to not adhere
    """
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        household_haz_rate_scale=0.8,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        test_before_propagate_tracing=False,
        hh_prob_propensity_to_leave_isolation=0.5)

    assert model.houses.household(1).propensity_to_leave_isolation in (True, False)


def test_leave_isolation():
    """Give all household the propensity to not adhere, and sets the probability to not adhere to 1,
    runs the leave isolation method and checks that the node immediately decides to not adhere
    """

    # All households have the propensity
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        household_haz_rate_scale=0.8,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        test_before_propagate_tracing=False,
        hh_prob_propensity_to_leave_isolation=1,
        leave_isolation_prob=1)

    # set node 1 to the isolation status
    model.nodes.node(1).isolated = True

    # see if the node leaves isolation over the next 50 days
    model.decide_if_leave_isolation(node=model.nodes.node(1))

    assert model.nodes.node(1).isolated is False


def test_update_adherence_to_isolation():
    """Runs a simulation for a bit, and records the number of nodes in isolation.
    Performs the method that updates whether a node is adhering to isolation, and
    then checks that the number of nodes in isolation has decreased.
    """

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.803782,
        household_haz_rate_scale=0.8,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=10,
        hh_prob_propensity_to_leave_isolation=1,
        leave_isolation_prob=0.1
    )

    # let a simulation run for a bit
    model.run_simulation(20)

    # record how many nodes are currently isolated
    initially_isolated_ids = [
        node.node_id for node in model.nodes.all_nodes()
        if node.isolated
        and not node.recovered
        and node.household().propensity_to_leave_isolation
    ]

    # update the isolation 10 
    for _ in range(10):
        model.update_adherence_to_isolation()

    # get the list of nodes in isolation
    secondary_isolated_ids = [
        node.node_id for node in model.nodes.all_nodes()
        if node.isolated
        and not node.recovered
        and node.household().propensity_to_leave_isolation
    ]

    # check that the number of nodes in isolation has decreased
    assert initially_isolated_ids != secondary_isolated_ids

def test_update_adherence_to_isolation_manually():
    """Isolates a household, checks the nodes are isolated, runs the update adherence
    method with prob 1 and checks that the nodes are not adhering
    """

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.803782,
        household_haz_rate_scale=0.8,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=10,
        hh_prob_propensity_to_leave_isolation=1,
        leave_isolation_prob=0
    )

    # isolate a household
    # current leave isolation prob is 0, so it shouldn't have left
    model.isolate_household(model.houses.household(1))

    assert model.nodes.node(1).isolated is True

    # set the leave isolation probability to 1
    model.leave_isolation_prob = 1

    model.update_adherence_to_isolation()

    assert model.nodes.node(1).isolated is False

def test_node_colour():

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.8,
        household_haz_rate_scale=0.8,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=10,
        hh_prob_propensity_to_leave_isolation=1
    )
    model.nodes.node(1).isolated = True
    model.nodes.node(2).had_contacts_traced = True
    assert model.node_colour(model.nodes.node(1)) is "yellow"
    assert model.node_colour(model.nodes.node(2)) is "orange"
    assert model.node_colour(model.nodes.node(3)) is "white"


def test_onset_to_isolation_times(basic_model):

    # set up a model
    model = basic_model

    model.isolate_household(model.houses.household(1))
    node1 = model.nodes.node(1)

    assert model.onset_to_isolation_times() == [-node1.symptom_onset_time]

    assert model.onset_to_isolation_times(include_self_reports=False) == []


def test_infection_to_isolation_times(basic_model):

    # set up a model
    model = basic_model

    model.isolate_household(model.houses.household(1))

    assert model.infected_to_isolation_times() == [0]

    assert model.infected_to_isolation_times(include_self_reports=False) == []


def test_household_not_uptake_isolation():
    """Sets up a model where the households will not uptake isolation, attempts to isolate a household
    and checks that the household does not uptake the isolation
    """

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.8,
        household_haz_rate_scale=1,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=10,
        hh_prob_will_take_up_isolation=0
    )

    model.isolate_household(model.houses.household(1))

    assert model.houses.household(1).isolated is False
    assert model.houses.household(1).contact_traced is True
    
    # unpack the generator
    hh_node_list = [node for node in model.houses.household(1).nodes()]
    
    assert hh_node_list[0].isolated is False

def test_nodes_inherit_contact_traced_status():
    # If a household has been contact traced nodes should inherit this status
    # new household, contact trace it, new infection, assert new infection has contact traced status

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.8,
        household_haz_rate_scale=1,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=1,
    )

    # Give the household a size of at least 2
    model.houses.household(1).house_size = 6

    model.isolate_household(model.houses.household(1))

    model.new_within_household_infection(model.nodes.node(1), serial_interval = 2)

    assert model.nodes.node(2).contact_traced == True

def test_default_uptake_isolation():
    # tests that the default value for uptake isolation is 1, otherwise households never uptake isolation

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.8,
        household_haz_rate_scale=1,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        hh_prob_propensity_to_leave_isolation=0,
        prob_has_trace_app=0.0,
        starting_infections=1,
    )

    model.isolate_household(model.houses.household(1))

    assert model.houses.household(1).isolated is True
    assert model.houses.household(1).isolated is True

    nodes_in_hh = [node for node in model.houses.household(1).nodes()]
    node0 = nodes_in_hh[0]

    assert node0.isolated is True
    assert node0.contact_traced is True

def test_household_propensity_app():

    simulation = hct.household_sim_contact_tracing(
    haz_rate_scale=0.2,
    household_haz_rate_scale=0.8,
    overdispersion=0.32,
    contact_tracing_success_prob=0.9,
    contact_trace_delay_par=1.5,
    infection_reporting_prob=0.2,
    contact_trace=True,
    starting_infections=1,
    prob_has_trace_app=1,
    hh_propensity_to_use_trace_app=0)

    # No households have the propensity to use the trace app
    assert simulation.nodes.node(1).has_contact_tracing_app is False

    simulation = hct.household_sim_contact_tracing(
    haz_rate_scale=0.2,
    household_haz_rate_scale=0.8,
    overdispersion=0.32,
    contact_tracing_success_prob=0.9,
    contact_trace_delay_par=1.5,
    infection_reporting_prob=0.2,
    contact_trace=True,
    starting_infections=1,
    prob_has_trace_app=1,
    hh_propensity_to_use_trace_app=1)

    # No households have the propensity to use the trace app
    assert simulation.nodes.node(1).has_contact_tracing_app is True


def test_two_step():
    """The max tracing index is a function of whether or not we are doing two step tracing
    """


    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        household_haz_rate_scale=0.8,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        do_2_step=True
    )

    assert model.max_tracing_index == 2

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        household_haz_rate_scale=0.8,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
    )

    assert model.max_tracing_index == 1


def test_per_household_contact_reductions():

    simulation = hct.household_sim_contact_tracing(
    haz_rate_scale=0.2,
    household_haz_rate_scale=0.8,
    overdispersion=0.32,
    contact_tracing_success_prob=0.9,
    contact_trace_delay_par=1.5,
    infection_reporting_prob=0.2,
    contact_trace=True,
    reduce_contacts_by=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
    starting_infections=1,
    prob_has_trace_app=1,
    hh_propensity_to_use_trace_app=1)

    assert simulation.get_contact_rate_reduction(4) == 0.3

def test_two_step_index():

    simulation = hct.household_sim_contact_tracing(
    haz_rate_scale=0.2,
    household_haz_rate_scale=0.8,
    overdispersion=0.32,
    contact_tracing_success_prob=1,
    contact_trace_delay_par=0,
    infection_reporting_prob=0.2,
    contact_trace=True,
    reduce_contacts_by=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
    starting_infections=1,
    prob_has_trace_app=1,
    hh_propensity_to_use_trace_app=1)

    simulation.run_simulation(0)

    simulation.new_outside_household_infection(
        infecting_node=simulation.nodes.node(1),
        serial_interval=0
        )
    simulation.new_outside_household_infection(
        infecting_node=simulation.nodes.node(2),
        serial_interval=0
        )

    simulation.isolate_household(simulation.houses.household(1))

    simulation.propagate_contact_tracing(simulation.houses.household(1))
    simulation.propagate_contact_tracing(simulation.houses.household(2))

    assert simulation.houses.household(1).contact_tracing_index == 0
    assert simulation.houses.household(2).contact_tracing_index == 1
    assert simulation.houses.household(3).contact_tracing_index == 2

def test_two_step_index_update():

    simulation = hct.household_sim_contact_tracing(
    haz_rate_scale=0.2,
    household_haz_rate_scale=0.8,
    overdispersion=0.32,
    contact_tracing_success_prob=1,
    contact_trace_delay_par=0,
    infection_reporting_prob=0.2,
    contact_trace=True,
    reduce_contacts_by=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
    test_before_propagate_tracing=False,
    starting_infections=1,
    prob_has_trace_app=1,
    hh_propensity_to_use_trace_app=1)

    simulation.run_simulation(0)

    simulation.new_outside_household_infection(
        infecting_node=simulation.nodes.node(1),
        serial_interval=0
        )
    simulation.new_outside_household_infection(
        infecting_node=simulation.nodes.node(2),
        serial_interval=0
        )

    simulation.isolate_household(simulation.houses.household(1))

    simulation.propagate_contact_tracing(simulation.houses.household(1))
    simulation.propagate_contact_tracing(simulation.houses.household(2))

    # Symptom onset has now happened, therefore it should be an index 1 case
    # given there are no teting delays
    simulation.update_isolation()

    simulation.update_contact_tracing_index()

    simulation.nodes.node(2).symptom_onset_time=0

    simulation.update_contact_tracing_index()

    assert simulation.houses.household(1).contact_tracing_index == 0
    assert simulation.houses.household(2).contact_tracing_index == 0
    assert simulation.houses.household(3).contact_tracing_index == 1

def test_two_step_index_update_2nd_case():

    simulation = hct.household_sim_contact_tracing(
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

    simulation.run_simulation(0)

    simulation.new_outside_household_infection(
        infecting_node=simulation.nodes.node(1),
        serial_interval=0
        )
    simulation.new_outside_household_infection(
        infecting_node=simulation.nodes.node(2),
        serial_interval=0
        )

    simulation.isolate_household(simulation.houses.household(1))

    simulation.propagate_contact_tracing(simulation.houses.household(1))
    simulation.propagate_contact_tracing(simulation.houses.household(2))

    simulation.update_isolation()

    simulation.update_contact_tracing_index()

    simulation.nodes.node(3).symptom_onset_time=0

    simulation.update_contact_tracing_index()
    simulation.houses.household(3).contact_tracing_index

    assert simulation.houses.household(1).contact_tracing_index == 0
    assert simulation.houses.household(2).contact_tracing_index == 1
    assert simulation.houses.household(3).contact_tracing_index == 0

def test_count_non_recovered_nodes(basic_model):
    
    assert basic_model.count_non_recovered_nodes() == 1

# def test_run_simulation_detection_times():
#     assert False

