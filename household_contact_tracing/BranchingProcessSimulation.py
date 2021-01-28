from typing import Dict, Iterator, List, Optional, Tuple
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import networkx as nx
import scipy as s
import scipy.integrate as si
import math
from matplotlib.lines import Line2D
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or pydot")

# Code for demonstrating contact tracing on networks

# parameters for the generation time distribution
# mean 5, sd = 1.9
gen_shape = 2.826
gen_scale = 5.665

def weibull_pdf(t):
    out = (gen_shape / gen_scale) * (t / gen_scale)**(gen_shape - 1) * math.exp(-(t / gen_scale)**gen_shape)
    return out


def weibull_hazard(t):
    return (gen_shape / gen_scale) * (t / gen_scale)**(gen_shape - 1)


def weibull_survival(t):
    return math.exp(-(t / gen_scale)**gen_shape)


# Probability of a contact causing infection
def unconditional_hazard_rate(t, survive_forever):
    """
    Borrowed from survival analysis.

    To get the correct generation time distribution, set the probability
    of a contact on day t equal to the generation time distribution's hazard rate on day t

    Since it is not guaranteed that an individual will be infected, we use improper variables and rescale appropriately.
    The R0 scaling parameter controls this, as R0 is closely related to the probability of not being infected
    The relationship does not hold exactly in the household model, hence model tuning is required.

    Notes on the conditional variable stuff https://data.princeton.edu/wws509/notes/c7s1

    Returns
    The probability that a contact made on day t causes an infection.

    Notes:
    Currently this is using a weibull distribution, as an example.
    """
    unconditional_pdf = (1 - survive_forever) * weibull_pdf(t)
    unconditional_survival = (1 - survive_forever) * weibull_survival(t) + survive_forever
    return unconditional_pdf / unconditional_survival 


def current_hazard_rate(t, survive_forever):
    """Integrates over the unconditional hazard rate to get the probability of a contact causing infection on day t.

    survive_forever controls the probability that an infection never occurs, and is important to set R0.

    Arguments:
        t {int} -- current day
        survive_forever {float} -- rescales the hazard rate so that it is possible to not be infected
    """
    if t == 0:
        return si.quad(lambda t: unconditional_hazard_rate(t, survive_forever), 0, 0.5)[0]
    else:
        return si.quad(lambda t: unconditional_hazard_rate(t, survive_forever), t - 0.5, t + 0.5)[0]


def current_rate_infection(t):
    """Integrates over the unconditional hazard rate to get the probability of a contact causing infection on day t.

    survive_forever controls the probability that an infection never occurs, and is important to set R0.

    Arguments:
        t {int} -- current day
        survive_forever {float} -- rescales the hazard rate so that it is possible to not be infected
    """
    if t == 0:
        return si.quad(lambda t: weibull_pdf(t), 0, 0.5)[0]
    else:
        return si.quad(lambda t: weibull_pdf(t), t - 0.5, t + 0.5)[0]

def current_prob_leave_isolation(t, survive_forever):
    """Integrates over the unconditional hazard rate to get the probability of a contact causing infection on day t.

    survive_forever controls the probability that an infection never occurs, and is important to set R0.

    Arguments:
        t {int} -- current day
        survive_forever {float} -- rescales the hazard rate so that it is possible to not be infected
    """
    return si.quad(lambda t: unconditional_hazard_rate(t, survive_forever), t, t+1)[0]

def negbin_pdf(x, m, a):
    """
    We need to draw values from an overdispersed negative binomial distribution, with non-integer inputs. Had to
    generate the numbers myself in order to do this.
    This is the generalized negbin used in glm models I think.

    m = mean
    a = overdispertion 
    """
    A = math.gamma(x + 1 / a) / (math.gamma(x + 1) * math.gamma(1 / a))
    B = (1 / (1 + a * m))**(1 / a)
    C = (a * m / (1 + a * m))**x
    return A * B * C


def compute_negbin_cdf(mean, overdispersion, length_out):
    """
    Computes the overdispersed negative binomial cdf, which we use to generate random numbers by generating uniform(0,1)
    rv's.
    """
    pdf = [negbin_pdf(i, mean, overdispersion) for i in range(length_out)]
    cdf = [sum(pdf[:i]) for i in range(length_out)]
    return cdf


class Node:

    def __init__(
        self,
        nodes: 'NodeCollection',
        houses: 'HouseholdCollection',
        node_id: int,
        time_infected: int,
        generation: int,
        household: int,
        isolated: bool,
        symptom_onset_time: int,
        serial_interval: int,
        recovery_time: int,
        will_report_infection: bool,
        time_of_reporting: int,
        has_contact_tracing_app: bool,
        testing_delay: int,
        contact_traced: bool,
        had_contacts_traced=False,
        outside_house_contacts_made=0,
        spread_to: Optional[List[int]] = None,
        recovered=False
    ):
        """An object describing a single infection in the model.

        Args:
            nodes (NodeCollection): The NodeCollection that this node belongs to
            houses (HouseholdCollection): The HouseholdCollection that this node belongs to
            node_id (int): The ID of the node
            time_infected (int): The time at which the node was infected
            generation (int): The generation of the epidemic that this node belongs to
            household (int): The household id that of the household the node belongs to
            isolated (bool): Is the node currently isolated?
            symptom_onset_time (int): When the node develops symptoms
            serial_interval (int): The serial interval of the node
            recovery_time (int): The time at which the node is believed to be recovered
            will_report_infection (bool): Will the node report their symptoms?
            time_of_reporting (int): The time at which the node will report symptoms, and maybe book a test.
            has_contact_tracing_app (bool): Is the node using a contact tracing app?
            testing_delay (int): If the node gets tested, how long will it take?
            contact_traced (bool): Has the node been contact traced?
            had_contacts_traced (bool, optional): Has the node had their contact traced? Defaults to False.
            outside_house_contacts_made (int, optional): How many outside household contact has the node made? Defaults to 0.
            spread_to (Optional[List[int]], optional): Which nodes has this node infected. Defaults to None.
            recovered (bool, optional): has the node recovered? Defaults to False.
        """
        self.nodes = nodes
        self.houses = houses
        self.node_id = node_id
        self.time_infected = time_infected
        self.generation = generation
        self.household_id = household
        self.isolated = isolated
        self.symptom_onset_time = symptom_onset_time
        self.serial_interval = serial_interval
        self.recovery_time = recovery_time
        self.will_report_infection = will_report_infection
        self.time_of_reporting = time_of_reporting
        self.has_contact_tracing_app = has_contact_tracing_app
        self.testing_delay = testing_delay
        self.contact_traced = contact_traced
        self.had_contacts_traced = had_contacts_traced
        self.outside_house_contacts_made = outside_house_contacts_made
        self.spread_to = spread_to if spread_to else []
        self.spread_to_global_node_time_tuples = []
        self.recovered = recovered
        self.time_propagated_tracing = None
        self.propagated_contact_tracing = None

    def household(self) -> 'Household':
        """Returns the household object of the household that the node belongs to.

        Returns:
            Household: The nodes household
        """
        return self.houses.household(self.household_id)


class NodeCollection:

    def __init__(self, houses: 'HouseholdCollection'):
        """A collection object containing all infections.

        Args:
            houses (HouseholdCollection): The household collection object for all nodes.
        """
        self.G = nx.Graph()
        self.houses = houses

    def add_node(
        self,
        node_id: int,
        time: int,
        generation: int,
        household: int,
        isolated: bool,
        symptom_onset_time: int,
        serial_interval: int,
        recovery_time: int,
        will_report_infection: bool,
        time_of_reporting: int,
        has_contact_tracing_app: bool,
        testing_delay: int,
        contact_traced: bool) -> Node:
        """Adds a new node to the node collection.

        Args:
            node_id (int): The ID of the node
            time (int): The time at which the node was infected
            generation (int): The generation of the epidemic that this node belongs to
            household (int): The household id that of the household the node belongs to
            isolated (bool): Is the node currently isolated?
            symptom_onset_time (int): When the node develops symptoms
            serial_interval (int): The serial interval of the node
            recovery_time (int): The time at which the node is believed to be recovered
            will_report_infection (bool): Will the node report their symptoms?
            time_of_reporting (int): The time at which the node will report symptoms, and maybe book a test.
            has_contact_tracing_app (bool): Is the node using a contact tracing app?
            testing_delay (int): If the node gets tested, how long will it take?
            contact_traced (bool): Has the node been contact traced?

        Returns:
            Node: Adds a new node to the NodeCollection, with attributes as specified above.
        """

        self.G.add_node(node_id)

        node = Node(
            self,
            self.houses,
            node_id,
            time,
            generation,
            household,
            isolated,
            symptom_onset_time,
            serial_interval,
            recovery_time,
            will_report_infection,
            time_of_reporting,
            has_contact_tracing_app,
            testing_delay,
            contact_traced
        )
        self.G.nodes[node_id]['node_obj'] = node
        return node

    def node(self, node_id: int) -> Node:
        """Fetches a node object from the node collection.

        Args:
            node_id (int): The ID of the node you are trying to fetch.

        Returns:
            Node: The node object for the node with the specified node_id
        """
        return self.G.nodes[node_id]['node_obj']

    def all_nodes(self) -> Iterator[Node]:
        """Iterator that returns all nodes contained in the node collection

        Yields:
            Iterator[Node]: A node in NodeCollection    
        """
        return (self.node(n) for n in self.G)


class Household:

    def __init__(
        self,
        houses: 'HouseholdCollection',
        nodecollection: NodeCollection,
        house_id: int,
        house_size: int,
        time_infected: int,
        propensity: bool,
        hh_will_take_up_isolation: bool,
        generation: int,
        infected_by: int,
        infected_by_node: int,
        propensity_trace_app: bool
    ):
        """A Household object that stores informaton about the local within household epidemic
        and the current isolation status of the household. Households are not created without an infection,
        as uninfected households do not really exist in the household branching process.

        Args:
            houses (HouseholdCollection): A HouseholdCollection object that this household belongs to
            nodecollection (NodeCollection): A NodeCollection objet that the nodes in this household belong to
            house_id (int): The id of the household
            house_size (int): The number of individuals contained in the household
            time_infected (int): The time at which the first member of the household was infected
            propensity (bool): Does the household have the propensity to not fully adhere
            hh_will_take_up_isolation (bool): Will the household uptake isolation?
            generation (int): The household generation - different to the generations of individuals in the model
            infected_by (int): The id of the household who infected this household
            infected_by_node (int): The node which infected this household
            propensity_trace_app (bool): Does the household have the propensity to use the tracing app
        """
        self.houses = houses
        self.nodecollection = nodecollection
        self.house_id = house_id
        self.size = house_size
        self.time = time_infected
        self.susceptibles = house_size - 1
        self.isolated = False
        self.isolated_time = float('inf')
        self.propensity_to_leave_isolation = propensity
        self.hh_will_take_up_isolation = hh_will_take_up_isolation
        self.propensity_trace_app = propensity_trace_app
        self.contact_traced = False
        self.time_until_contact_traced = float('inf')
        self.contact_traced_household_ids: List[int] = []
        self.being_contact_traced_from: Optional[int] = None
        self.propagated_contact_tracing = False
        self.time_propagated_tracing: Optional[int] = None
        self.contact_tracing_index = 0
        self.generation = generation
        self.infected_by_id = infected_by
        self.spread_to_ids: List[int] = []
        self.node_ids: List[int] = []
        self.infected_by_node = infected_by_node
        self.within_house_edges: List[Tuple[int, int]] = []
        self.had_contacts_traced = False

    def nodes(self) -> Iterator[Node]:
        """An iterator over the nodes in the household.

        Yields:
            Iterator[Node]: A node in the household
        """
        return (self.nodecollection.node(n) for n in self.node_ids)

    def add_node_id(self, node_id: int):
        """Add a node_id to the list of node's who are in the household.

        Args:
            node_id (int): Node id to be added to the household.
        """
        self.node_ids.append(node_id)

    def contact_traced_households(self) -> Iterator['Household']:
        """A list of all households that have been contact traced from this household.

        Returns:
            tuple: A tuple containing all the household that have been traced from this household.
        """
        return (self.houses.household(hid) for hid in self.contact_traced_household_ids)

    def spread_to(self) -> Iterator['Household']:
        """Lists all household that have been infected from this household.

        Returns:
            tuple: A list of all household that have been infected from this household.
        """
        return (self.houses.household(hid) for hid in self.spread_to_ids)

    def infected_by(self) -> 'Household':
        """Returns the household object which spread the infection to this household.

        Returns:
            Household: The infecting household
        """
        if self.infected_by_id is None:
            return None
        return self.houses.household(self.infected_by_id)


class HouseholdCollection:

    def __init__(self, nodes: NodeCollection):
        """An object containing a collection of households. Allows for easy iteration on household objects.

        Args:
            nodes (NodeCollection): A linked node collection
        """
        self.house_dict: Dict[int, Household] = {}
        self.nodes = nodes

    def add_household(
        self,
        house_id: int,
        house_size: int,
        time_infected: int,
        propensity: bool,
        hh_will_take_up_isolation: bool,
        generation: int,
        infected_by: int,
        infected_by_node: int,
        propensity_trace_app: bool
    ) -> Household:
        """Add a new household to the household collection

        Args:
            house_id (int): The id of the new household
            house_size (int): The number of individuals living in the household 
            time_infected (int): At what time was the household infected (and therefore created)
            propensity (bool): Does the household have the propensity to not adhere?
            hh_will_take_up_isolation (bool): Will the household take up isolation?
            generation (int): The household generation
            infected_by (int): The id of the infecting household
            infected_by_node (int): The if od the infecting node
            propensity_trace_app (bool): Does the houseold have the propensity to use the tracing app?

        Returns:
            Household: Adds a household to the collection
        """
        new_household = Household(
            self, self.nodes, house_id,
            house_size, time_infected, propensity, hh_will_take_up_isolation,
            generation, infected_by, infected_by_node, propensity_trace_app
        )
        self.house_dict[house_id] = new_household
        return new_household

    def household(self, house_id) -> Household:
        """For a given household id, returns the corresponding household.

        Args:
            house_id (int): The id of the household you want to lookup.

        Returns:
            Household: The household corresponding to the desired household.
        """
        return self.house_dict[house_id]

    @property
    def count(self) -> int:
        """Returns the number of household in the collection

        Returns:
            int: Number of households in the collection
        """
        return len(self.house_dict)

    def all_households(self) -> Iterator[Household]:
        """Loop over all households.

        Yields:
            Iterator[Household]: [description]
        """
        return (self.household(hid) for hid in self.house_dict)

class household_sim_contact_tracing:
    # We assign each node a recovery period of 14 days, after 14 days the probability of causing a new infections is 0,
    # due to the generation time distribution
    # we don't actually model recovery as this is not important in our model - nodes will remember if they had symptoms.
    # instead this is more of a flag that we can stop simulating these nodes, since they are unlikely to do any more infecting
    effective_infectious_period = 14

    # Working out the parameters of the incubation period
    ip_mean = 4.83
    ip_var = 2.78**2
    ip_scale = ip_var / ip_mean
    ip_shape = ip_mean ** 2 / ip_var

    # Visual Parameters:
    contact_traced_edge_colour_within_house = "blue"
    contact_traced_edge_between_house = "magenta"
    default_edge_colour = "black"
    failed_contact_tracing = "red"
    app_traced_edge = "green"

    # Local contact probability:
    local_contact_probs = [0, 0.826, 0.795, 0.803, 0.787, 0.819]

    # The mean number of contacts made by each household
    total_contact_means = [7.238, 10.133, 11.419, 12.844, 14.535, 15.844]

    def __init__(self,
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
                 leave_isolation_prob=0):
        """A household contact tracing object capable of simulating household branching process and the contact tracing process.

        For this object, contact tracing is performed at this household level. Please see paper for a definition of what this means.

        There are several hardcoded parameters in this model. We have fixed this in future versions.

        Args:
            haz_rate_scale (float): A calibration parameter which controls infectiousness of outside household contacts
            contact_tracing_success_prob (float): The probability that a contact tracing attempt succeeds
            contact_trace_delay_par (float): The mean contact tracing delay. Contact tracing delays are poisson distributed.
            overdispersion (float): The overdispersion in the distributions of social contacts.
            infection_reporting_prob (float): The probability that an infection will report symptoms, and self isolate as a result.
            contact_trace (bool): If true, then contact tracing interventions are applied.
            household_haz_rate_scale (bool): A calibration parameter controlling the household secondary attack rate.
            do_2_step (bool, optional): Should two step on the household be used? Defaults to False.
            backwards_trace (bool, optional): Switches on, or off backwards tracing. Defaults to True.
            reduce_contacts_by (int, optional): Probability of each contact not occurring due to social distancing. Defaults to 0.
            prob_has_trace_app (int, optional): Probability that a node has the contact tracing application, if they live in a household with the propensity to use the app. Defaults to 0.
            hh_propensity_to_use_trace_app (int, optional): Probability that a household will have the propensity to use the contact tracing application. Defaults to 1.
            test_delay_mean (float, optional): The mean test delays. Defaults to 1.52.
            test_before_propagate_tracing (bool, optional): If false, then contact tracing is propagated upon symptom onset. Defaults to True.
            starting_infections (int, optional): Number of starting infections in the model. Defaults to 1.
            hh_prob_will_take_up_isolation (int, optional): Probability that a household will uptake isolation if required. Defaults to 1.
            hh_prob_propensity_to_leave_isolation (int, optional): Probability that a household will have the propensity to leave isolation early. Defaults to 0.
            leave_isolation_prob (int, optional): If a node is in a household with the propensity to leave isolation early, then this is the daily probability of leaving early. Defaults to 0.
        """
        # Probability of each household size
        house_size_probs = [0.294591195, 0.345336927, 0.154070081, 0.139478886, 0.045067385, 0.021455526]

        # Precomputing the cdf's for generating the overdispersed contact data
        self.cdf_dict = {
            1: compute_negbin_cdf(self.total_contact_means[0], overdispersion, 100),
            2: compute_negbin_cdf(self.total_contact_means[1], overdispersion, 100),
            3: compute_negbin_cdf(self.total_contact_means[2], overdispersion, 100),
            4: compute_negbin_cdf(self.total_contact_means[3], overdispersion, 100),
            5: compute_negbin_cdf(self.total_contact_means[4], overdispersion, 100),
            6: compute_negbin_cdf(self.total_contact_means[5], overdispersion, 100)
        }

        # Calculate the expected local contacts
        expected_local_contacts = [self.local_contact_probs[i] * i for i in range(6)]

        # Calculate the expected global contacts
        expected_global_contacts = np.array(self.total_contact_means) - np.array(expected_local_contacts)

        # Size biased distribution of households (choose a node, what is the prob they are in a house size 6, this is
        # biased by the size of the house)
        size_mean_contacts_biased_distribution = [(i + 1) * house_size_probs[i] * expected_global_contacts[i] for i in range(6)]
        total = sum(size_mean_contacts_biased_distribution)
        self.size_mean_contacts_biased_distribution = [prob / total for prob in size_mean_contacts_biased_distribution]

        # Parameter Inputs:
        self.haz_rate_scale = haz_rate_scale
        self.household_haz_rate_scale = household_haz_rate_scale
        self.contact_tracing_success_prob = contact_tracing_success_prob
        self.contact_trace_delay_par = contact_trace_delay_par
        self.overdispersion = overdispersion
        self.infection_reporting_prob = infection_reporting_prob
        self.contact_trace = contact_trace
        self.prob_has_trace_app = prob_has_trace_app
        self.hh_propensity_to_use_trace_app = hh_propensity_to_use_trace_app
        self.reduce_contacts_by = reduce_contacts_by
        self.do_2_step = do_2_step
        self.backwards_trace = backwards_trace
        self.test_before_propagate_tracing = test_before_propagate_tracing
        self.test_delay_mean = test_delay_mean
        self.starting_infections = starting_infections
        self.hh_prob_will_take_up_isolation = hh_prob_will_take_up_isolation
        self.hh_prob_propensity_to_leave_isolation = hh_prob_propensity_to_leave_isolation
        self.leave_isolation_prob = leave_isolation_prob
        if do_2_step:
            self.max_tracing_index = 2
        else:
            self.max_tracing_index = 1
        if type(self.reduce_contacts_by) in [tuple, list]:
            self.contact_rate_reduction_by_household = True
        else:
            self.contact_rate_reduction_by_household = False

        # Precomputing the infection probabilities for the within household epidemics.
        contact_prob = 0.8
        day_0_infection_prob = current_hazard_rate(0, self.household_haz_rate_scale)/contact_prob
        infection_probs = np.array(day_0_infection_prob)
        for day in range(1, 15):
            survival_function = (1 - infection_probs*contact_prob).prod()
            hazard = current_hazard_rate(day, self.household_haz_rate_scale)
            current_prob_infection = hazard * survival_function / contact_prob
            infection_probs = np.append(infection_probs, current_prob_infection)
        self.hh_infection_probs = infection_probs

        # Precomputing the global infection probabilities
        self.global_infection_probs = []
        for day in range(15):
            self.global_infection_probs.append(self.haz_rate_scale * current_rate_infection(day))

        # Calls the simulation reset function, which creates all the required dictionaries
        self.reset_simulation()

    def contact_trace_delay(self, app_traced_edge: bool) -> int:
        """Generates the contact tracing delay from the tracing delay distribution.

        Args:
            app_traced_edge (bool): Do both ends of the edge have the contact tracing app?

        Returns:
            int: The contact tracing delay in days.
        """
        if app_traced_edge:
            return 0
        else:
            return npr.poisson(self.contact_trace_delay_par)

    def incubation_period(self) -> int:
        """Generates a values from the incubation period distribution

        Returns:
            int: The incubation period in days
        """
        return round(npr.gamma(
            shape=self.ip_shape,
            scale=self.ip_scale))

    def testing_delay(self) -> int:
        """Draws the testing delay from the testing delay distribution

        Returns:
            int: testing delay in days
        """
        if self.test_before_propagate_tracing is False:
            return 0
        else:
            return round(npr.gamma(
                shape=self.test_delay_mean**2 / 1.11**2,
                scale=1.11**2 / self.test_delay_mean))

    def reporting_delay(self) -> int:
        """The delay from symptom onset to symptom reporting.

        Returns:
            int: delay in days
        """
        return round(npr.gamma(
            shape=2.62**2/2.38**2,
            scale=2.38**2/2.62))

    def hh_propensity_to_leave_isolation(self) -> bool:
        """Generates a boolean value of whether or not a household has the propensity to leave isolation

        Returns:
            bool: [description]
        """
        if npr.binomial(1, self.hh_prob_propensity_to_leave_isolation) == 1:
            return True
        else:
            return False
        
    def hh_will_take_up_isolation(self) -> bool:
        """Generates a boolean value of whether the household will uptake isolation or not

        Returns:
            bool: Will the household uptake isolation or not
        """
        if npr.binomial(1, self.hh_prob_will_take_up_isolation) == 1:
            return True
        else:
            return False

    def hh_propensity_use_trace_app(self) -> bool:
        """Generates a boolean value of whether or not the household will use the contact tracing app.

        If false, then no nodes in the household will use the contact tracing app.
        If true, then each node in the household uses the app with probability parameter prob_has_trace_app

        Returns:
            bool: Will 
        """
        if npr.binomial(1, self.hh_propensity_to_use_trace_app) == 1:
            return True
        else:
            return False

    def contacts_made_today(self, household_size: int) -> int:
        """Draw a the total number of social contacts made from the overdispersed negative binomial distribution, given a nodes household size.

        Args:
            household_size (int): The size of the household that the node belongs

        Returns:
            int: The total number of social contacts made by a node.
        """
        random = npr.uniform()
        cdf = self.cdf_dict[household_size]
        obs = sum([int(cdf[i] < random) for i in range(100)])
        return obs

    def size_of_household(self)->int:
        """Draws a household size from the size biased distribution of households.

        Returns:
            int: The size of a household
        """
        return npr.choice([1, 2, 3, 4, 5, 6], p=self.size_mean_contacts_biased_distribution)

    def has_contact_tracing_app(self)->bool:
        """Generates a boolean value of whether the node will use the contact tracing app or not

        Returns:
            bool: Does the node use the contact tracing app
        """
        return npr.binomial(1, self.prob_has_trace_app) == 1

    def count_non_recovered_nodes(self) -> int:
        """Returns the number of nodes not in the recovered state.

        If this is 0, then the epidemic has ended.

        Returns:
            int: -- Number of non-recovered nodes.
        """
        return len([node for node in self.nodes.all_nodes() if not node.recovered])

    def new_infection(self, node_count: int, generation: int, household_id: int, serial_interval: int=None, infecting_node: Node=None) -> Node:
        """Adds a new infection to the network.

        Args:
            node_count (int): The number of nodes currently in the network.
            generation (int): The generation of the node to be created. This should be 1 more than the node who infected it.
            household_id (int): The id of the household that the node belongs to.
            serial_interval (int, optional): The time that has passed since the infector was infected. Defaults to None.
            infecting_node (Node, optional): The infecting Node. Defaults to None.

        Returns:
            Node: The newly added node
        """
        # Symptom onset time
        symptom_onset_time = self.time + self.incubation_period()
        # When a node reports it's infection
        if npr.binomial(1, self.infection_reporting_prob) == 1:
            will_report_infection = True
            time_of_reporting = symptom_onset_time + self.reporting_delay()
        else:
            will_report_infection = False
            time_of_reporting = float('Inf')

        # We assign each node a recovery period of 21 days, after 21 days the probability of causing a new infections is
        # 0, due to the generation time distribution
        recovery_time = self.time + 14

        household = self.houses.household(household_id)

        # If the household has the propensity to use the contact tracing app, decide if the node uses the app.
        if household.propensity_trace_app:
            has_trace_app = self.has_contact_tracing_app()
        else:
            has_trace_app = False

        node = self.nodes.add_node(
            node_id=node_count,
            time=self.time,
            generation=generation,
            household=household_id,
            isolated=household.isolated,
            contact_traced=household.contact_traced,
            symptom_onset_time=symptom_onset_time,
            serial_interval=serial_interval,
            recovery_time=recovery_time,
            will_report_infection=will_report_infection,
            time_of_reporting=time_of_reporting,
            has_contact_tracing_app=has_trace_app,
            testing_delay=self.testing_delay(),
        )

        # Updates to the household dictionary
        # Each house now stores a the ID's of which nodes are stored inside the house, so that quarantining can be done at the household level
        household.node_ids.append(node_count)

        # A number of days may have passed since the house was isolated
        # We need to decide if the node has left isolation already, since it did not previously exist
        if household.isolated:
            days_isolated = int(self.time - household.isolated_time)
            for _ in range(days_isolated):
                self.decide_if_leave_isolation(node)

        node.infected_by_node = infecting_node

        if infecting_node:
            if infecting_node.household().house_id == household_id:
                node.locally_infected = True
        else:
            node.locally_infected = False

    def new_household(self, new_household_number: int, generation: int, infected_by: int, infected_by_node: int) -> Household:
        """Add a new household to the model

        Args:
            new_household_number (int): The id of the household to be added
            generation (int): The generation of the household to be added
            infected_by (int): The infecting household
            infected_by_node (int): The infecting node in the infecting household

        Returns:
            Household: the household which has been added to the model
        """
        house_size = self.size_of_household()

        propensity = self.hh_propensity_to_leave_isolation()

        propensity_trace_app = self.hh_propensity_use_trace_app()

        self.houses.add_household(
            house_id=new_household_number,
            house_size=house_size,
            time_infected=self.time,
            propensity=propensity,
            hh_will_take_up_isolation=self.hh_will_take_up_isolation(),
            generation=generation,
            infected_by=infected_by,
            infected_by_node=infected_by_node,
            propensity_trace_app=propensity_trace_app
        )

    def get_edge_between_household(self, house1: Household, house2: Household) -> tuple:
        """For two input households, find the edge between the nodes which connects the two households. 

        Args:
            house1 (Household): One household
            house2 (Household): A different connected household

        Returns:
            tuple: A 2 element tuple, where the first element is the a node in household 1, and the second element is a node in household 2, and there was a transmission between the two nodes.
        """
        for node1 in house1.nodes():
            for node2 in house2.nodes():
                if self.G.has_edge(node1.node_id, node2.node_id):
                    return (node1.node_id, node2.node_id)

    def is_edge_app_traced(self, edge) -> bool:
        """For a given edge between two nodes, if both ends of the edge are using the contact tracing app then returns true, as the edge will be app traced.

        Args:
            edge (bool): 2 element tuple, where each element is a node_id, and the nodes are connected because there was a transmission between them.

        Returns:
            bool: true if both ends of the edge are using the contact tracing app.
        """
        return self.nodes.node(edge[0]).has_contact_tracing_app and self.nodes.node(edge[1]).has_contact_tracing_app

    @property
    def active_infections(self):
        """
        Returns a list of nodes who have not yet recovered, and can still infect.
        These nodes may be isolated and not able to infect globally however.

        Returns:
            list: list of nodes able to infect
        """
        return [
            node for node in self.nodes.all_nodes()
            if not node.recovered
        ]

    def get_contact_rate_reduction(self, house_size):
        """For a house size input, returns a contact rate reduction

        Arguments:
            house_size {int} -- The household size
        """
        if self.contact_rate_reduction_by_household is True:
            return self.reduce_contacts_by[house_size - 1]
        else:
            return self.reduce_contacts_by

    def increment_infection(self):
        """
        This is the algorithm which increments the epidemic.

        We outline the following steps

        1) Loop through all non-recovered nodes
        2) Work out how many local and global contact that node makes on a given day
        3) For each of those contacts, compute how many of these contacts lead to transmission of the virus
        4) For each global transmission, create a new out of household infection (which requires the creation of a new household)
        5) For each local transmission, create a new within household transmission
        """

        for node in self.active_infections:
            household = node.household()

            # Extracting useful parameters from the node
            days_since_infected = self.time - node.time_infected

            outside_household_contacts = -1

            while outside_household_contacts < 0:

                # The number of contacts made that day
                contacts_made = self.contacts_made_today(household.size)

                # How many of the contacts are within the household
                local_contacts = npr.binomial(household.size - 1, self.local_contact_probs[household.size - 1])

                # How many of the contacts are outside household contacts
                outside_household_contacts = contacts_made - local_contacts

            # If node is isolating or has reported their symptoms and is staying at home
            if node.isolated:
                outside_household_contacts = 0
            else: 
                # If there is social distancing perform bernoulli thinning of the global contacts
                outside_household_contacts = npr.binomial(
                    outside_household_contacts,
                    1 - self.get_contact_rate_reduction(house_size=household.size)
                )

            # Within household, how many of the infections would cause new infections
            # These contacts may be made with someone who is already infected, and so they will again be thinned
            local_infective_contacts = npr.binomial(
                local_contacts,
                self.hh_infection_probs[days_since_infected]
            )

            for _ in range(local_infective_contacts):
                # A further thinning has to happen since each attempt may choose an already infected person
                # That is to say, if everyone in your house is infected, you have 0 chance to infect a new person in your house

                # A one represents a susceptibles node in the household
                # A 0 represents an infected member of the household
                # We choose a random subset of this vector of length local_infective_contacts to determine infections
                # i.e we are choosing without replacement
                household_composition = [1]*household.susceptibles + [0]*(household.size - 1 - household.susceptibles)
                within_household_new_infections = sum(npr.choice(household_composition, local_infective_contacts, replace=False))

                # If the within household infection is successful:
                for _ in range(within_household_new_infections):
                    self.new_within_household_infection(
                        infecting_node=node,
                        serial_interval=days_since_infected
                    )

            # Update how many contacts the node made
            node.outside_house_contacts_made += outside_household_contacts

            # How many outside household contacts cause new infections
            outside_household_new_infections = npr.binomial(outside_household_contacts, self.global_infection_probs[days_since_infected])

            for _ in range(outside_household_new_infections):
                self.new_outside_household_infection(
                    infecting_node=node,
                    serial_interval=days_since_infected)

                node_time_tuple = (nx.number_of_nodes(self.G), self.time)

                node.spread_to_global_node_time_tuples.append(node_time_tuple)

    def new_within_household_infection(self, infecting_node: Node, serial_interval: Optional[int]):
        """Creates a new infection within a household. This is achieved by adding a new Node, and updating the attributes of the household.

        Args:
            infecting_node (Node): The node which infected the the infection being added.
            serial_interval (Optional[int]): The amount of time that has passed since the infector was infected.
        """
        # Add a new node to the network, it will be a member of the same household that the node that infected it was
        node_count = nx.number_of_nodes(self.G) + 1

        # We record which node caused this infection
        infecting_node.spread_to.append(node_count)

        infecting_node_household = infecting_node.household()

        # Adds the new infection to the network
        self.new_infection(node_count=node_count,
                           generation=infecting_node.generation + 1,
                           household_id=infecting_node_household.house_id,
                           serial_interval=serial_interval,
                           infecting_node=infecting_node)

        # Add the edge to the graph and give it the default colour if the house is not traced/isolated
        self.G.add_edge(infecting_node.node_id, node_count)

        if self.nodes.node(node_count).household().isolated:
            self.G.edges[infecting_node.node_id, node_count].update({"colour": self.contact_traced_edge_colour_within_house})
        else:
            self.G.edges[infecting_node.node_id, node_count].update({"colour": self.default_edge_colour})

        # Decrease the number of susceptibles in that house by 1
        infecting_node_household.susceptibles -= 1

        # We record which edges are within this household for visualisation later on
        infecting_node_household.within_house_edges.append((infecting_node.node_id, node_count))

    def new_outside_household_infection(self, infecting_node: Node, serial_interval: Optional[int]):
        """Creates a new out of household infection. This is achieved by adding a new household to the model and adding a new infection to the model.
        The new infection lives in the the newly created household.

        Args:
            infecting_node (Node): The infecting Node
            serial_interval (Optional[int]): The amount of time that has passed since the node was infected.
        """
        # We assume all new outside household infections are in a new household
        # i.e: You do not infect 2 people in a new household
        # you do not spread the infection to a household that already has an infection
        self.house_count += 1
        node_count = nx.number_of_nodes(self.G) + 1
        infecting_household = infecting_node.household()

        # We record which node caused this infection
        infecting_node.spread_to.append(node_count)

        # We record which house spread to which other house
        infecting_household.spread_to_ids.append(self.house_count)

        # Create a new household, since the infection was outside the household
        self.new_household(new_household_number=self.house_count,
                           generation=infecting_household.generation + 1,
                           infected_by=infecting_node.household_id,
                           infected_by_node=infecting_node.node_id)

        # add a new infection in the house just created
        self.new_infection(node_count=node_count,
                           generation=infecting_node.generation + 1,
                           household_id=self.house_count,
                           serial_interval=serial_interval,
                           infecting_node=infecting_node)

        # Add the edge to the graph and give it the default colour
        self.G.add_edge(infecting_node.node_id, node_count)
        self.G.edges[infecting_node.node_id, node_count].update({"colour": "black"})

    def update_isolation(self):
        """Loops over nodes in the model, and updates their isolation status.

        This includes isolating household who have been contact traced and household that have had symptom onset.
        """
        # Update the contact traced status for all households that have had the contact tracing process get there
        [
            self.contact_trace_household(household)
            for household in self.houses.all_households()
            if household.time_until_contact_traced <= self.time
            and not household.contact_traced
        ]

        # Isolate all non isolated households where the infection has been reported (excludes those who will not take up isolation if prob <1)
        [
            self.isolate_household(node.household())
            for node in self.nodes.all_nodes()
            if node.time_of_reporting <= self.time
            and not node.isolated
            and not node.household().contact_traced
        ]


    def increment_contact_tracing(self):
        """This is the algorithm which increments the contact tracing process across the generated transmission network.
        
        The is the version of this algorithm that specifically applies household level contact tracing. See paper for details on the distinction
        between household level contact tracing and individual level contact tracing.

        We note that if testing is not required, then the testing delay is 0.

        The outline is as follows:
        1) If there are any traced households which develop symptoms, isolate these households.
        2) If a household reports possible infection (time of reporting), then contact tracing is propagated.
        3) If a household is contact traced and there is a symptom onset, then contact tracing is propagated.
        4) If two step contact tracing is employed, then contact tracing is propagated for all households that are distance 1 from a household with a known infection.
        """

        # Isolate all households under observation that now display symptoms (excludes those who will not take up isolation if prob <1)
        [
            self.isolate_household(node.household())
            for node in self.nodes.all_nodes()
            if node.symptom_onset_time <= self.time
            and node.contact_traced
            and not node.isolated
        ]

        # Look for houses that need to propagate the contact tracing because their test result has come back
        # Necessary conditions: household isolated, symptom onset + testing delay = time

        # Propagate the contact tracing for all households that self-reported and have had their test results come back
        [
            self.propagate_contact_tracing(node.household())
            for node in self.nodes.all_nodes()
            if node.time_of_reporting + node.testing_delay == self.time
            and not node.household().propagated_contact_tracing
        ]

        # Propagate the contact tracing for all households that are isolated due to exposure, have developed symptoms and had a test come back
        [
            self.propagate_contact_tracing(node.household())
            for node in self.nodes.all_nodes()
            if node.symptom_onset_time <= self.time
            and not node.household().propagated_contact_tracing
            and node.household().isolated_time + node.testing_delay <= self.time
        ]

        # Update the contact tracing index of households
        # That is, re-evaluate how far away they are from a known infected case (traced + symptom_onset_time + testing_delay)
        self.update_contact_tracing_index()

        if self.do_2_step:
            # Propagate the contact tracing from any households with a contact tracing index of 1
            [
                self.propagate_contact_tracing(household)
                for household in self.houses.all_households()
                if household.contact_tracing_index == 1
                and not household.propagated_contact_tracing
                and household.isolated
            ]

    def contact_trace_household(self, household: Household):
        """Applies the contact traced status to traced households.

        Nodes in the traced household will now be on the lookout for symptom onset and will report it earlier.
        If there is already symptom onset in the household, then the household goes into isolation.

        Args:
            household (Household): The household who will now have been contact traced.
        """
        # Update the house to the contact traced status
        household.contact_traced = True

        # Update the nodes to the contact traced status
        for node in household.nodes():
            node.contact_traced = True

        # Colour the edges within household
        [
            self.G.edges[edge[0], edge[1]].update({"colour": self.contact_traced_edge_colour_within_house})
            for edge in household.within_house_edges
        ]

        # If there are any nodes in the house that are symptomatic, isolate the house:
        symptomatic_nodes = [node for node in household.nodes() if node.symptom_onset_time <= self.time]
        if symptomatic_nodes != []:
            self.isolate_household(household)
        else:
            self.isolate_household(household)


    def perform_recoveries(self):
        """Loops over all nodes and performs recoveries if appropriate.
        """
        for node in self.nodes.all_nodes():
            if node.recovery_time <= self.time and not node.recovered:
                node.recovered = True


    def colour_node_edges_between_houses(self, house_to: Household, house_from: Household, new_colour):
        """Finds the edge between two households and applies a colour to the edge. This is primarily used when plotting.

        Args:
            house_to (Household): One end of an edge on the household hypergraph
            house_from (Household): The other end of an edge on the household hypergraph
            new_colour (str): The colour to be applied to the edge
        """
        # Annoying bit of logic to find the edge and colour it
        for node_1 in house_to.nodes():
            for node_2 in house_from.nodes():
                if self.nodes.G.has_edge(node_1.node_id, node_2.node_id):
                    self.nodes.G.edges[node_1.node_id, node_2.node_id].update({"colour": new_colour})


    def attempt_contact_trace_of_household(self, house_to: Household, house_from: Household, contact_trace_delay: int = 0):
        """Attempts to contact trace a household who had contact with the household who is propagating the contact
        tracing (house_from)

        Args:
            house_to (Household): The household who is being contact traced
            house_from (Household): The household who is attempting contact tracing
            contact_trace_delay (int, optional): How long the contact tracing attempt takes. Defaults to 0.
        """
        # Decide if the edge was traced by the app
        app_traced = self.is_edge_app_traced(self.get_edge_between_household(house_from, house_to))

        # Get the success probability
        if app_traced:
            success_prob = 1
        else:
            success_prob = self.contact_tracing_success_prob

        # is the trace successful
        if (npr.binomial(1, success_prob) == 1):
            # Update the list of traced households from this one
            house_from.contact_traced_household_ids.append(house_to.house_id)

            # Assign the household a contact tracing index, 1 more than it's parent tracer
            house_to.contact_tracing_index = house_from.contact_tracing_index + 1

            # work out the time delay
            contact_trace_delay = contact_trace_delay + self.contact_trace_delay(app_traced)
            proposed_time_until_contact_trace = self.time + contact_trace_delay

            # Get the current time until contact trace, and compare against the proposed time until contact trace
            # Note this starts as infinity
            # If the new proposed time is quicker, change the route
            if proposed_time_until_contact_trace < house_to.time_until_contact_traced:
                house_to.time_until_contact_traced = proposed_time_until_contact_trace
                house_to.being_contact_traced_from = house_from.house_id

            # Edge colouring
            if app_traced:
                self.colour_node_edges_between_houses(house_to, house_from, self.app_traced_edge)
            else:
                self.colour_node_edges_between_houses(house_to, house_from, self.contact_traced_edge_between_house)
        else:
            self.colour_node_edges_between_houses(house_to, house_from, self.failed_contact_tracing)


    def isolate_household(self, household: Household):
        """
        Isolates a house so that all infectives in that household may no longer infect others.

        If the house is being surveillance due to a successful contact trace, and not due to reporting symptoms,
        update the edge colour to display this.

        For households that were connected to this household, they are assigned a time until contact traced

        When a house has been contact traced, all nodes in the house are under surveillance for symptoms. When a node becomes symptomatic, the house moves to isolation status.
        """
        household.contact_traced = True

        for node in household.nodes():
            node.contact_traced = True

        # Households have a probability to take up isolation if traced
        if household.hh_will_take_up_isolation:
            
            # The house moves to isolated status if it has been assigned to take up isolation if trace, given a probability
            household.isolated = True
            # household.contact_traced = True
            household.isolated_time = self.time

            # Update every node in the house to the isolated status
            for node in household.nodes():
                node.isolated = True

            # Which house started the contact trace that led to this house being isolated, if there is one
            # A household may be being isolated because someone in the household self reported symptoms
            # Hence sometimes there is a None value for House which contact traced
            if household.being_contact_traced_from is not None:
                house_which_contact_traced = self.houses.household(household.being_contact_traced_from)
                
                # Initially the edge is assigned the contact tracing colour, may be updated if the contact tracing does not succeed
                if self.is_edge_app_traced(self.get_edge_between_household(household, house_which_contact_traced)):
                    self.colour_node_edges_between_houses(household, house_which_contact_traced, self.app_traced_edge)
                else:
                    self.colour_node_edges_between_houses(household, house_which_contact_traced, self.contact_traced_edge_between_house)
                        
                    # We update the colour of every edge so that we can tell which household have been contact traced when we visualise
            [
                self.G.edges[edge[0], edge[1]].update({"colour": self.contact_traced_edge_colour_within_house})
                for edge in household.within_house_edges
            ]


    def decide_if_leave_isolation(self, node: Node):
        """
        If a node lives in a household with the propensity to not adhere to isolation, then this
        function decides if the node will leave isolation, conditional upon how many days it's been
        since the node was isolated.

        Only makes sense to apply this function to isolated nodes, in a household with propensity to
        leave isolation
        """
        if npr.binomial(1, self.leave_isolation_prob) == 1:
            node.isolated = False


    def propagate_contact_tracing(self, household: Household):
        """Spreads contact tracing attempts at the household level.

        Contact tracing attempts are made to all households connected to the inputted households. By connected,
        we mean that there has been a transmission between the inputted household and a connected household.

        If the tracing attempt is successful, then a time_until_contact_tracing is achieved for the connected household.

        Args:
            household (Household): Household that will propagate contact tracing attempts to all connected households
        """
        # update the propagation data
        household.propagated_contact_tracing = True
        household.time_propagated_tracing = self.time

        # Contact tracing attempted for the household that infected the household currently propagating the infection
        
        infected_by = household.infected_by()

        # If infected by = None, then it is the origin node, a special case
        if self.backwards_trace is True:
            if infected_by and not infected_by.isolated:
                self.attempt_contact_trace_of_household(infected_by, household)

        # Contact tracing for the households infected by the household currently traced
        child_households_not_traced = [h for h in household.spread_to() if not h.isolated]
        for child in child_households_not_traced:
            self.attempt_contact_trace_of_household(child, household)


    def update_contact_tracing_index(self):
        """The contact tracing index is used when calculating two-step contact tracing.

        The contact tracing index is the distance the household is from a known infection. The distance
        is calculated on the household hypergraph.


        For one step tracing, only households with a known infection propagate contact tracing.
        For two step tracing, households that have a contact tracing index of 1 also propagate contact tracing. 
        """ 
        for household in self.houses.all_households():
            # loop over households with non-zero indexes, those that have been contact traced but with
            if household.contact_tracing_index != 0:
                for node in household.nodes():

                    # Necessary conditions for an index 1 household to propagate tracing:
                    # The node must have onset of symptoms
                    # The node households must be isolated
                    # The testing delay must be passed
                    # The testing delay starts when the house have been isolated and symptoms have onset
                    critical_time = max(node.symptom_onset_time, household.isolated_time)

                    if critical_time + node.testing_delay <= self.time:
                        household.contact_tracing_index = 0

                        for index_1_hh in household.contact_traced_households():
                            if index_1_hh.contact_tracing_index == 2:
                                index_1_hh.contact_tracing_index = 1


    def update_adherence_to_isolation(self):
        """Loops over nodes currently in quarantine, and updates whether they are currently adhering to
        quarantine, if their household has the propensity to not adhere.

        Once a node leaves quarantine, it does not return to quarantine.
        """
        [
            self.decide_if_leave_isolation(node)
            for node in self.nodes.all_nodes()
            if node.isolated
            and node.household().propensity_to_leave_isolation
        ]


    def isolate_self_reporting_cases(self):
        """Applies the isolation status to nodes who have reached their self-report time.
        They may of course decide to not adhere to said isolation, or may be a member of a household
        who will not uptake isolation
        """
        for node in self.nodes.all_nodes():

            if node.household().hh_will_take_up_isolation:
                 if node.time_of_reporting == self.time:
                    node.isolated = True


    def release_nodes_from_quarantine(self):
        """If a node has completed the quarantine according to the following rules, they are released from
        quarantine.

        You are released from isolation if:
            * it has been 7 days since your symptoms onset
            and
            * it has been a minimum of 14 days since your household was isolated
        """
        for node in self.nodes.all_nodes():
            if self.time >= node.symptom_onset_time + 7 and self.time >= node.household().isolated_time + 14:
                node.isolated = False


    def simulate_one_day(self):
        """The main simulation loop of the model. Performs one days worth of infections, and then 1 days worth of contact
        tracing.
        """
        # perform a days worth of infections
        self.increment_infection()
        # isolate nodes reached by tracing, isolate nodes due to self-reporting
        self.update_isolation()
        # propagate contact tracing
        for _ in range(5):
            self.increment_contact_tracing()
        # node recoveries
        self.perform_recoveries()
        # release nodes from quarantine the time has arrived
        self.release_nodes_from_quarantine()
        # nodes that have self-reported and therefore have booked a test self-isolate
        # although it is possible that they may not adhere to this isolation
        self.isolate_self_reporting_cases()
        # update whether nodes are still adhering to quarantine
        self.update_adherence_to_isolation()
        # increment time
        self.time += 1


    def reset_simulation(self):
        """
        Returns the simulation to it's initially specified values
        """

        self.time = 0

        # Stores information about the contact tracing that has occurred.
        self.contact_tracing_dict = {
            "contacts_to_be_traced": 0,         # connections made by nodes that are contact traced and symptomatic
            "possible_to_trace_contacts": 0,    # contacts that are possible to trace assuming a failure rate, not all connections will be traceable
            "total_traced_each_day": [0],       # A list recording the the number of contacts added to the system each day
            "daily_active_surveillances": [],   # A list recording how many surveillances were happening each day
            "currently_being_surveilled": 0,    # Ongoing surveillances
            "day_800_cases_traced": None        # On which day was 800 cases reached
        }

        # Create the empty graph - we add the houses properly below
        self.nodes = NodeCollection(None)

        # Stores information about the households.
        self.houses = HouseholdCollection(self.nodes)
        self.nodes.houses = self.houses

        # make things available as before
        self.G = self.nodes.G

        # Create first household
        self.house_count = 0

        # Initial values
        node_count = 1
        generation = 0

        # Create the starting infectives
        for _ in range(self.starting_infections):
            self.house_count += 1
            node_count = nx.number_of_nodes(self.G) + 1
            self.new_household(self.house_count, 1, None, None)
            self.new_infection(node_count, generation, self.house_count)

    def run_simulation_detection_times(self):
        """Runs the model until the first case is detected.

        Possible end reason: extinction, infection detected. 
        """
        
         # Create all the required dictionaries and reset parameters
        self.reset_simulation()

        # For recording the number of cases over time
        self.total_cases = []

        # Initial values
        self.end_reason = ''
        self.timed_out = False
        self.extinct = False
        self.day_extinct = -1

        nodes_reporting_infection = [
            node
            for node in self.nodes.all_nodes() 
            if (node.time_of_reporting + node.testing_delay == self.time)
        ]

        currently_infecting = len([node for node in self.nodes.all_nodes() if not node.recovered])

        while self.end_reason == '':

            nodes_reporting_infection = [
                node
                for node in self.nodes.all_nodes() 
                if (node.time_of_reporting + node.testing_delay == self.time)
            ]

            # This chunk of code executes a days worth on infections and contact tracings
            node_count = nx.number_of_nodes(self.G)
            self.simulate_one_day()
            
            self.total_cases.append(node_count)

            # While loop ends when there are no non-isolated infections
            currently_infecting = len([node for node in self.nodes.all_nodes() if not node.recovered])

            if currently_infecting == 0:
                self.end_reason = 'extinct'
                self.died_out = True
                self.day_extinct = self.time

            if len(nodes_reporting_infection) != 0:
                self.end_reason = 'infection_detected'

        # Infection Count output
        self.inf_counts = self.total_cases


    def run_simulation(self, time_out: int, stop_when_5000_infections: bool=False):
        """Runs a simulation for a fixed period of time, or until 5000 infections reached.

        Args:
            time_out ([type]): time at which the simulations stops running.
            stop_when_5000_infections (bool, optional): If true, stops the simulation when there are 5000 active infections. Defaults to False.
        """

        # Create all the required dictionaries and reset parameters
        self.reset_simulation()

        # For recording the number of cases over time
        self.total_cases = []

        # Initial values
        self.end_reason = ''
        self.timed_out = False
        self.extinct = False
        self.day_extinct = -1

        # Sometimes you want to set up the simulation but not run it
        # In this case, set the time_out to 0
        if self.time == time_out:
            self.end_reason = 'timed_out'
            self.timed_out = True

        # While loop ends when there are no non-isolated infections
        currently_infecting = len([node for node in self.nodes.all_nodes() if not node.recovered])

        while self.end_reason == '':

            # This chunk of code executes a days worth on infections and contact tracings
            node_count = nx.number_of_nodes(self.G)
            self.simulate_one_day()

            self.house_count = self.houses.count
            self.total_cases.append(node_count)

            # While loop ends when there are no non-isolated infections
            currently_infecting = len([node for node in self.nodes.all_nodes() if not node.recovered])

            if currently_infecting == 0:
                self.end_reason = 'extinct'
                self.died_out = True
                self.day_extinct = self.time

            if self.time == time_out:
                self.end_reason = 'timed_out'
                self.timed_out = True

            if stop_when_5000_infections is True and currently_infecting > 5000:
                self.end_reason = 'more_than_5000'
                self.timed_out = True

        # Infection Count output
        self.inf_counts = self.total_cases

    def onset_to_isolation_times(self, include_self_reports=True):
        """Returns the time at which isolation occurs.

        Args:
            include_self_reports (bool, optional): Should nodes who self report, and therefore are not contact traced be added. Defaults to True.

        Returns:
            list[int]: A list of onset to isolation times.
        """
        if include_self_reports:
            return [
                node.household().isolated_time - node.symptom_onset_time
                for node in self.nodes.all_nodes()
                if node.isolated
            ]
        else:
            return [
                node.household().isolated_time - node.symptom_onset_time
                for node in self.nodes.all_nodes()
                if node.isolated
                and node.household().being_contact_traced_from is not None
            ]

    def infected_to_isolation_times(self, include_self_reports=True):
        """Returns the time at which isolation occurs.

        Args:
            include_self_reports (bool, optional): Should nodes who self report, and therefore are not contact traced be added. Defaults to True.

        Returns:
            list[int]: A list of onset to isolation times.
        """
        if include_self_reports:
            return [
                node.household().isolated_time - node.time_infected
                for node in self.nodes.all_nodes()
                if node.isolated
            ]
        else:
            return [
                node.household().isolated_time - node.time_infected
                for node in self.nodes.all_nodes()
                if node.isolated
                and node.household().being_contact_traced_from is not None
            ]


    def make_proxy(self, clr, **kwargs):
        """Used to draw the lines we use in the draw network legend.

        Arguments:
            clr {str} -- the colour of the line to be drawn.

        Returns:
            Line2D -- A Line2D object to be passed to the 
        """
        return Line2D([0, 1], [0, 1], color=clr, **kwargs)


    def node_colour(self, node: Node):
        """Returns a node colour, given the current status of the node.

        Arguments:
            node {int} -- The node id

        Returns:
            str -- The colour assigned
        """
        if node.isolated is True:
            return "yellow"
        elif node.had_contacts_traced:
            return "orange"
        else:
            return "white"


    def draw_network(self):
        """Draws the network generated by the model"""

        node_colour_map = [self.node_colour(node) for node in self.nodes.all_nodes()]

        # The following chunk of code draws the pretty branching processes
        edge_colour_map = [self.nodes.G.edges[edge]["colour"] for edge in self.nodes.G.edges()]

        # Legend for explaining edge colouring
        proxies = [
            self.make_proxy(clr, lw=1) for clr in (
                self.default_edge_colour,
                self.contact_traced_edge_colour_within_house,
                self.contact_traced_edge_between_house,
                self.app_traced_edge,
                self.failed_contact_tracing
            )
        ]
        labels = (
            "Transmission, yet to be traced",
            "Within household contact tracing",
            "Between household contact tracing",
            "App traced edge",
            "Failed contact trace"
        )

        node_households = {}
        for node in self.nodes.all_nodes():
            node_households.update({node.node_id: node.household_id})

        self.pos = graphviz_layout(self.G, prog='twopi')
        plt.figure(figsize=(10, 10))

        nx.draw(
            self.nodes.G,
            #self.pos,
            node_size=150, alpha=0.75, node_color=node_colour_map, edge_color=edge_colour_map,
            labels=node_households
        )
        plt.axis('equal')
        plt.title("Household Branching Process with Contact Tracing")
        plt.legend(proxies, labels)


class uk_model(household_sim_contact_tracing):

    def __init__(self,
        haz_rate_scale,
        contact_tracing_success_prob,
        contact_trace_delay_par,
        overdispersion,
        infection_reporting_prob,
        contact_trace,
        household_haz_rate_scale,
        number_of_days_to_trace_backwards=2,
        number_of_days_to_trace_forwards=7,
        reduce_contacts_by=0,
        prob_has_trace_app=0,
        hh_propensity_to_use_trace_app=1,
        test_delay_mean=1.52,
        test_before_propagate_tracing=True,
        probable_infections_need_test=True,
        backwards_tracing_time_limit=None,
        starting_infections=1,
        hh_prob_will_take_up_isolation=1,
        hh_prob_propensity_to_leave_isolation=0,
        leave_isolation_prob=0.0,
        recall_probability_fall_off=1.0):
    
        """A household contact tracing object capable of simulating household branching process and the contact
        tracing process.

        For this object, contact tracing is performed at the individual level. Contact tracing is only propagated when an individual tests positive or develops symptoms

        Args:
            haz_rate_scale (float): A calibration parameter which controls infectiousness of outside household contacts
            contact_tracing_success_prob (float): The probability that a contact tracing attempt succeeds
            contact_trace_delay_par (float): The mean contact tracing delay. Contact tracing delays are poisson distributed.
            overdispersion (float): The overdispersion in the distributions of social contacts.
            infection_reporting_prob (float): The probability that an infection will report symptoms, and self isolate as a result.
            contact_trace (bool): If true, then contact tracing interventions are applied.
            household_haz_rate_scale (bool): A calibration parameter controlling the household secondary attack rate.
            number_of_days_to_trace_backwards (int): Infections are not traced if they occur more than this time days prior to symptom onset.
            number_of_days_to_trace_forwards (int): Infections are not traced if they occur more than this days after symptom onset.
            backwards_trace (bool, optional): Switches on, or off backwards tracing. Defaults to True.
            reduce_contacts_by (int, optional): Probability of each contact not occurring due to social distancing. Defaults to 0.
            prob_has_trace_app (int, optional): Probability that a node has the contact tracing application, if they live in a household with the propensity to use the app. Defaults to 0.
            hh_propensity_to_use_trace_app (int, optional): Probability that a household will have the propensity to use the contact tracing application. Defaults to 1.
            test_delay_mean (float, optional): The mean test delays. Defaults to 1.52.
            test_before_propagate_tracing (bool, optional): If false, then contact tracing is propagated upon symptom onset. Defaults to True.
            probable_infections_need_test (bool, optional): If false, then if an individual is in a household with a confirmed infections and develops symptoms, they do not need to be tested. Defaults to True
            starting_infections (int, optional): Number of starting infections in the model. Defaults to 1.
            hh_prob_will_take_up_isolation (int, optional): Probability that a household will uptake isolation if required. Defaults to 1.
            hh_prob_propensity_to_leave_isolation (int, optional): Probability that a household will have the propensity to leave isolation early. Defaults to 0.
            leave_isolation_prob (float, optional): If a node is in a household with the propensity to leave isolation early, then this is the daily probability of leaving early. Defaults to 0.
            recall_probability_fall_off: The daily probability that a contact is remembered. Defaults to 1.0 (implying that all contacts are remembered)
        """

        super().__init__(
            haz_rate_scale=haz_rate_scale,
            contact_tracing_success_prob=contact_tracing_success_prob,
            contact_trace_delay_par=contact_trace_delay_par,
            overdispersion=overdispersion,
            infection_reporting_prob=infection_reporting_prob,
            contact_trace=contact_trace,
            household_haz_rate_scale=household_haz_rate_scale,
            do_2_step=False,
            backwards_trace=True,
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
        
        self.probable_infections_need_test = probable_infections_need_test
        self.backwards_tracing_time_limit = backwards_tracing_time_limit
        if self.backwards_tracing_time_limit is None:
            self.backwards_tracing_time_limit = float('inf')
        self.number_of_days_to_trace_backwards = number_of_days_to_trace_backwards
        self.number_of_days_to_trace_forwards = number_of_days_to_trace_forwards
        self.recall_probability_fall_off = recall_probability_fall_off

    def increment_contact_tracing(self):
        """This is the algorithm which increments the contact tracing process across the generated transmission network.
        
        The is the version of this algorithm that specifically applies individual level contact tracing. See paper for details on the distinction
        between household level contact tracing and individual level contact tracing.

        We note that if testing is not required, then the testing delay is 0.

        The outline is as follows:
        1) If there are any individuals in traced households who develop symptoms, isolate these households.
        2) If a household reports possible infection (time of reporting), then contact tracing is propagated.
        3) If an individual tests positive, then contact tracing is propagated to their contacts.
        """

        # Isolate all households under observation that now display symptoms (excludes those who will not take up isolation if prob <1)
        [
            self.isolate_household(node.household())
            for node in self.nodes.all_nodes()
            if node.symptom_onset_time <= self.time
            and node.contact_traced
            and not node.isolated
        ]


        # Look for nodes in households that have been isolated, and a node in that household has had symptom onset
        # and the test result has come back. Then this is a list of households where the infection has been confirmed
        # and the node has not already propagated contact tracing
        # Households with symptoms are isoalt
        households_with_confirmed_infection = [
            node.household()
            for node in self.nodes.all_nodes()
            if node.household().isolated
            and node.household().isolated_time + node.testing_delay <= self.time
        ]

        # Remove duplicates
        households_with_confirmed_infection = list(set(households_with_confirmed_infection))

        # Propagate the contact tracing for nodes that have had symptom onset in a household that has a confirmed infection
        
        for household in households_with_confirmed_infection:
            for node in household.nodes():

                # A node is only tested when their household has been isolated and they have had symptom onset
                node_positive_test_time = max(node.symptom_onset_time, node.household().isolated_time) + node.testing_delay

                if not node.propagated_contact_tracing and node.symptom_onset_time <= self.time and not self.probable_infections_need_test:
                    self.propagate_contact_tracing(node)
                elif not node.propagated_contact_tracing and node_positive_test_time <= self.time and self.probable_infections_need_test:
                    self.propagate_contact_tracing(node)
    
    def propagate_contact_tracing(self, node: Node):
        """Propagates individual level contact tracing.

        Contact tracing attempts are propagated to all nodes connected to he inputted Node.

        Args:
            node (Node): The node who is propagating contact tracing attempts to connected nodes.
        """
        # update the propagation data
        node.propagated_contact_tracing = True
        node.time_propagated_tracing = self.time

        # Contact tracing attempted for the household that infected the household currently propagating the infection
        infected_by_node = node.infected_by_node

        days_since_symptom_onset = self.time - node.symptom_onset_time

        # Determine if the infection was so long ago, that it is not worth contact tracing
        if days_since_symptom_onset > self.backwards_tracing_time_limit:
            time_limit_hit = True
        else:
            time_limit_hit = False

        # If the node was globally infected, we are backwards tracing and the infecting node is not None
        if not node.locally_infected and self.backwards_trace and infected_by_node and not time_limit_hit:

            # if the infector is not already isolated and the time the node was infected captured by going backwards
            # the node.time_infected is when they had a contact with their infector.
            if  not infected_by_node.isolated and node.time_infected >= self.time - self.number_of_days_to_trace_backwards:

                # Then attempt to contact trace the household of the node that infected you
                self.attempt_contact_trace_of_household(
                    house_to=infected_by_node.household(),
                    house_from=node.household(),
                    days_since_contact_occurred=self.time - node.time_infected
                    )

        # spread_to_global_node_time_tuples stores a list of tuples, where the first element is the node_id
        # of a node who was globally infected by the node, and the second element is the time of transmission
        for global_infection in node.spread_to_global_node_time_tuples:
            
            # Get the child node_id and the time of transmission/time of contact
            child_node_id, time = global_infection

            child_node = self.nodes.node(child_node_id)

            # If the node was infected 2 days prior to symptom onset, or 7 days post and is not already isolated
            if time > node.symptom_onset_time - self.number_of_days_to_trace_backwards and time < node.symptom_onset_time + self.number_of_days_to_trace_forwards and not child_node.isolated:

                self.attempt_contact_trace_of_household(
                    house_to=child_node.household(),
                    house_from=node.household(),
                    days_since_contact_occurred=self.time - time
                    )


    def attempt_contact_trace_of_household(self, house_to: Household, house_from: Household, days_since_contact_occurred: int, contact_trace_delay: int = 0):
        """Attempts to contact trace a household who had contact with the household who is propagating the contact
        tracing (house_from)

        Args:
            house_to (Household): The household who is being contact traced
            house_from (Household): The household who is attempting contact tracing
            contact_trace_delay (int, optional): How long the contact tracing attempt takes. Defaults to 0.
        """

        # Decide if the edge was traced by the app
        app_traced = self.is_edge_app_traced(self.get_edge_between_household(house_from, house_to))

        # Get the success probability
        if app_traced:
            success_prob = 1
        else:
            success_prob = self.contact_tracing_success_prob * self.recall_probability_fall_off ** days_since_contact_occurred

        # is the trace successful
        if (npr.binomial(1, success_prob) == 1):

            # Update the list of traced households from this one
            house_from.contact_traced_household_ids.append(house_to.house_id)

            # Assign the household a contact tracing index, 1 more than it's parent tracer
            house_to.contact_tracing_index = house_from.contact_tracing_index + 1

            # work out the time delay
            contact_trace_delay = contact_trace_delay + self.contact_trace_delay(app_traced)
            proposed_time_until_contact_trace = self.time + contact_trace_delay

            # Get the current time until contact trace, and compare against the proposed time until contact trace
            # Note this starts as infinity
            # If the new proposed time is quicker, change the route
            if proposed_time_until_contact_trace < house_to.time_until_contact_traced:
                house_to.time_until_contact_traced = proposed_time_until_contact_trace
                house_to.being_contact_traced_from = house_from.house_id

            # Edge colouring
            if app_traced:
                self.colour_node_edges_between_houses(house_to, house_from, self.app_traced_edge)
            else:
                self.colour_node_edges_between_houses(house_to, house_from, self.contact_traced_edge_between_house)
        else:
            self.colour_node_edges_between_houses(house_to, house_from, self.failed_contact_tracing)


    def new_infection(self, node_count: int, generation: int, household_id: int, serial_interval=None, infecting_node=None):
        """Adds a new infection to the network. This is a specific version with 
        some attributes at the node level, as this is required for individual level
        contact tracing.

        Args:
            node_count (int): The number of nodes currently in the network.
            generation (int): The generation of the node to be created. This should be 1 more than the node who infected it.
            household_id (int): The id of the household that the node belongs to.
            serial_interval (int, optional): The time that has passed since the infector was infected. Defaults to None.
            infecting_node (Node, optional): The infecting Node. Defaults to None.

        Returns:
            Node: The newly added node
        """

        super().new_infection(
            node_count=node_count,
            generation=generation,
            household_id=household_id,
            serial_interval=serial_interval,
            infecting_node=infecting_node)

        node = self.nodes.node(node_count)
        node.locally_infected = False
        if infecting_node:
            if infecting_node.household().house_id == household_id:
                node.locally_infected = True
            else:
                node.locally_infected = False
        else:
            node.locally_infected = False

        node.propagated_contact_tracing = False


class model_calibration(household_sim_contact_tracing):

    def estimate_secondary_attack_rate(self):
        """
        For the inputted parameters, computes the household secondary attack rate.
        The number of samples is equal to the number of starting infections in the model.
        """

        # Reset the simulation to it's initial state
        self.reset_simulation()

        # Initial households are allowed to run the household epidemics
        starting_households = list(range(1, self.starting_infections))

        while len(self.active_infections) is not 0:

            # Increment the infection process
            self.increment_infection()

            # recover nodes that need it
            self.perform_recoveries()

            # set any node that was an outside-household infection to the recovered state, so that they are not simulated.
            for node in self.nodes.all_nodes():
                if node.household_id not in starting_households and not node.recovered:
                    node.recovered = True

            self.time += 1

        total_infected = sum([
            len(self.houses.household(house_id).node_ids) - 1
            for house_id in starting_households
        ])

        total_exposed = sum([
            self.houses.household(house_id).size - 1
            for house_id in starting_households
        ])

        return total_infected/total_exposed
