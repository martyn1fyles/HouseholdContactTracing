# household-contact-tracing

## Introduction
household-contact-tracing is a model which aims to produce estimates of the effectiveness of contact tracing for the SARS-CoV-2 epidemic. As household structure is particularly important for the SARS-CoV-2 epidemic, we have paid particular attention to the interactions between contact tracing and the household structure. For example, if one person in a household tests positive it is reasonable to expect that all other members of the household immediately take up isolation. For other recent contacts of a case, they will be contact traced following a random delay.

## What's in the repository?
A tutorial notebook in notebooks > tutorial.ipynb has been created which provides examples of how to calibrate the model and to perform simulations.

We have provided the code required to reproduce the results in our paper on household contact tracing.

## Mathematical details
The SARS-CoV-2 epidemic is modelled using a household branching process. We consider this household branching process to generate a time-indexed transmission network, with a defined household hypernetwork. Contact tracing spreads across the time-indexed transmission network as a type of "superinfection". Nodes that are superinfected by contact tracing have reduced ability to transmit. For more details, please see the supplementary information of our paper.

# Contact tracing
Two different kinds of contact tracing are considered:
1) Household level contact tracing - if an individual tests positive, then as it is possible that other members of the household are positive but pre-symptomatic or asymptomatic, all members of the household have their contacts traced.
2) Individual level contact tracing - if an individual tests positive, then their contacts are traced. Members of their household are on the lookout for symptom onset.

This results in two main objects; household_contact_tracing (household level contact tracing), and uk_model (individual level contact tracing). These objects contain the majority of the functionality of the package. Unfortunately, there are a large number of parameters available to choose when setting up a model.

A calibrate_model object is also used when trying to estimate epidemic parameters for calibration purposes - this is used to compute the household secondary attack rate.
## Epidemic parameters
We focus on parameters that relate specifically to the epidemic. There are 3 key parameters;

1) household_haz_rate_scale - controls the infectiousness of within household contacts. This value is Essentially the pairwise survival probability.
2) haz_rate_scale - controls the infectiousness of outside household contacts.
3) reduce_contacts_by - a measure of social distancing. Each day, a node attempts to make contacts according to a baseline model of social contacts. To model social distancing, each contacts occurs with probability reduce_contacts_by.

## Contact tracing parameters   
The majority of these are self explanatory, and documented in the code, see help(uk_model) for example.

The majority parameters are shared by the two model, and for the most part are self-explanatory. The uk_model has a couple of additional parameters; number_of_days_to_trace_backwards, number_of_days_to_trace_forwards, probable_infections_need_test.

These are included to allow modelling of some scenarios that are more specific to individual level contact tracing;
*   number_of_day_to_trace_backwards - contacts are attempted to be traced if they occur <= number_of_days_to_trace_backwards days prior to symptom onset.
*   number_of_days_to_trace_forwards - contact are attempted to be traced if they occur <= number_of_days_to_trace_forwards days after symptom onset.
*   probable_infections_need_test - if an individual is in a household where another individual has tested positive, and they themselves develop symptoms, they do not need to be tested if probable_infections_need_test is False.

### App based tracing parameters
We assume that the contact tracing app is able to trace individuals with probability 1 and delay of 0 if both individuals use the tracing app.

The parameters associated with app based tracing are:
*   household_propensity_to_use_tracing_app - clusters app uptake behaviour by households. This parameter controls the probability that the members of a household will have the propensity to use the tracing app. If a nodes household does not have the propensity to use the tracing app, then they will not use the tracing app regardless of prob_has_trace_app. We default this to 1 so that all households have the propensity to use the tracing app.
*   prob_has_trace_app - if an individual is a member of a household with the propensity to use the tracing app, then this is the probability that they will install it.

### Adherence parameters
We allow for two types of non-adherence - non-uptake of self-isolation and early exit form self-isolation (not completing the full quarantine period)

Non-uptake:
*   Non-uptake is clustered at the household level. The probability that a household will uptake isolation is given by hh_prob_will_take_up_isolation

Early exit:
*   Every day, there is a probability that a an individual will leave isolation. This probability is constant and given by hh_prob_propensity_to_leave_isolation.

### Recall
This is specific to uk_model.

Contacts that occur long ago in the past are difficult to recall. We include a simple model of this, where the probability that a contact is recalled decays at a geometric rate. THis is controlled by parameter recall_probability_fall_off. If this parameter is 1, then there is no recall decay. If the parameter is 0.9, then recall decays at a rate of 10% per day. If a contact occurs $t$ days in the past, then the probability that the contact will be recalled is given by (recall_probability_fall_off)^$t$.
