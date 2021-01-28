from patsy import dmatrices
import pandas as pd
import statsmodels.api as sma
import numpy as np

def estimate_growth_rates(data_in, days_to_use = 20):
    """
    Estimates the growth rate of simulated epidemics using a robust linear regression model.

    Arguments:
        data_in {pandas.DataFrame} returned by one of our simulation scripts
        being the total number of infections on day t

    Keyword Arguments:
        days_to_use {int} -- The data cut off, dates after this point in
        are not used to estimate the growth rate (default: {20})

    Returns:
        [pandas.Series] -- data_in with an appended estimated growth rate
    """
    
    infection_counts = [str(i) for i in range(days_to_use)]
    
    data_subset = data_in[infection_counts]
    
    growth_rates = []
    
    for _ in range(data_in.shape[0]):
        
        # Print the current progress
        print(f"current fit; {_}", end = "\r")

        # Difference the data and log it
        log_diff = data_subset.iloc[_].diff(1).apply(lambda x: np.log(x))

        # Set up a dictionary to create the dataframe
        data_set_up = {
            "log_case_incidence": log_diff[10:days_to_use],
            "time": list(range(10, days_to_use))
        }
        dataframe = pd.DataFrame(data_set_up)

        # Linear model fitting
        y, X = dmatrices('log_case_incidence ~ time', 
                         data = dataframe)
        model = sma.RLM(y, X)
        res = model.fit()
        growth_rates.append(res.params[1])

        #fig = sm.graphics.regressionplots.plot_fit(res, 1)

        if res.params[1] == -float("inf"):
            print(f"missing data in line {_}")

    growth_rate_series = pd.Series(growth_rates)
    data_in["growth_rate"] = growth_rate_series
    
    return data_in
