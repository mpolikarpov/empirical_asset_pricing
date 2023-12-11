import pandas as pd
import numpy as np
from scipy.stats import f
from scipy.stats import chi2
import statsmodels.api as sm
import scipy.io

# import data
m = pd.read_csv('m.csv')
industries = pd.read_csv('industries.csv')

# define a date variable
m['Date'] = pd.to_datetime(m['Date'], format='%Y%m')
industries['Date'] = pd.to_datetime(industries['Date'], format='%Y%m')

# prepare m for merge
m['emkt'] = m['Mkt-RF']/100
m['rf'] = m['RF']/100
columns_to_drop = ['Mkt-RF', 'SMB', 'HML', 'RF']
m = m.drop(columns_to_drop, axis=1)

# merge
df = pd.merge(m, industries, on='Date')

# Define a list of dependent variables (industry portfolios)
dependent_variables = industries.columns[1:]  # Assuming first column is 'Date'

# Rolling window regression for each dependent variable
rolling_window = 60

rw_results = {dep_var: [] for dep_var in dependent_variables}

for dep_var in dependent_variables:
    for i in range(len(df) - rolling_window + 1):
        window_data = df.iloc[i:i+rolling_window]
        non_null_count = window_data[dep_var].notnull().sum()

        # Check if minimum 36 of the past 60 monthly returns are available
        if non_null_count >= 36:
            X = sm.add_constant(window_data['emkt'])
            y = window_data[dep_var]

            model = sm.OLS(y, X)
            result = model.fit()

            rw_results[dep_var].append(result.params['emkt'])
        else:
            rw_results[dep_var].append(np.nan)  # Insert NaN if minimum data is not available

# Construct a DataFrame from the rolling window results
beta_df = pd.DataFrame(rw_results)


# Fama-Macbeth regression
# Lag to determine all betas at time t-1
for column in beta_df.columns:
    beta_df[column] = beta_df[column].shift(1)
beta_df = beta_df[1:]  # Remove the first row with NaNs

# Industry portfolios adjusted to match the length of beta_df
df_adj = df.tail(len(beta_df))

fm_results = []

for portfolio in dependent_variables:
    ret = df_adj[portfolio]  # Excess return for each portfolio
    model = sm.OLS(ret, sm.add_constant(df_adj['emkt']))
    result = model.fit()

    fm_results.append({
        'Portfolio': portfolio,
        'Alpha': result.params['const'],
        'Beta': result.params['emkt'],
        'Alpha T-stat': result.tvalues['const'],
        'Beta T-stat': result.tvalues['emkt'],
        'R-squared adj': result.rsquared_adj
    })

print(pd.DataFrame(fm_results))
