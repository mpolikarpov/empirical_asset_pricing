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

# only industries that have a minimum of 36 of the past 60 monthly returns available

non_null_cols = df.iloc[:24].notnull().any()
df_filtered = df.loc[:, non_null_cols[non_null_cols].index]

# rolling window regression

dependent_variables = ['Agric', 'Food ', 'Beer ', 'Smoke', 'Toys ', 'Fun  ', 'Books', 'Hshld',
       'Clths', 'MedEq', 'Drugs', 'Chems', 'Txtls', 'BldMt', 'Cnstr', 'Steel',
       'Mach ', 'ElcEq', 'Autos', 'Aero ', 'Ships', 'Mines', 'Coal ', 'Oil  ',
       'Util ', 'Telcm', 'PerSv', 'BusSv', 'Hardw', 'Chips', 'LabEq', 'Boxes',
       'Trans', 'Whlsl', 'Rtail', 'Meals', 'Banks', 'Insur', 'RlEst', 'Fin  ',
       'Other'] # just copy-paste from df_filtered

rw_results = {dep_var: [] for dep_var in dependent_variables}

for dep_var in dependent_variables:
    for i in range(len(df) - 60 + 1):
        window_data = df.iloc[i:i+60]
        X = sm.add_constant(window_data.emkt)
        y = window_data[dep_var]

        model = sm.OLS(y, X)
        result = model.fit()

        rw_results[dep_var].append(result.params['emkt'])

# Fama-Macbeth regression

# Lag to determine all betas at time t-1
beta_df = pd.DataFrame(rw_results)
for column in beta_df.columns:
    beta_df[column] = beta_df[column].shift(1)
beta_df = beta_df[1:]

# Industry portfolios
df_adj = df.tail(len(beta_df)) # Adjust df to be the same length as beta_df
fm_portfolios = df_adj[dependent_variables]

fm_results = []
fm_residuals = []

for portfolio in fm_portfolios:
    ret = df[portfolio] # Excess return for each portfolio
    model = sm.OLS(ret, sm.add_constant(df['emkt']))
    result = model.fit()

    fm_residuals.append(result.resid)

    # Append the results summary to the list
    fm_results.append({
        'Portfolio': portfolio,
        'Alpha': result.params['const'],
        'Beta': result.params['emkt'],
        'Alpha T-stat': result.tvalues['const'],
        'Beta T-stat': result.tvalues['emkt'],
        'R-squared adj': result.rsquared_adj
    })

print(pd.DataFrame(fm_results))
