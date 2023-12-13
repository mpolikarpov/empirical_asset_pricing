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
# Drop Industries with NaN values in beta_df
beta_df.dropna(axis=1, inplace=True)

# Fama-MacBeth regression
fm_monthly_results = []  # Store monthly regression results

# Tail the original dataframe to match the length of beta_df
df_adj = df.tail(len(beta_df))

dependent_variables = beta_df.columns

residuals = []

# Loop through each month starting from July 1931
for t in range(rolling_window, len(beta_df)):
    monthly_beta = beta_df.iloc[t - 1]  # Beta values for month t-1
    month_data = df_adj.iloc[t]  # Data for month t

    # Prepare data for the cross-sectional regression
    monthly_returns = month_data[dependent_variables]  # Excess returns for each portfolio
    X = sm.add_constant(monthly_beta)  # Using previous month's betas
    y = monthly_returns

    # Convert columns to numeric in y and X
    y = y.apply(pd.to_numeric, errors='coerce')
    X = X.apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values in y and X
    y = y.dropna()
    X = X.dropna()
    model = sm.OLS(y, X, missing='drop')  # Drop missing values
    result = model.fit()

    residuals.append(result.resid)

    # Calculate Newey-West standard errors and T-stats
    T = len(residuals[0])  # Number of observations
    lag_value = int(np.ceil(4 * (T / 100) ** (2 / 9)))  # Lag length using Barlett kernel
    result_NW = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag_value})  # Adjust lag_value as needed
    robust_results = result_NW.get_robustcov_results(cov_type='HAC', maxlags=lag_value)

    fm_monthly_results.append({
        'Month': month_data['Date'],
        'Alpha': result.params['const'],
        'Beta': result.params.iloc[1],  # Coefficient for beta
        'Alpha T-stat': result.tvalues['const'],
        'Beta T-stat': result.tvalues.iloc[1],  # T-stat for beta
        'R-squared adj': result.rsquared_adj,
        'Observations': result.nobs,
        'Alpha NW se': robust_results.bse[0],
        'Alpha NW T-stat': result.params['const'] / robust_results.bse[0],
        'Beta NW se': robust_results.bse[1],
        'Beta NW T-stat': result.params.iloc[1] / robust_results.bse[1]

    })

# Calculate average Alpha and Beta
avg_alpha = np.mean([res['Alpha'] for res in fm_monthly_results])
avg_beta = np.mean([res['Beta'] for res in fm_monthly_results])

# Calculate average R-squared and average observations per cross-section
avg_rsquared = np.mean([res['R-squared adj'] for res in fm_monthly_results])
avg_observations = np.mean([res['Observations'] for res in fm_monthly_results])

# Display Fama-MacBeth results
fm_results_df = pd.DataFrame(fm_monthly_results)
print(fm_results_df)

# Display coefficients
print(f"Average Alpha: {avg_alpha}")
print(f"Average Beta: {avg_beta}")

# Display NW T-stats
print(f"Alpha NW T-stat: {np.mean([res['Alpha NW T-stat'] for res in fm_monthly_results])}")
print(f"Beta NW T-stat: {np.mean([res['Beta NW T-stat'] for res in fm_monthly_results])}")

print(f"Average Adjusted R-squared: {avg_rsquared}")
print(f"Average Observations per Cross-section: {avg_observations}")
