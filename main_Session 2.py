import pandas as pd
import numpy as np
from scipy.stats import f
from scipy.stats import chi2
import statsmodels.api as sm

# import data

m = pd.read_excel('m.xlsx')
ewm = pd.read_excel('ewm.xlsx')
bm = pd.read_excel('bm.xlsx')
size = pd.read_excel('size.xlsx')

# define a date variable

m['Date'] = pd.to_datetime(m['Date'], format='%Y%m')
ewm['Date'] = pd.to_datetime(ewm['Date'], format='%Y%m')

# For at home
size['Date'] = pd.to_datetime(size['Date'], format='%Y%m')
bm['Date'] = pd.to_datetime(bm['Date'], format='%Y')

df = pd.merge(ewm, m, on='Date')

# excess return on the benchmark asset and rf

df['eMkt'] = df['Mkt-RF']/100
df['rf'] = df['RF']/100

# Filter data starting from June 1931
start_date = '1931-06-01'
filtered_df = df[df['Date'] >= start_date]

portfolios = list(df.columns[1:-6])  # Assuming industry columns start from index 1

results = []

for portfolio in portfolios:
    # Select relevant columns for the specific portfolio and excess market return
    portfolio_data = filtered_df[['Date', 'eMkt', portfolio]]
    portfolio_data.set_index('Date', inplace=True)

    rolling_window = 60
    rolling_betas = []

    for i in range(len(portfolio_data) - rolling_window + 1):
        window = portfolio_data.iloc[i: i + rolling_window]

        # Filter to include only periods with 36 or more observations in the last 60 months
        valid_window = window.dropna(thresh=36)

        if len(valid_window) >= 36:
            y = valid_window[portfolio]
            excess_return = df[portfolio] - df['RF']  # Calculating excess return for each portfolio
            model = sm.OLS(excess_return, sm.add_constant(valid_window['Mkt-RF']))
            result = model.fit()
            rolling_betas.append(result.params[1])  # Extract the beta coefficient

    if len(rolling_betas) > 0:  # Include results only if data meets the threshold
        avg_rolling_beta = np.mean(rolling_betas)

        # Append the results summary to the list
        results.append({
            'Portfolio': portfolio,
            'Rolling Beta': avg_rolling_beta,
            'Number of Months': len(rolling_betas)
        })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
print(results_df)

