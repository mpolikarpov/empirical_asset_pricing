import pandas as pd
import numpy as np
from scipy.stats import f
from scipy.stats import chi2
import statsmodels.api as sm

# Function definition
def grs_test(resid: np.ndarray, alpha: np.ndarray, factors: np.ndarray) -> tuple:
    """ Perform the Gibbons, Ross and Shaken (1989) test.
        :param resid: Matrix of residuals from the OLS of size TxK.
        :param alpha: Vector of alphas from the OLS of size Kx1.
        :param factors: Matrix of factor returns of size TxJ.
        :return Test statistic and pValue of the test statistic.
    """
    # Determine the time series and assets
    iT, iK = resid.shape

    # Determine the amount of risk factors
    iJ = factors.shape[1]

    # Input size checks
    assert alpha.shape == (iK, 1)
    assert factors.shape == (iT, iJ)

    # Covariance of the residuals, variables are in columns.
    mCov = np.cov(resid, rowvar=False)

    # Mean of excess returns of the risk factors
    vMuRF = np.nanmean(factors, axis=0)

    try:
        assert vMuRF.shape == (1, iJ)
    except AssertionError:
        vMuRF = vMuRF.reshape(1, iJ)

    # Duplicate this series for T timestamps
    mMuRF = np.repeat(vMuRF, iT, axis=0)

    # Test statistic
    mCovRF = (factors - mMuRF).T @ (factors - mMuRF) / (iT - 1)
    dTestStat = (iT / iK) * ((iT - iK - iJ) / (iT - iJ - 1)) * \
                (alpha.T @ (np.linalg.inv(mCov) @ alpha)) / \
                (1 + (vMuRF @ (np.linalg.inv(mCovRF) @ vMuRF.T)))

    pVal = 1 - f.cdf(dTestStat, iK, iT-iK-1)

    return dTestStat, pVal

### ALL SAMPLE

# import data
m = pd.read_csv('m.csv')
vwm = pd.read_csv('vwm.csv')

# define a date variable
m['Date'] = pd.to_datetime(m['Date'], format='%Y%m')
vwm['Date'] = pd.to_datetime(vwm['Date'], format='%Y%m')
df = pd.merge(vwm, m, on='Date')

# excess return on the benchmark asset and rf
df['eMkt'] = df['Mkt-RF']/100
df['rf'] = df['RF']/100

# Slice the df
df_pre = df[df['Date'] <= '1963-06-30'] # Filter rows before June 1963 (including June 1963)
df_post = df[df['Date'] > '1963-06-30'] # Filter rows before June 1963 (including June 1963)

list_of_dfs = [df, df_pre, df_post]
file_names_reg = ['results_regressions_sample.xlsx', 'results_regressions_pre.xlsx', 'results_regressions_post.xlsx']  # Define file names for each regression table in list_DataFrame
file_names_stats = ["stats_sample.xlsx", "stats_pre.xlsx", "stats_post.xlsx"] # Define file names for each stats table in list_DataFrame
portfolios = ['Lo 10', 'Dec-02', 'Dec-03', 'Dec-04', 'Dec-05', 'Dec-06', 'Dec-07', 'Dec-08', 'Dec-09', 'Hi 10']

# Loop to iterate 3 times (one time for each df: sample, pre June 1963, post June 1963
for i, df in enumerate(list_of_dfs):
    results = []
    residuals = []

    # Compute the regression for each portfolio and SE, T-Stats for the estimated coefficients
    for portfolio in portfolios:
        excess_return = df[portfolio] - df['RF']  # Calculating excess return for each portfolio
        model = sm.OLS(excess_return, sm.add_constant(df['Mkt-RF']))
        result = model.fit()

        residuals.append(result.resid)

        # Calculate Newey-West standard errors and T-stats
        T = len(residuals[0])  # Number of observations
        lag_value = int(np.ceil(4 * (T / 100) ** (2 / 9)))  # Lag length using Barlett kernel
        result_NW = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag_value})  # Adjust lag_value as needed
        robust_results = result_NW.get_robustcov_results(cov_type='HAC', maxlags=lag_value)

        # Append the results summary to the list
        results.append({
            'Portfolio': portfolio,

            'Alpha': result.params['const'],
            'Alpha T-stat': result.tvalues['const'],
            'Alpha White se': result.get_robustcov_results(cov_type='HC0').HC1_se[0],
            'Alpha White T-stat': result.params['const'] / result.get_robustcov_results(cov_type='HC1').HC1_se[0],
            'Alpha NW se': robust_results.bse[0],
            'Alpha NW T-stat': result.params['const'] / robust_results.bse[0],
            'Alpha P-value': result.pvalues['const'],

            'Beta': result.params['Mkt-RF'],
            'Beta T-stat': result.tvalues['Mkt-RF'],
            'Beta White se': result.get_robustcov_results(cov_type='HC0').HC1_se[1],
            'Beta White T-stat': result.params['Mkt-RF'] / result.get_robustcov_results(cov_type='HC1').HC1_se[1],
            'Beta NW se': robust_results.bse[1],
            'Beta NW T-stat': result.params['Mkt-RF'] / robust_results.bse[1],
            'Beta P-value': result.pvalues['Mkt-RF'],

            'R-squared': result.rsquared

        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    print(results_df.T)
    file_name = file_names_reg[i]
    results_df.T.to_excel(file_name)

    # Using the GRS function
    residual_matrix = np.column_stack(residuals) # Combine residuals into a matrix
    residual_covariance = np.cov(residual_matrix) # Calculate the residual covariance matrix
    mean_excess_returns = np.array([result['Alpha'] for result in results]) # Calculate the mean excess return across portfolios
    grs_statistic, grs_p_value = grs_test(residual_matrix, mean_excess_returns.reshape(-1, 1), df[['Mkt-RF']].values) # Call the GRS test function

    # Calculate the Wald statistic for the joint test that all alphas are zero
    wald_statistic = grs_statistic * (len(residuals[0]) - len(mean_excess_returns) - 1) / len(mean_excess_returns)
    wald_p_value = 1 - f.cdf(wald_statistic, len(mean_excess_returns), len(residuals[0]) - len(mean_excess_returns) - 1) # Calculate the p-value using F-distribution

    # Collect estimated alphas and their standard errors
    alphas = results_df.iloc[:,1]
    alphas_White_se = results_df.iloc[:,3]
    alphas_NW_se = results_df.iloc[:,5]

    # Construct covariance matrix based on the standard errors of alphas
    cov_matrix_alphas_White = np.diag(alphas_White_se)
    cov_matrix_alphas_NW = np.diag(alphas_NW_se)

    # Compute the Wald statistic
    wald_statistic_White = (alphas @ np.linalg.inv(cov_matrix_alphas_White) @ alphas)
    wald_statistic_NW = (alphas @ np.linalg.inv(cov_matrix_alphas_NW) @ alphas)
    wald_df = len(alphas)  # Degrees of freedom for the chi-square distribution
    wald_p_value_White = 1 - chi2.cdf(wald_statistic_White, wald_df) # Calculate the p-value using the chi-square distribution
    wald_p_value_NW = 1 - chi2.cdf(wald_statistic_NW, wald_df) # Calculate the p-value using the chi-square distribution

    stats = []
    stats.append({"GRS Test Statistic": grs_statistic,
                     "GRS p-value": grs_p_value,
                     "Wald Statistic": wald_statistic,
                     "Wald p-value": wald_p_value,
                     "Wald Statistic White": wald_statistic_White,
                     "p-value White": wald_p_value_White,
                     "Wald Statistic NW": wald_statistic_NW,
                     "p-value NW": wald_p_value_NW
    })

    # Convert the list of dictionaries to a DataFrame
    df_stats = pd.DataFrame(stats)
    print(df_stats.T)
    file_name = file_names_stats[i]
    df_stats.T.to_excel(file_name)

