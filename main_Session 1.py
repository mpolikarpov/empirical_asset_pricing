import pandas as pd
import numpy as np
from scipy.stats import f
from scipy.stats import chi2
import statsmodels.api as sm

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

portfolios = ['Lo 10', 'Dec-02', 'Dec-03', 'Dec-04', 'Dec-05', 'Dec-06', 'Dec-07', 'Dec-08', 'Dec-09', 'Hi 10']

results = []
residuals = []

for portfolio in portfolios:
    excess_return = df[portfolio] - df['RF']  # Calculating excess return for each portfolio
    model = sm.OLS(excess_return, sm.add_constant(df['Mkt-RF']))
    result = model.fit()

    residuals.append(result.resid)

    # Append the results summary to the list
    results.append({
        'Portfolio': portfolio,
        'Alpha': result.params['const'],
        'Beta': result.params['Mkt-RF'],
        'Alpha T-stat': result.tvalues['const'],
        'Beta T-stat': result.tvalues['Mkt-RF'],
        'Alpha White se': result.get_robustcov_results(cov_type='HC0').HC1_se[0],
        'Beta White se': result.get_robustcov_results(cov_type='HC0').HC1_se[1],
        'Alpha White T-stat': result.params['const'] / result.get_robustcov_results(cov_type='HC1').HC1_se[0],
        'Beta White T-stat': result.params['Mkt-RF'] / result.get_robustcov_results(cov_type='HC1').HC1_se[1],
        'R-squared': result.rsquared,
        'P-value': result.pvalues['Mkt-RF']
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
white_test = ['Portfolio','Alpha', 'Beta', 'Alpha White se', 'Beta White se', 'Alpha White T-stat', 'Beta White T-stat']
results_df[white_test].to_csv('white_results.csv', index=True)

# Number of portfolios
num_portfolios = len(results)

# Combine residuals into a matrix
residual_matrix = np.column_stack(residuals)

# Calculate the residual covariance matrix
residual_covariance = np.cov(residual_matrix)

# Calculate the mean excess return across portfolios
mean_excess_returns = np.array([result['Alpha'] for result in results])

# Calculate White's standard errors (heteroscedasticity-consistent standard errors)
white_cov = np.linalg.inv(residual_matrix @ residual_matrix.T) @ (residual_matrix @ residual_matrix.T) @ residual_covariance @ (residual_matrix @ residual_matrix.T) @ np.linalg.inv(residual_matrix @ residual_matrix.T)
white_std_errors = np.sqrt(np.diag(white_cov))

# Check for NaN or infinite values in white_std_errors
if np.any(np.isnan(white_std_errors)) or np.any(np.isinf(white_std_errors)):
    print("NaN or infinite values encountered in White's standard errors calculation. Check your data or approach.")

# Calculate Newey-West standard errors with lag length determined by Barlett kernel
T = len(residuals[0])  # Number of observations
lag = int(np.ceil(4 * (T / 100) ** (2 / 9)))  # Lag length using Barlett kernel
newey_west_cov = sm.stats.sandwich_covariance.cov_hac(result, nlags=lag)
newey_west_std_errors = np.sqrt(np.diag(newey_west_cov))

# Print standard errors for each coefficient
print("Newey-West Standard Errors:", newey_west_std_errors)

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

# Call the GRS test function
grs_statistic, grs_p_value = grs_test(residual_matrix, mean_excess_returns.reshape(-1, 1), df[['Mkt-RF']].values)

print("GRS Test Statistic:", grs_statistic)
print("p-value:", grs_p_value)


# Calculate the Wald statistic for the joint test that all alphas are zero
wald_statistic = grs_statistic * (len(residuals[0]) - len(mean_excess_returns) - 1) / len(mean_excess_returns)

# Calculate the p-value using F-distribution
wald_p_value = 1 - f.cdf(wald_statistic, len(mean_excess_returns), len(residuals[0]) - len(mean_excess_returns) - 1)

print("Wald Statistic:", wald_statistic)
print("p-value:", wald_p_value)

# Calculate the Wald statistic using White's standard errors
alpha_vector = mean_excess_returns.reshape(-1, 1)
wald_statistic_white = alpha_vector.T @ np.linalg.inv(white_cov) @ alpha_vector
wald_p_value_white = 1 - chi2.cdf(wald_statistic_white, len(mean_excess_returns))

# Calculate the Wald statistic using Newey-West standard errors with lag
wald_statistic_newey_west_lagged = alpha_vector.T @ np.linalg.inv(newey_west_cov) @ alpha_vector
wald_p_value_newey_west_lagged = 1 - chi2.cdf(wald_statistic_newey_west_lagged, len(mean_excess_returns))

print("Wald Statistic (White's Standard Errors):", wald_statistic_white)
print("p-value (White's Standard Errors):", wald_p_value_white)
print("\nWald Statistic (Newey-West Standard Errors with Lag):", wald_statistic_newey_west_lagged)
print("p-value (Newey-West Standard Errors with Lag):", wald_p_value_newey_west_lagged)


"""
Matlab code

% excess returns on the test assets
eBMport = tempPort(:,2:end)/100-rf;

% check for errors
plot([eMkt rf eBMport])
mean([eMkt rf eBMport])*12
std([eMkt rf eBMport])

% run regressions for Question 2
% number of observations
T = size(eMkt,1);
% matrix of the dependent variables
Y = eBMport;
% independent variable plus a column vector of ones
X = [ones(T,1) eMkt];

% estimate the coefficients
B = X\Y; % = (X'*X)\(X'*Y);
% residual from regression (T x 10 matrix of regression residuals)
e = Y-X*B;
% Covariance matrix of Bhat
% covariance of Bhat first regression
sum(e(:,1).^2)/(T-2)*inv(X'*X)
% t-statistic of the first regression
B(:,1)./sqrt(diag(sum(e(:,1).^2)/(T-2)*inv(X'*X)))
% But we need the covariance matrix of all regression 
% coefficients (e.g., interrelations between regressions)
covB = kron(e'*e/(T-2),inv(X'*X));

% Question 3
xlswrite("Table1.xls",[B(1,:)' B(1,:)'./diag(covB(1:2:end,1:2:end)).^0.5 ...
    B(2,:)' B(2,:)'./(diag(covB(2:2:end,2:2:end)).^0.5) ...
    (1-sum(e.^2)./sum((Y-mean(Y)).^2))'])

% Question 4
% GRS test statistic
GRSstatistic = (T-10-1)/10*(1/T)*(B(1,:)*inv(covB(1:2:end,1:2:end))*B(1,:)');
pvalueGRS = 1-fcdf(GRSstatistic,10,T-10-1);

% GRS pvalue is  0.0039 -> we reject the hypothesis that the market
% portfolio is efficient at the 1% singificance level.

% Question 5
% Wald statistic
Waldstat = B(1,:)*inv(covB(1:2:end,1:2:end))*B(1,:)';
pvalueWaldstat = 1-chi2cdf(Waldstat,10);
% Pvalue of Waldstat is 0.0033 -> we reject the hypothesis that the market
% portfolio is efficient at the 1% singificance level.
"""
