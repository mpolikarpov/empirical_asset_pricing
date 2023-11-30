import pandas as pd

# import data

m = pd.read_csv('/Users/mikhail/PycharmProjects/Empirical_Asset_Pricing/Data/m.csv')
vwm = pd.read_csv('/Users/mikhail/PycharmProjects/Empirical_Asset_Pricing/Data/vwm.csv')

# define a date variable

m['Date'] = pd.to_datetime(m['Date'], format='%Y%m')
vwm['Date'] = pd.to_datetime(vwm['Date'], format='%Y%m')

df = pd.merge(vwm, m, on='Date')

# excess return on the benchmark asset and rf

df['eMkt'] = df['Mkt-RF']/100
df['rf'] = df['RF']/100

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
