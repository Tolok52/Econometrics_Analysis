import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:\\Users\\gianc\\OneDrive\\Documents\\GrEnFin-DESKTOP-6MSVVS0\\Econometrics of financial markets\\Melendez_Giancarlo.csv'
data = pd.read_csv(file_path)

# Define a function to perform Augmented Dickey-Fuller test
def adf_test(series, signif=0.05, name='', print_results=True):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    if print_results:
        print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
        print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
        print(f' Significance Level    = {signif}')
        print(f' Test Statistic        = {output["test_statistic"]}')
        print(f' No. Lags Chosen       = {output["n_lags"]}')

        for key, val in r[4].items():
            print(f' Critical value {adjust(key)} = {round(val, 3)}')

        if p_value <= signif:
            print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
            print(f" => Series is Stationary.")
        else:
            print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
            print(f" => Series is Non-Stationary.")
    return p_value

# Function to plot time series data
def plot_series(data, title='', labels=None):
    plt.figure(figsize=(14, 6))
    for series, label in zip(data, labels):
        plt.plot(series, label=label)
    plt.title(title)
    plt.legend()
    plt.show()

# Apply ADF Test to each column (for exercise 1)
data_var = data[['oil', 'inf', 'growth']]
for name, column in data_var.iteritems():
    adf_test(column, name=column.name)
    print("\n")

# Fitting a VAR model and selecting the lag order based on AIC and BIC
model = VAR(data_var)
results = model.fit(maxlags=15, ic='aic')
lag_order_results = results.model.select_order(15)
print(lag_order_results.summary())

# Displaying the summary of the model
model_summary = results.summary()
print(model_summary)

# Estimating the VAR model with the selected lag order of 1
var_model = VAR(data[['oil', 'inf', 'growth']])
var_results = var_model.fit(1)

# Displaying the results
var_results_summary = var_results.summary()
var_results_summary

# Check for serial correlation in VAR residuals using the Durbin-Watson statistic
dw_results = durbin_watson(var_results.resid)
dw_results_dict = dict(zip(['oil', 'inf', 'growth'], dw_results))

# Report the Durbin-Watson statistics
print('Durbin-Watson Statistics:', dw_results_dict)

# Granger Causality Tests
print('Granger Causality Oil causes Inflation:', grangercausalitytests(data_var[['inf', 'oil']], maxlag=lag_order_results.aic))
print('Granger Causality Oil causes Growth:', grangercausalitytests(data_var[['growth', 'oil']], maxlag=lag_order_results.aic))


# Estimating an ARIMA model for oil prices (1,1)
arma_model_oil = ARIMA(data['oil'], order=(1, 0, 1))
arma_results_oil = arma_model_oil.fit()

# Estimating an ARIMA model for oil prices (1,0)
arma_model_oil_2 = ARIMA(data['oil'], order=(1, 0, 0))
arma_results_oil_2 = arma_model_oil_2.fit()

# Displaying the summary of the ARMA model
arma_results_summary = arma_results_oil.summary()
print(arma_results_summary)

arma_results_summary_2 = arma_results_oil_2.summary()
print(arma_results_summary_2)

# Exercise 2: PPP and Cointegration Analysis

# Perform the ADF test on the variables
adf_results = {
    'pa': adf_test(data['pa'], name='pa'),
    'pb': adf_test(data['pb'], name='pb'),
    'e': adf_test(data['e'], name='e'),
}

# Differencing the data
differenced_data = data[['pa', 'pb', 'e']].diff().dropna()

# Perform the ADF test on the differenced variables
adf_results_differenced = {
    'diff_pa': adf_test(differenced_data['pa'], name='diff pa'),
    'diff_pb': adf_test(differenced_data['pb'], name='diff pb'),
    'diff_e': adf_test(differenced_data['e'], name='diff e'),
}

# Perform the Johansen cointegration test
johansen_test = coint_johansen(data[['pa', 'e', 'pb']], det_order=0, k_ar_diff=1)

# Fit the VECM model with 'co' deterministic term
vecm_model_co = VECM(data[['pa', 'e', 'pb']], k_ar_diff=1, coint_rank=1, deterministic='co')
vecm_result_co = vecm_model_co.fit()

# Fit the VECM model with 'ci' deterministic term
vecm_model_ci = VECM(data[['pa', 'e', 'pb']], k_ar_diff=1, coint_rank=1, deterministic='ci')
vecm_result_ci = vecm_model_ci.fit()

#Display the results

print({
    'Trace Statistic': johansen_test.lr1,
    'Critical Values (Trace Statistic)': johansen_test.cvt[:, 1],
    'Eigen Statistic': johansen_test.lr2,
    'Critical Values (Eigen Statistic)': johansen_test.cvm[:, 1],
    'Eigenvalues': johansen_test.eig
})
# Print summaries of the VECM models
print("VECM with co deterministic term summary:")
print(vecm_result_co.summary())
print("\nVECM with ci deterministic term summary:")
print(vecm_result_ci.summary())

# Diagnostic Checks
# Check for serial correlation in VECM residuals using Durbin-Watson statistic
def check_serial_correlation(model_results):
    out = durbin_watson(model_results.resid)
    for col, val in zip(data.columns, out):
        print((f'durbin_watson statistic for {col} = {val:.2f}'))

# Apply the function to the VECM result
print("Checking for serial correlation in VECM residuals:")
check_serial_correlation(vecm_result_co)

# Impulse Response Analysis
# Generate impulse response function (IRF) analysis to track the impact of a one standard deviation shock to one of the variables on other variables
irf = vecm_result_co.irf(10) # 10 periods ahead
irf.plot(orth=True) # orthogonalized IRF
plt.show()

# Report the IRF analysis results (you may need to capture and process the IRF results as per your requirements)

# Additional Model Diagnostics
# Here you could add further checks for things like model stability (eigenvalue check), residual normality, etc.

# ... [Any additional analysis or summaries] ...

# Remember to save any figures or tables that you wish to include in your appendix
# For example, to save an IRF plot:
irf_fig = irf.plot(orth=True)
irf_fig.savefig('impulse_response_function.png')


##PLOTS

#Plot oil, inf and growth
plot_series([data['oil'], data['inf'], data['growth']], title='Time Series', labels=['Oil', 'Inflation', 'GDP Growth'])

# Plotting the impulse response functions
irf = var_results.irf(10)
irf.plot(orth=True)
plt.show()

# Plot the original and differentiated 'pa', 'e', and 'pb' series

plot_series([data['pa'], data['e'], data['pb']], title='Original Series', labels=['PA', 'E', 'PB'])
plot_series([differenced_data['pa'], differenced_data['e'], differenced_data['pb']], title='Differenced Series', labels=['Differenced PA', 'Differenced E', 'Differenced PB'])
# Plotting the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for each variable
fig, axes = plt.subplots(nrows=3, ncols=2, dpi=120, figsize=(10,6))

# Plotting for oil
plot_acf(data.oil, ax=axes[0, 0], lags=20, title="Oil ACF")
plot_pacf(data.oil, ax=axes[0, 1], lags=20, title="Oil PACF")

# Plotting for inf
plot_acf(data.inf, ax=axes[1, 0], lags=20, title="Inflation ACF")
plot_pacf(data.inf, ax=axes[1, 1], lags=20, title="Inflation PACF")

# Plotting for growth
plot_acf(data.growth, ax=axes[2, 0], lags=20, title="GDP Growth ACF")
plot_pacf(data.growth, ax=axes[2, 1], lags=20, title="GDP Growth PACF")

plt.tight_layout()
plt.show()
