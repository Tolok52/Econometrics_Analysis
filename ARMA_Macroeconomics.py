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



##PLOTS

#Plot oil, inf and growth
plot_series([data['oil'], data['inf'], data['growth']], title='Time Series', labels=['Oil', 'Inflation', 'GDP Growth'])

# Plotting the impulse response functions
irf = var_results.irf(10)
irf.plot(orth=True)
plt.show()

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
