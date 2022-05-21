#!/usr/bin/env python
# coding: utf-8

# In[4]:


import warnings                                 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xlrd

from matplotlib.pylab import style

from itertools import product
from tqdm import tqdm_notebook
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

from sklearn.metrics import mean_absolute_error


# In[5]:


style.use('ggplot')

filename = 'procedure_imported_data/data (2005-2015).xls'
disease = pd.read_excel(filename, index_col='date')

full_value_name = 'procedure_imported_data/data (2005-2017).xls'
observed_value = pd.read_excel(full_value_name, index_col='date')

# draw the time sequence diagram.
disease.plot(label='Observed value')
plt.xlabel('Date')
plt.ylabel('Rate')
plt.legend(loc='upper left')

# stability test (p<0.05 indicates a smooth series)
print("The ADF test results for the original series are " + str(ADF(disease['Observed value'])))


# In[3]:


# seasonal difference
ds_disease = disease['Observed value'] - disease['Observed value'].shift(12)
ds_disease = ds_disease[12:]

plt.plot(ds_disease)
plt.xlabel('Date')
plt.ylabel('Rate')
plt.legend(loc='upper left')
plt.savefig('seasonal difference (2005-2015).pdf', dpi=300)

# stability test (p<0.05 indicates a smooth series)
print("The ADF test results for the original series are: " + str(ADF(ds_disease)))

# white noise test (p<0.05 indicates non-white noise)
print("White noise test results for first-order difference series: " + str(acorr_ljungbox(ds_disease, lags=1)))


# In[4]:


# make a first-order difference and perform a smoothness test
d1_disease = ds_disease.diff(periods=1).dropna()
plt.plot(d1_disease)
plt.xlabel('Date')
plt.ylabel('Rate')
plt.legend(loc='upper left')
plt.savefig('first-order seasonal and first-order differencing.pdf', dpi=300)

print("The ADF test for the series after first-order differencing is " + str(ADF(d1_disease)))


# In[19]:


# create a possible sequence of parameters
ps = range(0, 3)
d=1
qs = range(0, 3)
Ps = range(0, 3)
D=1 
Qs = range(0, 3)
s = 12

# create a list of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)


# In[20]:


def optimizeSARIMA_aic(parameters_list, d, D, s):
    
    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model = sm.tsa.statespace.SARIMAX(disease, order=(param[0], d, param[1]),
                                              seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC, and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table_aic = pd.DataFrame(results)
    result_table_aic.columns = ['parameters', 'aic']

    # sorting in ascending order, the lower AIC is - the better
    result_table_aic = result_table_aic.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table_aic

def optimizeSARIMA_bic(parameters_list, d, D, s):
    """
    Return dataframe with parameters and corresponding BIC
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    
    results = []
    best_bic = float("inf")

    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model = sm.tsa.statespace.SARIMAX(disease, order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        bic = model.bic
        # saving best model, AIC, and parameters
        if bic < best_bic:
            best_model = model
            best_bic = bic
            best_param = param
        results.append([param, model.bic])

    result_table_bic = pd.DataFrame(results)
    result_table_bic.columns = ['parameters', 'bic']

    # sorting in ascending order, the lower BIC is - the better
    result_table_bic = result_table_bic.sort_values(by='bic', ascending=True).reset_index(drop=True)
    
    return result_table_bic


# In[21]:


warnings.filterwarnings("ignore") 
result_table_aic = optimizeSARIMA_aic(parameters_list, d, D, s)
result_table_bic = optimizeSARIMA_bic(parameters_list, d, D, s)
print(result_table_aic.head(10))
print(result_table_bic.head(10))


# In[11]:


model = sm.tsa.statespace.SARIMAX(disease['Observed value'],
                                  order=(1, 1, 1),
                                  seasonal_order=(0, 1, 1, 12))
results = model.fit(disp=-1)
print(results.summary())

sm.graphics.tsa.plot_acf(results.resid, lags=36)
plt.ylabel('Correlation Coefficients')
plt.xlabel('Lags')
plt.tight_layout()
plt.savefig('SARIMA-acf.pdf', dpi=300)

sm.graphics.tsa.plot_pacf(results.resid, lags=36)
plt.ylabel('Correlation Coefficients')
plt.xlabel('Lags')
plt.tight_layout()
plt.savefig('SARIMA-pacf.pdf', dpi=300)


# In[6]:


# residual tests (QQ-plot and D-W test)
qqplot(results.resid, line='q', fit=True).show()
print("D-W test results: ", sm.stats.durbin_watson(results.resid.values))
print("White noise test results for the residual series: " + str(acorr_ljungbox(results.resid, lags=1)))


# In[7]:


# fitting data
pred = results.get_prediction(start=pd.to_datetime('2005-02-01'), dynamic=False, full_results=True)
pred_ci = pred.conf_int()
prediction_value = pred.predicted_mean
print(prediction_value)


# In[8]:


# predicting the future data
forecast = results.get_forecast(steps=24)

# get the prediction confidence interval
forecast_ci = forecast.conf_int()
forecast_value = forecast.predicted_mean
pred_concat = pd.concat([forecast_value,forecast_ci],axis=1)
pred_concat.columns = [u'Predicted value',u'LCL',u'UCL']
pred_concat.head()


# In[9]:


# time series chart
ax = observed_value.plot(label='Observed value')
forecast.predicted_mean.plot(ax=ax, label='SARIMA predicted value')
prediction_value.plot(ax=ax, label='Fit value')

ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1], color='k', alpha=.2)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2016-01-01'), observed_value.index[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Date')
ax.set_ylabel('Rate')
plt.legend(loc='upper left')
plt.ylim(0, 16)

plt.savefig('SARIMA prediction.pdf', dpi=600)

plt.show()


# In[10]:


# 24-steps
forecast_value = forecast.predicted_mean
truth = observed_value['2016-01-01':]
forecast_value = pd.DataFrame(forecast_value)
truth = pd.DataFrame(truth)

# calculation errors (RMSE, MAE, MAPE, and sMAPE)
mse = ((forecast_value.iloc[:, 0] - truth.iloc[:, 0]) ** 2).mean()
print('RMSE is {:.4f}'.format(np.sqrt(mse)))
mae = mean_absolute_error(truth.iloc[:, 0], forecast_value.iloc[:, 0])
print('MAE is {:.4f}'.format(mae))
mape = np.mean(np.abs((truth.iloc[:, 0] - forecast_value.iloc[:, 0]) / truth.iloc[:, 0])) * 100
print('MAPE is {:.4f}'.format(mape))
smape = 2.0 * np.mean(np.abs(truth.iloc[:, 0] - forecast_value.iloc[:, 0]) / (np.abs(truth.iloc[:, 0]) + np.abs(forecast_value.iloc[:, 0]))) * 100
print('SMAPE is {:.4f}'.format(smape))


# In[11]:


# 12-steps
forecast_value = forecast.predicted_mean['2016-01-01':'2016-12-01']
truth = observed_value['2016-01-01':'2016-12-01']
forecast_value = pd.DataFrame(forecast_value)
truth = pd.DataFrame(truth)

# calculation errors (RMSE, MAE, MAPE, and sMAPE)
mse = ((forecast_value.iloc[:, 0] - truth.iloc[:, 0]) ** 2).mean()
print('RMSE is {:.4f}'.format(np.sqrt(mse)))
mae = mean_absolute_error(truth.iloc[:, 0], forecast_value.iloc[:, 0])
print('MAE is {:.4f}'.format(mae))
mape = np.mean(np.abs((truth.iloc[:, 0] - forecast_value.iloc[:, 0]) / truth.iloc[:, 0])) * 100
print('MAPE is {:.4f}'.format(mape))
smape = 2.0 * np.mean(np.abs(truth.iloc[:, 0] - forecast_value.iloc[:, 0]) / (np.abs(truth.iloc[:, 0]) + np.abs(forecast_value.iloc[:, 0]))) * 100
print('SMAPE is {:.4f}'.format(smape))


# In[12]:


# 6-steps
forecast_value = forecast.predicted_mean['2016-01-01':'2016-06-01']
truth = observed_value['2016-01-01':'2016-06-01']
forecast_value = pd.DataFrame(forecast_value)
truth = pd.DataFrame(truth)

# calculation errors (RMSE, MAE, MAPE, and sMAPE)
mse = ((forecast_value.iloc[:, 0] - truth.iloc[:, 0]) ** 2).mean()
print('RMSE is {:.4f}'.format(np.sqrt(mse)))
mae = mean_absolute_error(truth.iloc[:, 0], forecast_value.iloc[:, 0])
print('MAE is {:.4f}'.format(mae))
mape = np.mean(np.abs((truth.iloc[:, 0] - forecast_value.iloc[:, 0]) / truth.iloc[:, 0])) * 100
print('MAPE is {:.4f}'.format(mape))
smape = 2.0 * np.mean(np.abs(truth.iloc[:, 0] - forecast_value.iloc[:, 0]) / (np.abs(truth.iloc[:, 0]) + np.abs(forecast_value.iloc[:, 0]))) * 100
print('SMAPE is {:.4f}'.format(smape))

