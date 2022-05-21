#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
from math import sqrt
from numpy import concatenate

from matplotlib import pyplot as plt
from matplotlib.pylab import style

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM


# In[2]:


style.use('ggplot')

# import date and time information and parse it into Pandas DataFrame index
def parse(x):
    return datetime.strptime(x, '%Y %m')

filename = 'procedure_imported_data/TB_LASSO_index17.csv'

dataset = pd.read_csv(filename, engine='python', parse_dates=[['year', 'month']], 
                      index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)

# manually specify column names
dataset.columns = ['rate', 'avr_air_pressure', 'avr_wind_spd', 'rel_humidity', 
                   'month_sun', 'hour_sun', 'extreme_wind_speed', 'min_air_pressure',
                   'max_temp', 'day_max_precipitation', 'power', 'industrial_add', 
                   'sales_rate','fin_budget', 'cpi', 'pas_turn', 'im_ex', 
                   'ex_factory_index']
dataset.index.name = 'date'

print(dataset.head(5))

# save as processed data file
completed_filename = 'completed_data/completed_TB_LASSO_index17.csv'
dataset.to_csv(completed_filename)


# In[3]:


# data settings
dataset = pd.read_csv(completed_filename, engine='python', header=0, index_col=0)
values = dataset.values

groups = [0, 1, 2, 3, 4, 5, 6, 7]
i = 1

plt.figure(dpi=300)
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group], color='tab:blue')
    plt.title(dataset.columns[group], y=0.3, loc='right')
    i += 1

plt.savefig('the relationship between incidence and various factors.svg', dpi=600)
plt.show()


# In[4]:


# convert data sets
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # input sequences (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # predict sequences (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            
    # information summary
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # delete rows with 'NAN'
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# data setup
new_filename = 'completed_data/completed_TB_LASSO_index17.csv'
values = dataset.values
values = values.astype('float32')

# data normalization process (Min-Max method)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# specify the number of steps and features
n_hours = 2
n_features = 18

reframed = series_to_supervised(scaled, n_hours, 1)
print(reframed.shape)
print(reframed.head())


# In[5]:


# split into training and testing sets
values = reframed.values
n_train_hours = 11 * 12 - 2
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and output
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)

# reshape input to 3D [sample, steps, feature]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[6]:


# design neural network (add network structure layer by layer)
model = Sequential()
model.add(LSTM(8, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_X, train_y, epochs=200, batch_size=26, 
                    validation_data=(test_X, test_y), verbose=2, shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[7]:


# predict
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))

# predicted back-scaling
inv_yhat = concatenate((yhat, test_X[:, -17:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# observed back-scaling
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -17:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# comparison of observed and predicted values
plt.plot(inv_yhat, linestyle='-', label='LSTM predicted value', marker='o')
plt.plot(inv_y, linestyle='--', label='Observed value', marker='o')
plt.legend()
plt.xlabel('Time series (2016-2017)')
plt.ylabel('Rate')
plt.savefig('Multivariate 2-step LSTM prediction.pdf', dpi=600)
plt.show()


# In[8]:


# 24-steps
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
mae = mean_absolute_error(inv_y, inv_yhat)
mape = np.mean(np.abs((inv_y - inv_yhat) / inv_y)) * 100
smape = 2.0 * 100 * np.mean(np.abs(inv_y - inv_yhat) / (np.abs(inv_y) + np.abs(inv_yhat)))
print('Test RMSE: %.4f' % rmse)
print('Test MAE: %.4f' % mae)
print('Test MAPE: %.4f' % mape)
print('Test SMAPE: %.4f' % smape)


# In[9]:


# 12-steps
rmse = sqrt(mean_squared_error(inv_y[:12], inv_yhat[:12]))
mae = mean_absolute_error(inv_y[:12], inv_yhat[:12])
mape = np.mean(np.abs((inv_y[:12] - inv_yhat[:12]) / inv_y[:12])) * 100
smape = 2.0 * 100 * np.mean(np.abs(inv_y[:12] - inv_yhat[:12]) / (np.abs(inv_y[:12]) + np.abs(inv_yhat[:12])))
print('Test RMSE: %.4f' % rmse)
print('Test MAE: %.4f' % mae)
print('Test MAPE: %.4f' % mape)
print('Test SMAPE: %.4f' % smape)


# In[10]:


# 6-steps
rmse = sqrt(mean_squared_error(inv_y[:6], inv_yhat[:6]))
mae = mean_absolute_error(inv_y[:6], inv_yhat[:6])
mape = np.mean(np.abs((inv_y[:6] - inv_yhat[:6]) / inv_y[:6])) * 100
smape = 2.0 * 100 * np.mean(np.abs(inv_y[:6] - inv_yhat[:6]) / (np.abs(inv_y[:6]) + np.abs(inv_yhat[:6])))
print('Test RMSE: %.4f' % rmse)
print('Test MAE: %.4f' % mae)
print('Test MAPE: %.4f' % mape)
print('Test SMAPE: %.4f' % smape)

