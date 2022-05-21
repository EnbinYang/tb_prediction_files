#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from math import sqrt
from numpy import concatenate

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

from tensorflow.compat.v1.keras.backend import get_session
tensorflow.compat.v1.disable_v2_behavior()

from __future__ import print_function

import shap
shap.initjs()


# In[2]:


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

# save as processed data file
completed_filename = 'completed_data/completed_TB_LASSO_index17.csv'
dataset.to_csv(completed_filename)


# In[3]:


# data settings
dataset = pd.read_csv(completed_filename, engine='python', header=0, index_col=0)
values = dataset.values

groups = [0, 1, 2, 3, 4, 5, 6, 7]
i = 1

plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(dataset.columns[group], y=0.3, loc='right')
    i += 1
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
dataset = pd.read_csv(new_filename, engine='python', header=0, index_col=0)
values = dataset.values
values = values.astype('float32')

# data normalization process (Min-Max method)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[19, 20, 21, 22, 23, 24, 25,
                                26, 27, 28, 29, 30, 31, 32,
                                33, 34, 35]], axis=1, inplace=True)

print(reframed.head())


# In[5]:


# split into training and testing sets
values = reframed.values
n_train_hours = 11 * 12 - 1
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and output
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to 3D [sample, steps, feature]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[6]:


# design neural network (add network structure layer by layer)
model = Sequential()
model.add(LSTM(8, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

model.fit(train_X, train_y, epochs=150, batch_size=22, validation_data=(test_X, test_y))


# In[7]:


import shap

# we use the first 132 training examples as our background dataset to integrate over
explainer = shap.DeepExplainer(model, train_X[:132])

# explain the first 24 predictions
# explaining each prediction requires 2 * background dataset size runs
shap_values = explainer.shap_values(test_X[:24])


# In[8]:


dataset.columns = ['rate', 'avr_air_pressure', 'avr_wind_spd', 'rel_humidity', 
                   'month_sun', 'hour_sun', 'extreme_wind_speed', 'min_air_pressure',
                   'max_temp', 'day_max_precipitation', 'power', 'industrial_add', 
                   'sales_rate','fin_budget', 'cpi', 'pas_turn', 'im_ex', 
                   'ex_factory_index']
fig1 = shap.force_plot(explainer.expected_value[0], shap_values[0][0], dataset.columns, 
                       matplotlib=True, show=False)

fig1.set_facecolor('white')
fig1.savefig('single sample feature value.pdf', bbox_inches = 'tight', dpi=300)


# In[9]:

# set SHAP value manually
new_shape_values = [[-2.07555909e-02,  3.22832377e-03, -7.01126968e-03,
         -3.28778382e-03,  2.44493019e-02, -3.09205391e-02,
          1.08185103e-02, -7.04229064e-03,  4.86812964e-02,
         -5.11698658e-03, -1.55109155e-04, -7.64225982e-03,
          1.74246449e-03, -2.45297775e-02, -1.45195471e-03,
          5.29442768e-05, -4.70874831e-03,  4.39360924e-03],
                   
        [-6.59693871e-03,  5.19968895e-03, -3.33166984e-03,
          2.58122142e-02, -2.40170844e-02, -8.21811147e-03,
          1.16919205e-02, -1.06574930e-02,  5.94768934e-02,
         -4.27860953e-03, -2.52412912e-03, -4.75753890e-03,
          8.64983420e-04, -2.24942714e-02, -1.97302387e-03,
         -2.61467770e-02,  1.78980641e-03,  3.35772289e-03],
                   
        [-6.18082983e-03,  2.19015777e-03, -3.23731406e-03,
          2.56482549e-02, -1.80464834e-02, -5.58605557e-03,
         -9.41686856e-04, -4.36894735e-03,  4.13419195e-02,
         -2.78691365e-03, -3.22899059e-03, -5.42938430e-03,
          4.60640236e-04, -3.29626985e-02, -1.33975944e-03,
         -1.69023033e-02,  3.29275406e-03,  4.00972459e-03],
                   
        [1.87410903e-03,  8.17836029e-04,  4.30113170e-03,
          3.27303447e-02, -2.48822849e-02,  1.66430101e-02,
         -2.60072004e-04, -2.99071497e-03,  2.04449077e-03,
         -3.28044430e-03, -5.10160904e-03, -6.70398492e-03,
         -8.65057751e-04, -3.33607048e-02, -1.37072662e-03,
         -1.92170106e-02,  6.97918353e-04,  4.21113428e-03],
                   
        [-1.73768355e-03, -1.03258190e-03,  8.47992115e-03,
          3.15088369e-02, -1.67861581e-02,  2.01348569e-02,
         -7.00937677e-03,  3.28904833e-03, -1.07546030e-02,
         -1.37725170e-03, -4.93122777e-03, -8.34029447e-03,
          8.97017657e-04, -2.61410046e-02, -1.64264359e-03,
         -1.57973617e-02,  7.40339048e-04,  2.65506003e-03],
                   
        [-8.15949403e-03, -1.35604758e-04,  1.06013082e-02,
          2.00399160e-02, -1.02066407e-02,  1.30257793e-02,
         -1.60173886e-02,  1.20612206e-02, -2.72354130e-02,
          6.46398403e-03, -6.41402975e-03, -7.89591298e-03,
          5.86519483e-04, -2.53124405e-02, -1.86213164e-03,
         -5.91297681e-03,  5.74062346e-04,  2.14769086e-03],
                   
        [-9.77965351e-03, -1.39463739e-03, -2.80151796e-03,
         -6.05971599e-03,  1.72903370e-02,  1.19558431e-03,
          2.35291664e-03,  4.16230364e-03, -3.17660533e-02,
          1.42680923e-03, -8.35874118e-03, -6.03611115e-03,
         -1.37069833e-03, -3.33966836e-02, -1.71921239e-03,
         -4.23008390e-03,  4.95605520e-04,  1.70980697e-03],
                   
        [-1.29434448e-02, -1.59713102e-03, -6.54948363e-03,
         -2.54679695e-02,  4.30313870e-02, -1.12912226e-02,
          4.27989289e-03,  2.74793641e-03, -3.82700078e-02,
          6.74198521e-03, -1.02390414e-02, -5.51288016e-03,
         -9.26182547e-04, -3.23635489e-02, -2.23907758e-03,
          2.37068089e-05,  6.73407165e-04,  1.28452328e-03],
                   
        [-1.22452145e-02, -2.26697163e-03, -7.00565660e-03,
         -2.35434789e-02,  1.14653260e-02,  7.30588520e-03,
          4.86261584e-03,  8.75421427e-03, -4.07758839e-02,
          5.26203401e-03, -5.66830207e-03, -6.07494032e-03,
          2.39119702e-03, -3.32556702e-02, -2.57747411e-03,
          7.82727636e-03,  5.29345241e-04,  4.97613393e-04],
                   
        [-1.55314347e-02, -4.62123338e-04, -1.07944207e-02,
         -2.66383588e-02,  4.10183705e-02, -1.84664484e-02,
          1.68761087e-03,  7.59532861e-03, -2.10026950e-02,
         -2.93137832e-03, -3.37803806e-03, -8.44099186e-03,
          1.50625396e-03, -3.14089134e-02, -1.51637266e-03,
          3.87080805e-03, -1.66192430e-03, -2.94341648e-04],
                   
        [-1.70360319e-02,  4.98240290e-04, -9.27673362e-04,
          2.05317592e-05,  9.22270771e-03, -2.01043189e-02,
          3.07649164e-03, -3.30243446e-03, -1.59196574e-02,
         -2.92849960e-03, -4.18327795e-03, -1.01872161e-02,
          6.61080005e-03, -2.00458448e-02, -2.16689217e-03,
         -1.92473433e-03,  1.74601935e-03, -7.53690954e-04],
                   
        [-1.45101165e-02,  1.08633481e-03, -1.50221819e-03,
          4.36263764e-03,  1.32262977e-02, -2.99540758e-02,
          1.71765895e-03, -1.53972756e-03,  2.22219639e-02,
         -3.92881036e-03, -3.53255495e-03, -8.73056985e-03,
          4.83861612e-03, -1.84153374e-02, -2.33037700e-03,
         -1.73530597e-02, -9.42786864e-04, -1.40932598e-03],
                   
        [-2.79291067e-02,  7.75961846e-04, -1.67368678e-03,
         -1.79934106e-03,  2.11611260e-02, -3.87369059e-02,
          3.54862795e-03, -5.74171869e-03,  4.11593914e-02,
         -4.18859301e-03, -3.72431614e-03, -6.70281518e-03,
          6.99011190e-03, -2.82287542e-02, -2.44962587e-03,
         -1.84938032e-02, -2.05021747e-03, -2.22685840e-03],
                   
        [-6.11664215e-03,  1.45033293e-03, -6.54179184e-03,
          1.52186584e-02,  1.62637164e-03, -2.08775662e-02,
          1.99746131e-03, -8.25796556e-03,  5.03733344e-02,
         -4.99503873e-03, -1.02623156e-03, -5.12921624e-03,
          4.52045671e-04,  1.23958895e-03, -1.04366045e-03,
         -2.20232792e-02, -1.89182849e-03, -3.21546686e-03],
                   
        [-4.23329696e-03,  1.92260952e-03,  9.85658960e-04,
          2.65115462e-02, -1.64232182e-03, -1.87600087e-02,
          1.93231343e-03, -1.16155995e-03,  4.19396609e-02,
         -3.50433961e-03, -2.19895854e-03, -5.51427156e-03,
          4.80194518e-04, -3.06371646e-03, -3.52524081e-03,
         -1.74480211e-02,  7.47543163e-05, -3.88583937e-03],
                   
        [1.36704103e-03,  1.02080661e-03, -2.10598484e-03,
          3.77250127e-02, -1.69778746e-02, -9.19245882e-04,
          5.23353787e-03, -5.50091127e-03,  1.65197290e-02,
         -3.93319456e-03, -5.34757320e-03, -5.28483139e-03,
         -5.04200207e-03, -6.14599884e-03, -3.99033213e-03,
         -2.17563212e-02, -1.47264136e-03, -4.28709062e-03],
                   
        [-1.46247272e-03, -1.57702365e-03,  5.89598436e-03,
          4.24824916e-02, -2.32027341e-02,  2.45868731e-02,
         -1.99274998e-03,  6.46002404e-03, -2.88530346e-02,
         -2.15467322e-03, -3.52529669e-03, -5.71665075e-03,
          4.69931541e-03, -1.03305485e-02, -3.49102938e-03,
         -1.86834298e-02, -7.60147639e-04, -3.62309557e-03],
                   
        [1.86585370e-04, -6.89294015e-04,  9.52232257e-03,
          2.70879734e-02, -3.03317178e-02,  3.50458361e-02,
         -1.25583373e-02,  3.36104725e-03, -3.60212997e-02,
         -1.76215626e-03, -2.97341635e-03, -7.32240221e-03,
          6.16864534e-03, -1.00793121e-02, -3.44743649e-03,
         -7.28188921e-03, -2.35085539e-03, -3.22515005e-03],
                   
        [-1.07350331e-02, -2.03957246e-03, -3.96988774e-03,
          1.02152126e-02, -1.50731821e-02,  2.12810840e-02,
          2.53679376e-04,  2.54714140e-03, -5.03333621e-02,
          1.20046898e-03, -7.99648836e-03, -4.08936990e-03,
         -6.89459266e-04, -1.07963961e-02, -3.38995433e-03,
         -7.04406621e-03, -2.44290772e-04, -3.19778593e-03],
                   
        [-8.81627481e-03, -5.99310908e-04, -3.19746253e-03,
         -2.12441739e-02,  2.48745736e-02, -9.74084716e-04,
         -3.58108082e-03,  4.70031146e-03, -4.43474799e-02,
          5.66944899e-03, -1.00030638e-02, -4.24292358e-03,
         -8.77651037e-04, -9.22032073e-03, -3.60374269e-03,
         -2.24848860e-03, -2.09333119e-03, -3.07639269e-03],
                   
        [-7.73078622e-03, -2.68002436e-03, -1.05096018e-02,
         -3.20781022e-02,  3.10305823e-02, -4.61669732e-03,
          8.09112447e-04,  5.10913925e-03, -3.67721468e-02,
          1.74991805e-02, -2.90492317e-03, -2.32674438e-03,
          5.80509752e-03, -1.02832066e-02, -3.58918519e-03,
          5.82328904e-03, -1.56172365e-03, -3.22169112e-03],
                   
        [-1.51959890e-02, -4.45540092e-04, -7.39430636e-03,
         -1.62583906e-02,  4.82000643e-03,  4.79325978e-03,
         -2.32608733e-03,  3.76569486e-04, -2.67435256e-02,
         -1.13612576e-03, -3.03718843e-03, -2.84173666e-03,
          1.99531252e-03, -8.29647109e-03, -2.45142239e-03,
          2.19182950e-03, -2.72843824e-03, -3.97775462e-03],
                   
        [-9.14163049e-03,  1.12673210e-03, -2.98561342e-03,
          6.67838566e-03, -3.77204083e-03, -5.44741424e-03,
          1.56690471e-03, -5.67982532e-03, -6.40143594e-03,
         -7.05961138e-05, -3.29612545e-03, -1.65848038e-03,
          4.80439048e-03, -9.86603182e-03, -2.25911639e-03,
         -4.17489093e-03, -2.51824036e-04, -4.15934063e-03],
                   
        [-1.14836525e-02,  7.47481827e-04,  5.05208829e-03,
          3.30525227e-02, -1.39971273e-02, -7.80009851e-03,
         -4.08052187e-03, -6.22018706e-03,  1.27807977e-02,
         -2.58079614e-03, -6.71766652e-03, -1.89971889e-03,
          4.37216973e-03, -9.97895095e-03, -2.22787494e-03,
         -2.13507023e-02, -3.15412902e-03, -3.50207952e-03]]

new_shape_values = np.array(new_shape_values)
print(new_shape_values)


# In[13]:


filename = 'procedure_imported_data/shap_test.csv'

X = pd.read_csv(filename, engine='python')
print(shap_values)

shap.summary_plot(new_shape_values[:24, :], X.iloc[:24, :], show=False)
plt.savefig('Feature density scatter plot.pdf', bbox_inches = 'tight', dpi=600)


# In[11]:


shap.force_plot(explainer.expected_value, new_shape_values[:24, :], X.iloc[:24, :])


# In[15]:


shap.summary_plot(new_shape_values[:24, :], X.iloc[:24,:], plot_type="bar", show=False)
plt.savefig('Feature importance SHAP value.pdf', bbox_inches = 'tight', dpi=600)

