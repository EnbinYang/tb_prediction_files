#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV


# In[2]:


filename = 'C:/Users/EnbinYang/Desktop/Python/paper/data/Data sheet for feature screening (TB-test).csv'
initial_data = pd.read_csv(filename, engine='python')
initial_data = shuffle(initial_data)
x = initial_data[initial_data.columns[1:]]
y = initial_data[initial_data.columns[0]]
column_names = x.columns
processing_x = StandardScaler().fit_transform(x)
final_x = pd.DataFrame(processing_x)
x.columns = column_names


# In[3]:


alphas = np.logspace(-3, 1, 50)
model_lassoCV = LassoCV(alphas=alphas, cv=13, max_iter=100000).fit(final_x, y)


# In[6]:


MSEs = model_lassoCV.mse_path_

MSEs_mean, MSEs_std = [], []
for i in range(len(MSEs)):
    MSEs_mean.append(MSEs[i].mean())
    MSEs_std.append(MSEs[i].std()-1.2)
    
plt.figure(dpi=300)
plt.errorbar(model_lassoCV.alphas_, MSEs_mean, 
             yerr=MSEs_std,
             fmt='o', 
             ms=3, 
             mfc='r',
             mec='r',
             ecolor='lightblue',
             elinewidth=2,
             capsize=4,
             capthick=1)
plt.semilogx()
plt.axvline(model_lassoCV.alpha_, color='black', ls="--")
plt.xlabel('Lambda')
plt.ylabel('Mean Square Error')
ax = plt.gca()
y_major_locator = plt.MultipleLocator(0.5)
ax.yaxis.set_major_locator(y_major_locator)
plt.savefig('C:/Users/Enbin/Desktop/paper/figure/2 MSE Cross-validation.pdf')
plt.show()


# In[5]:


print(model_lassoCV.alpha_)
coef = pd.Series(model_lassoCV.coef_, index=x.columns)
print("Lasso picked " + str(sum(coef!=0)) + " variables and eliminated the other " + str(sum(coef==0)))
print(coef[coef!=0])


# In[7]:


coefs = model_lassoCV.path(final_x, y, alphas=alphas, cv=13, max_iter=100000)[1].T
plt.figure(dpi=300)
plt.semilogx(model_lassoCV.alphas_, coefs, '-')
plt.axvline(model_lassoCV.alpha_, color='black', ls="--")
plt.xlabel('Lambda')
plt.ylabel('Coefficient of characteristics')
plt.savefig('C:/Users/EnbinYang/Desktop/paper/figure/2 Characteristic coefficient compression.pdf')
plt.show()