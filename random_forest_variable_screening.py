import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

filename = 'data/用于特征筛选的数据表（肺结核-Lasso筛选后17个）.csv'
initial_data = pd.read_csv(filename, engine='python')
x = initial_data[initial_data.columns[1:]]
y = initial_data[initial_data.columns[0]]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
param_grid = {
    'n_estimators':[5, 10, 20, 50, 100, 200],
    'max_depth':[3, 5, 7],
    'max_features':[0.6, 0.7, 0.8, 1]
}

rf = RandomForestRegressor()
grid = GridSearchCV(rf, param_grid=param_grid, cv=3)
grid.fit(x_train, y_train)

rf_reg = grid.best_estimator_
print(rf_reg)
estimator = rf_reg.estimators_[3]

print('特征重要性排序结果：')
feature_names=x.columns
feature_importances=rf_reg.feature_importances_
indices=np.argsort(feature_importances)[::-1]

for index in indices:
    print("%s:%f" %(feature_names[index], feature_importances[index]))