import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

"""
Using the LASSO algorithm to remove irrelevant factors.
"""

filename = 'procedure_imported_data/TB_feature_screening.csv'
initial_data = pd.read_csv(filename, engine='python')
initial_data = shuffle(initial_data)
x = initial_data[initial_data.columns[1:]]
y = initial_data[initial_data.columns[0]]
column_names = x.columns
processing_x = StandardScaler().fit_transform(x)
final_x = pd.DataFrame(processing_x)
x.columns = column_names

alphas = np.logspace(-5, 2, 200)
model_lassoCV = LassoCV(alphas=alphas, cv=13, max_iter=100000).fit(final_x, y)

print(model_lassoCV.alpha_)
coef = pd.Series(model_lassoCV.coef_, index=x.columns)
print("Lasso picked " + str(sum(coef!=0)) + " variables and eliminated the other " + str(sum(coef==0)))
print(coef[coef!=0])

