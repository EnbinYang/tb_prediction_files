import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

"""
If filling out the same missing values using kNN, the multivariate feature imputation is available.
"""

# load data
file_name = 'raw_data/power.csv'
data = pd.read_csv(file_name, encoding='gbk')
print(data.head(6))

# multivariate feature imputation
II = IterativeImputer(max_iter=10, random_state=0)

# training
data_transformed = II.fit_transform(data)
print(data_transformed)

# converted data format
completed_processing_data = pd.DataFrame(data_transformed)
print(completed_processing_data)
output_file_name = 'completed_data/completed_power.csv'
completed_processing_data.to_csv(output_file_name)

