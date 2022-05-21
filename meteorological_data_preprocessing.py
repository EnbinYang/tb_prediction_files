import pandas as pd
from sklearn.impute import KNNImputer

"""
This procedure is capable of pre-processing 15 meteorological factors at once.
"""

# import data for preprocessing
file_name = 'raw_data/raw_meteorological_data.csv'
data = pd.read_csv(file_name, engine='python')
print(data.head(6))

# default value addition using KNN algorithm
KI = KNNImputer(n_neighbors=5, weights='uniform')

# data training
data_transformed = KI.fit_transform(data)
print(data_transformed)

# transforming data formats
completed_processing_data = pd.DataFrame(data_transformed)
print(completed_processing_data)
output_file_name = 'compeleted_data/compeleted_meteorological_data.csv'
completed_processing_data.to_csv(output_file_name)

