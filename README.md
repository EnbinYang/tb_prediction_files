## A multivariate multi-step LSTM forecasting  model for tuberculosis incidence with model  explanation in Liaoning Province, China

This study has been published in *BMC Infectious Diseases* (Received: 28 November 2021, Accepted: 10 May 2022, Published: 23 May 2022). 

**Article Link (open access):** https://doi.org/10.1186/s12879-022-07462-8

## Data and Code

The full data and code are available at the link below.

- **Google Drive:** https://drive.google.com/drive/folders/1bKnOXzVA9ZAKJUOwcbOuUKkySOOBnrvN?usp=sharing

- **Baidu Netdisk:** https://pan.baidu.com/s/1-IIZgK368a9kjgf54LGyYQ?pwd=p83d (password: p83d) 

We also provide all the raw data. Notice that the data have not been pre-processed at all, are very messy, and in Chinese. It is available at the link below on request.

- **Google Drive:** https://drive.google.com/drive/folders/1AFGVDgL1pz1wi1w0UZahJS9JxdVaq2SL?usp=sharing

- **Baidu Netdisk:** https://pan.baidu.com/s/19OEu9KYe1OxxB9iFNkUoPQ?pwd=feyb (password: feyb) 

## Data Files Description

All the data we use are stored in three folders. The **"raw_data"** folder contains raw meteorological, economic and social data, where there are missing values. The data processed by the procedure will be stored in the **"completed_data"** folder. In some cases, we have combined some data to form new sheets for the procedure, and these files are stored in the **"procedure_imported_data"** folder.

All data is saved in <u>".csv"</u> files, which you can open directly to see the exact content.

## Code Files Description

Running the code in the following order can reproduce the results of our experiments.

1. **"meteorological_data_preprocessing.py":** using the kNN algorithm to fill in the missing values of the meteorological data.
2. **"economic_social_data_preprocessing.py":** using the multivariate feature imputation methods to fill in missing values of the economic and social data.
3. **"lasso_feature_screening.py":** using the LASSO to remove irrelevant factors, only 17 factors were left finally.
4. **"random_forest.py":** calculating the importance of features and obtaining the set of factors with size 10 and 5, respectively.
5. **"ARIMA_model.py":** the ARIMA model is a baseline.
6. **"SARIMA_model.py":** the SARIMA model is based on the ARIMA model with seasonal differencing.
7. **"multivariate_multistep_LSTM_model.py":** mining the relationship between tuberculosis (TB) and influencing factors. The "multi-step" was able to eliminate the increasing trend of the sequence.
8. **"multistep_ARIMA_LSTM_model.py":** the optimal prediction values of the ARIMA and LSTM model are used as input, adding multi-step lengths and using LSTM for training.
9. **"shap_analysis.py":** using the SHapley Additive exPlanation (SHAP) method to interpret the prediction results of multivariate multi-step LSTM models.

## Citation

Yang, E., Zhang, H., Guo, X. *et al.* A multivariate multi-step LSTM forecasting model for tuberculosis incidence with model explanation in Liaoning Province, China. *BMC Infect Dis* **22,** 490 (2022). https://doi.org/10.1186/s12879-022-07462-8



