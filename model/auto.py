from scipy.interpolate import griddata
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime
from math import sqrt
from sklearn.metrics import mean_squared_error
import  scipy.signal.signaltools
import datetime as dt
from dateutil import parser
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
import os
from tpot import TPOTRegressor
from tensorflow.keras.callbacks import EarlyStopping

#Reading previous data with river(climatology) and physics and atm
ARC = '/users/modellers/dba/Documents/PML/projects/ML/EXPER02/input_data/interp_atobs/ICES_LATEST_INTERP/riverine/output/merged/'
merged_df = pd.read_csv(ARC+'interp_DOMall_ICES_era5_riv_bgc_phy_1993_2018_mod.csv', index_col='Date', parse_dates=True, encoding='latin1')
merged_df = merged_df.sort_index(ascending=True)
merged_df=merged_df.drop_duplicates()
#removing previous riverine data (climatology) and keep only physics and atm
columns_to_drop = [col for col in merged_df.columns if col.startswith('closest_')]
merged_df = merged_df.drop(columns=columns_to_drop)

#Reading previous data with river with new 7km spread daily
ARC = '/users/modellers/dba/Documents/PML/projects/ML/EXPER02/input_data/interp_atobs/ICES_LATEST_INTERP_forbathy/riverine_daily/output/merged/'
merged_riv_df = pd.read_csv(ARC+'interp_full_ICES_riverine_7km.csv', index_col='Date', parse_dates=True, encoding='latin1')
merged_riv_df = merged_riv_df.sort_index(ascending=True)
merged_riv_df=merged_riv_df.drop_duplicates()


#merge river and physics and atm
merged_riv_df_reset = merged_riv_df.reset_index()
merged_df_reset = merged_df.reset_index()
merged_combined_df = pd.merge(merged_riv_df_reset, merged_df_reset,
                              on=['Date', 'lon', 'lat', 'WCDepth'])

merged_combined_df = merged_combined_df.drop(columns=['month_x', 'day_x', 'no3_obs_x'])

# Rename columns by removing the '_x' suffix
merged_combined_df.columns = [col.replace('_x', '') for col in merged_combined_df.columns]

# Rename specific columns to new names
rename_columns = {
    'ronh4_interp': 'closest_ronh4',
    'rorunoff_interp': 'closest_rorunoff',
    'rop_interp': 'closest_rop',
    'roo_interp': 'closest_roo',
    'rosio2_interp': 'closest_rosio2',
    'rono3_interp': 'closest_rono3',
    'month_y': 'month',
    'day_y': 'day',
    'no3_obs_y': 'no3_obs'
}

merged_combined_df.rename(columns=rename_columns, inplace=True)


ARC = '/users/modellers/dba/Documents/PML/projects/ML/EXPER02/input_data/interp_atobs/ICES_LATEST_INTERP_forbathy/output_bathy/merged/'
merged_bathy_df = pd.read_csv(ARC+'interp_DOM_ALL_ICES_riverine_era5bgcrean.csv', index_col='Date', parse_dates=True, encoding='latin1')
merged_bathy_df = merged_bathy_df.sort_index(ascending=True)
merged_bathy_df=merged_bathy_df.drop_duplicates()
merged_bathy_df = merged_bathy_df[['WCDepth', 'lon', 'lat', 'bathy_interp']]
merged_bathy_df_reset = merged_bathy_df.reset_index()

merged_df = pd.merge(merged_bathy_df_reset, merged_combined_df,
                           on=['Date', 'lon', 'lat', 'WCDepth'])
merged_df = merged_df.sort_index(ascending=True)
merged_df=merged_df.drop_duplicates()
merged_df=merged_df.set_index('Date')

#-------------------------------------------------------------
#-------------TRAINING MODEL ---------------------------------
#-------------------------------------------------------------
X = merged_df.drop('no3_obs',axis=1) #discarding no3 observation for training and testing
model_no3 = X['no3_interp']
model_no3 = model_no3.sort_index(ascending=True)
X = X.drop('no3_interp',axis=1) #discarding model no3 from input feature
X = X.drop('o2_interp',axis=1) #discarding model o2 from input feature
y = merged_df['no3_obs'] #keeping observed no3 for training and testing
training_order = ['lon', 'lat',
       'WCDepth', 'chl_interp', 'pft_interp', 'sstost_interp',
       'temp_interp', 'd2m_interp', 't2m_interp', 'sp_interp', 'tp_interp',
       'u10_interp', 'v10_interp', 'ssrdc_interp', 'phyc_interp', 'pp_interp',
       'closest_rorunoff', 'closest_rono3', 'closest_ronh4', 'closest_roo',
       'closest_rop', 'closest_rosio2', 'bathy_interp', 'month', 'day']
X = X.reindex(columns=training_order)
print(" ")
print("Dataset : FULL --")

print(X)

#TRAIN and VALD
start_date='1998-01-01'
end_date='2015-12-31'
X_t_v = X.loc[start_date:end_date] #slicing according to experiment period
y_t_v = y.loc[start_date:end_date]

print(X_t_v.columns)

import autokeras as ak
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_t_v, y_t_v, test_size=0.2)#, random_state=42)
regressor = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=200  # Number of different models to try
)

# Configure early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    min_delta=0.001,  # Minimum change to qualify as an improvement
    patience=10,  # How many epochs to wait before stopping
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity.
)

# Train the regressor with early stopping
regressor.fit(
    X_train,
    y_train,
    epochs=100,  
    callbacks=[early_stopping]
)

print(regressor.evaluate(X_test, y_test))

model = regressor.export_model()
print(model.summary())

model.save("TRAINED/best_model_autokeras", save_format='tf')
