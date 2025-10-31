import time 
start_time = time.time()

import netCDF4 as nc
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='pandas')
import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import glob
import matplotlib.pyplot as plt
import datetime
from math import sqrt
from sklearn.metrics import mean_squared_error
import scipy.signal.signaltools
import datetime as dt
from dateutil import parser
from keras.preprocessing.sequence import TimeseriesGenerator
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Activation
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import GRU, Dense, Input, Masking, Attention
import xesmf as xe
from sklearn.preprocessing import MinMaxScaler
import math
import calendar
from concurrent.futures import ProcessPoolExecutor
from keras.models import load_model
import autokeras as ak
import sys
import os

#-----------------------------------------------------------------------------
year = sys.argv[1]
#cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 1))

cpus = 1 #USE PMPC and always try to use 1 core. For multiple cores this routine isnt working making wrong values in predictions.

def print_nitro_ai_title():
    title = """
  _   _ _____ _______ _____   ____                _____
 | \ | |_   _|__   __|  __ \ / __ \         /\   |_   _|
 |  \| | | |    | |  | |__) | |  | |______ /  \    | |
 | . ` | | |    | |  |  _  /| |  | |______/ /\ \   | |
 | |\  |_| |_   | |  | | \ \| |__| |     / ____ \ _| |_
 |_| \_|_____|  |_|  |_|  \_\\____/     /_/    \_\_____|
     """
    print("----------------------------------------------------------")
    print("               NITRO-AI Model Initiation")
    print("----------------------------------------------------------")
    print(title)
    print("----------------------------------------------------------")
    print(f'Cores utilized                     :: {cpus}')
    print('')
    print(f'Surface Nitrate Prediction for Year  :: {year}')
    print('')
    print("Author: Deep S. Banerjee")
    print("Plymouth Marine Laboratory, UK")
    print("----------------------------------------------------------")

print_nitro_ai_title()

start_year = int(year)
end_year = int(year)
dep = 0.
#-----------------------------------------------------------------------------

variable_name = 'phyc'
ARC = '/data/proteus1/scratch/dba/CMEMS_BGC/PHYC/merged/interpolated/'
file_path = os.path.join(ARC, f'REG_AMM7_metoffice_foam1_amm7_NWS_PHYT_dm_{year}.nc')
ds_ref = xr.open_dataset(file_path)
ds_lon = ds_ref['longitude']
ds_lat = ds_ref['latitude']

#------------------USER DEFINED FUNCTIONS-------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

#------------------  FUNCTIONS ENDS HERE  ----------------------------------

#mask
var_name = 'mask'
ARC = '/data/proteus1/scratch/dba/dream_input/static_files/'
file_name = os.path.join(ARC, f'lsm_amm7_surface.nc')
ds_mask = xr.open_dataset(file_name)
mask_amm7 = ds_mask['mask']
masking_var = mask_amm7 == 1

#bathymetry
var_name = 'bath'
ARC_BATHY = '/data/proteus1/scratch/dba/bathy/interpolate/'
file_name = os.path.join(ARC_BATHY, 'masked_bathy.nc')
ds_bathy = xr.open_dataset(file_name)

#For pp
var_name = 'nppv'
ARC = '/data/proteus1/scratch/dba/CMEMS_BGC/PP/merged/interpolated/'
file_name = os.path.join(ARC, f'REG_AMM7_metoffice_foam1_amm7_NWS_PPRD_dm_{year}.nc')
ds_pp = xr.open_dataset(file_name)

#For riverine
ARC = '/data/proteus4/scratch/dba/RIVERINE/river_spread/'
file_name = os.path.join(ARC, f'NOWMAPS_rivers_sprd.{year}.nc')
ds_riv = xr.open_dataset(file_name)

ds_riv['ronh4'] = ds_riv['ronh4'].where(masking_var)
ds_riv['ronh4'] = ds_riv['ronh4'].fillna(0)

ds_riv['rorunoff'] = ds_riv['rorunoff'].where(masking_var)
ds_riv['rorunoff'] = ds_riv['rorunoff'].fillna(0)

ds_riv['rop'] = ds_riv['rop'].where(masking_var)
ds_riv['rop'] = ds_riv['rop'].fillna(0)

ds_riv['roo'] = ds_riv['roo'].where(masking_var)
ds_riv['roo'] = ds_riv['roo'].fillna(0)

ds_riv['rosio2'] = ds_riv['rosio2'].where(masking_var)
ds_riv['rosio2'] = ds_riv['rosio2'].fillna(0)

ds_riv['rono3'] = ds_riv['rono3'].where(masking_var)
ds_riv['rono3'] = ds_riv['rono3'].fillna(0)

#For phyc
var_name = 'phyc'
ARC = '/data/proteus1/scratch/dba/CMEMS_BGC/PHYC/merged/interpolated/'
file_name = os.path.join(ARC, f'REG_AMM7_metoffice_foam1_amm7_NWS_PHYT_dm_{year}.nc')
ds_phyc = xr.open_dataset(file_name)

#For chl
var_name = 'chl'
ARC = '/data/proteus1/scratch/dba/CMEMS_BGC/CHL/cmems_mod_nws_bgc-chl_my_7km-3D_P1D-m/merged/interpolate/'
file_name = os.path.join(ARC, f'REG_AMM7_metoffice_foam1_amm7_NWS_CPWC_dm_{year}.nc')
ds_chl = xr.open_dataset(file_name)

#For pft
var_name = 'diato'
ARC = '/data/proteus1/scratch/dba/CMEMS_BGC/PFT/cmems_mod_nws_bgc-pft_my_7km-3D-diato_P1D-m/merged/interpolate/'
file_name = os.path.join(ARC, f'REG_AMM7_metoffice_foam1_amm7_NWS_DIATO_CPWC_dm_{year}.nc')
ds_pft = xr.open_dataset(file_name)

#For temp
var_name = 'thetao'
ARC = '/data/proteus1/scratch/dba/CMEMS_PHY/TEMP/merged/interpolated/'
file_name = os.path.join(ARC, f'REG_AMM7_metoffice_foam1_amm7_NWS_TEMP_dm_{year}.nc')
ds_temp = xr.open_dataset(file_name)

#For sst
var_name = 'analysed_sst'
ARC = '/data/proteus1/scratch/dba/OSTIA_SST/merged/interpolate_2_amm7/'
file_name = os.path.join(ARC, f'REG_AMM7_UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB_REP-v02p0-fv02_{year}.nc')
ds_sst = xr.open_dataset(file_name)

#For era5
ARC = '/data/proteus1/scratch/dba/ERA5/extracted_atObs/interpolated_2_amm7/'
file_name = os.path.join(ARC, f'REG_AMM7_era5_{year}_1D.nc')
ds_era = xr.open_dataset(file_name)

#For era5_rad
var_name = 'ssrdc'
ARC = '/data/proteus1/scratch/dba/ERA5/radiation/interpolated_2_amm7/'
file_name = os.path.join(ARC, f'REG_AMM7_era5_ssrdc_{year}.nc')
ds_erad = xr.open_dataset(file_name)

#DEFININF FUNCTION FOR PREDICTION
def predict_for_index(index, ds_riv):
        i, j = index
  
        value_lon = ds_lon[j].values
        value_lat = ds_lat[i].values

        tser_nppv = []
        tser_phyc = []
        tser_chl = []
        tser_pft = []
        tser_temp = []
        tser_sst = []
        tser_d2m = []
        tser_sp = []
        tser_t2m = []
        tser_tp = []
        tser_u10 = []
        tser_v10 = []
        tser_ssrdc = []
        tser_rorunoff = [] #
        tser_rono3 = [] #
        tser_ronh4 = [] #
        tser_roo = [] #
        tser_rop = [] #
        tser_rosio2 = [] #
        tser_lon = []
        tser_lat = []
    
        for year in range(start_year, end_year + 1):  # 2010 to 2014 inclusive
    
            #For pp
            var_name = 'nppv'
            value = ds_pp[var_name][:, 0, i, j].values
            tser_nppv.extend(value)
            #print('pp done')

            #For riverine
            value_ronh4 = ds_riv['ronh4'][:, i, j].values
            tser_ronh4.extend(value_ronh4)

            value_rorunoff = ds_riv['rorunoff'][:, i, j].values
            tser_rorunoff.extend(value_rorunoff)

            value_rop = ds_riv['rop'][:, i, j].values
            tser_rop.extend(value_rop)

            value_roo = ds_riv['roo'][:, i, j].values
            tser_roo.extend(value_roo)

            value_rosio2 = ds_riv['rosio2'][:, i, j].values
            tser_rosio2.extend(value_rosio2)

            value_rono3 = ds_riv['rono3'][:, i, j].values
            tser_rono3.extend(value_rono3)
            #print('riverine done')

            #For phyc
            var_name = 'phyc'
            value = ds_phyc[var_name][:, 0, i, j].values  
            tser_phyc.extend(value)
 
            #For chl
            var_name = 'chl'
            value = ds_chl[var_name][:, 0, i, j].values  
            tser_chl.extend(value)
    
            #For pft
            var_name = 'diato'
            value = ds_pft[var_name][:, 0, i, j].values  
            tser_pft.extend(value)
    
            #For temp
            var_name = 'thetao'
            value = ds_temp[var_name][:, 0, i, j].values  
            tser_temp.extend(value)

            #For sst
            var_name = 'analysed_sst'
            value = ds_sst[var_name][:, i, j].values - 273.
            tser_sst.extend(value)

            #For era5
            value_d2m = ds_era['d2m'][:, i, j].values  
            tser_d2m.extend(value_d2m)
    
            value_sp = ds_era['sp'][:, i, j].values  
            tser_sp.extend(value_sp)
    
            value_t2m = ds_era['t2m'][:, i, j].values 
            tser_t2m.extend(value_t2m)
    
            value_tp = ds_era['tp'][:, i, j].values
            tser_tp.extend(value_tp)
    
            value_u10 = ds_era['u10'][:, i, j].values
            tser_u10.extend(value_u10)
    
            value_v10 = ds_era['v10'][:, i, j].values
            tser_v10.extend(value_v10)
    
            #For era5_rad
            var_name = 'ssrdc'
            value = ds_erad[var_name][:, 0, i, j].values  
            tser_ssrdc.extend(value)

            var_name = 'bath'
            bathy_value = ds_bathy[var_name][i, j].values
            
            days_in_year = 366 if calendar.isleap(year) else 365
            time_series = np.full(days_in_year, bathy_value)
            dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31")
            df_bathy = pd.DataFrame(time_series, index=dates, columns=[var_name])

        tser_nppv = np.array(tser_nppv)
        tser_phyc = np.array(tser_phyc)
        tser_chl = np.array(tser_chl)
        tser_pft = np.array(tser_pft)
        tser_temp = np.array(tser_temp)
        tser_sst = np.array(tser_sst)
        tser_d2m = np.array(tser_d2m)
        tser_sp = np.array(tser_sp)
        tser_t2m = np.array(tser_t2m)
        tser_tp = np.array(tser_tp)
        tser_u10 = np.array(tser_u10)
        tser_v10 = np.array(tser_v10)
        tser_ssrdc = np.array(tser_ssrdc)

        tser_ronh4 = np.array(tser_ronh4)
        tser_rorunoff = np.array(tser_rorunoff)
        tser_rop = np.array(tser_rop)
        tser_roo = np.array(tser_roo)
        tser_rosio2 = np.array(tser_rosio2)
        tser_rono3 = np.array(tser_rono3)

        date_index = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='D')
    
        if len(date_index) != len(tser_nppv):
          raise ValueError("The length of the time series must be equal to the length of the date_index")
    
        # Create the DataFrame
        df = pd.DataFrame({
            'pp_interp': tser_nppv,
            'phyc_interp': tser_phyc,
            'chl_interp': tser_chl,
            'pft_interp': tser_pft,
            'temp_interp': tser_temp,
            'sstost_interp': tser_sst,
            'd2m_interp': tser_d2m,
            'sp_interp': tser_sp,
            't2m_interp': tser_t2m,
            'tp_interp': tser_tp,
            'u10_interp': tser_u10,
            'v10_interp': tser_v10,
            'ssrdc_interp': tser_ssrdc,
            'closest_ronh4': tser_ronh4,
            'closest_rorunoff': tser_rorunoff,
            'closest_rop': tser_rop,
            'closest_roo': tser_roo,
            'closest_rosio2': tser_rosio2,
            'closest_rono3': tser_rono3
        }, index=date_index)

        df = df.join(df_bathy)

        rename_list = {
            'bath': 'bathy_interp'
        }

        df.rename(columns=rename_list, inplace=True)

        df['lon'] = value_lon
        df['lat'] = value_lat
        df['WCDepth'] = dep
        df.index.name = 'Date'
        df['month'] = df.index.month
        df['day'] = df.index.day

        training_order = ['lon', 'lat',
               'WCDepth', 'chl_interp', 'pft_interp', 'sstost_interp',
               'temp_interp', 'd2m_interp', 't2m_interp', 'sp_interp', 'tp_interp',
               'u10_interp', 'v10_interp', 'ssrdc_interp', 'phyc_interp', 'pp_interp',
               'closest_rorunoff', 'closest_rono3', 'closest_ronh4', 'closest_roo',
               'closest_rop', 'closest_rosio2', 'bathy_interp', 'month', 'day']

        df = df.reindex(columns=training_order)
        df = df.sort_index(ascending=True)
        df = df.astype('float64')

        #Prediction
        if df.isnull().any().any():
            # If there are NaN values, assign y_pred2 all values as NaNs
            y_pred2 = np.full((df.shape[0], 1), np.nan)
            print("NaN Found")
        else:
            # If there are no NaN values, proceed with the prediction
            model_newscaled = load_model('../../model/structured_data_regressor/best_model') 
            y_pred2 = model_newscaled.predict(df, verbose=0)

        y_pred2=y_pred2.reshape(-1,1)
        ydays = 366 if calendar.isleap(start_year) else 365
        y_pred2 = y_pred2[:ydays, 0]

        print(f"Predicted for position :: {value_lat.item():.2f} , {value_lon.item():.2f}")
        return (i, j, y_pred2)


def wrapper(index):
    return predict_for_index(index, ds_riv)

#MAKING PREDICTIONS AND SAVING NETCDF PARALLELY
#NetCDF output
with nc.Dataset(f'NO3_predictions_RF_paral_{year}.nc', 'w', format='NETCDF4') as dsout:
    # Define dimensions
    dsout.createDimension('time', len(ds_ref['time']))
    dsout.createDimension('lat', len(ds_ref['latitude']))
    dsout.createDimension('lon', len(ds_ref['longitude']))

    # Define variables
    times = dsout.createVariable('time', 'f4', ('time',))
    latitudes = dsout.createVariable('lat', 'f4', ('lat',))
    longitudes = dsout.createVariable('lon', 'f4', ('lon',))
    predictions = dsout.createVariable('prediction', 'f4', ('time', 'lat', 'lon'))

    # Copy attributes from ds_ref to the new variables
    times.setncatts(ds_ref['time'].attrs)
    latitudes.setncatts(ds_ref['latitude'].attrs)
    longitudes.setncatts(ds_ref['longitude'].attrs)


    reference_date = np.datetime64('1970-01-01')
    time_values_as_days_since_reference = (ds_ref['time'].values - reference_date) / np.timedelta64(1, 'D')
    times[:] = time_values_as_days_since_reference
    times.units = "days since 1970-01-01"

    latitudes[:] = ds_ref['latitude'].values
    longitudes[:] = ds_ref['longitude'].values

    variable_data = ds_ref[variable_name][0,0,:].data  # This will give you a numpy array containing the data of the variable 'phyc'
    #getting the mask of the variable data
    if '_FillValue' in ds_ref[variable_name].attrs:
        fill_value = ds_ref[variable_name].attrs['_FillValue']
        mask = variable_data == fill_value
    else:
        mask = np.isnan(variable_data)
    unmasked_indices = np.column_stack(np.where(~mask))

    #Debug starts here:
    #lats_from_indices = ds_lat[unmasked_indices[:, 0]].values
    #lons_from_indices = ds_lon[unmasked_indices[:, 1]].values

    ## Apply your conditions
    ##condition = (lons_from_indices > -4) & (lons_from_indices < -3.5) & \
    ##        (lats_from_indices > 40) & (lats_from_indices < 65)
    #condition = (lons_from_indices >= -7) & (lons_from_indices <= 10) & \
    #        (lats_from_indices >= 48) & (lats_from_indices <= 58)

    ##Predicted for position :: 48.00 , -4.56
    #condition = (lons_from_indices >= -4.6) & (lons_from_indices <= -4.5) & \
    #        (lats_from_indices >= 48) & (lats_from_indices <= 49)

    # condition = (lons_from_indices >= 3.2) & (lons_from_indices <= 3.56) & \
    #         (lats_from_indices >= 51.5) & (lats_from_indices <= 51.74)
 
    ##        (lats_from_indices > 40) & (lats_from_indices < 65)
    #condition = (lons_from_indices >= -7) & (lons_from_indices <= 10) & \
    #        (lats_from_indices >= 48) & (lats_from_indices <= 58)
 
    #condition = (lons_from_indices >= -0.522) & (lons_from_indices <=9.477 ) & \
    #        (lats_from_indices >= 47) & (lats_from_indices <= 59)
 
    # condition = (lons_from_indices >= -0.522) & (lons_from_indices <=9.477 ) & \
    #         (lats_from_indices >= 47) & (lats_from_indices <= 53) #59)
 
 
    # #For Stonehaven (Scotland)
    # boundary_latitude_min = 56.96283
    # boundary_latitude_max = 57.13017
    # boundary_longitude_min = -2.11517
    # boundary_longitude_max = -2.10233
 
    # #For L4
    # boundary_latitude_min = 50.1
    # boundary_latitude_max = 50.2
    # boundary_longitude_min = -4.20
    # boundary_longitude_max = -4.10
 
    # condition = (lats_from_indices >= boundary_latitude_min) & \
    #                   (lats_from_indices <= boundary_latitude_max) & \
    #                   (lons_from_indices >= boundary_longitude_min) & \
    #                   (lons_from_indices <= boundary_longitude_max)


    #filtered_indices = unmasked_indices[condition]
    ##Debug ends here

    #----------------------------------
    time_series_dict = {}

    with ProcessPoolExecutor(max_workers=cpus) as executor:
        #mapping the wrapper over the indices
        #results = list(executor.map(wrapper, filtered_indices))
        results = list(executor.map(wrapper, unmasked_indices))        
    
    for i, j, y_pred2 in results:
        predictions[:, i, j] = np.squeeze(y_pred2)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")
