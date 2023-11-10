#!/usr/bin/env python
# coding: utf-8

# To process weather data with Python, we first need to import a few libraries:

# In[19]:


import numpy as np                # library for mathematical operations with arrays
import pandas as pd               # library for data frames; includes useful functions for date arithmetic
import xarray as xr               # library for arrays, especially tailored to weather data
import matplotlib.pyplot as plt   # library for basic plotting

from plotting import plot_fields                           # function for visualizing spatial data in a map
from miscellaneous import get_nearest_grid_index           # helper function to find the grid indices for a selected location
from miscellaneous import quantile_mapping                 # implementation of quantile mapping with various special cases
from miscellaneous import calculate_pentad_accumulations   # function to calculate pentad accumulations from raw forecast files
from miscellaneous import interpolate_forecasts            # function for bilinear interpolation of forecasts


# The following two variables specify the paths where the forecast and observation data are stored:

# In[20]:


fcst_dir = '/home/confer/michael/data/'
#fcst_dir = '/home/ghacof/COF65/gcm/'
data_dir = '/home/confer/michael/data/'


# Now, we set a number of parameters defining our forecast domain, training period, forecast year, etc.:

# In[37]:


year_training_start = 2000   # first year for which we have forecast and observation data for estimating percentiles
year_training_end = 2021     # last year for which we have forecast and observation data for estimating percentiles
year_fcst = 2023             # year for which the bias-corrected forecasts should be generated

month_init = 8               # forecast initialization month

lon_bounds = [20, 53]        # longitude range of the domain of interest
lat_bounds = [-15, 23]       # latitude range of the domain of interest

system = 'ecmwf'             # forecast system to be bias-corrected
ground_truth = 'chirps'      # data set against which the forecasts should be bias-corrected

res = 0.25                   # desired horizontal resolution of the ground truth data set


# The following paramters are calculated from those defined above:

# In[38]:


res_str = {0.1:'0p1', 0.25:'0p25'}[res]

month_names = {1:'jan', 2:'feb', 3:'mar', 4:'apr', 5:'may', 6:'jun', 7:'jul', 8:'aug', 9:'sep', 10:'oct', 11:'nov', 12:'dec'}
month_init_str = month_names[month_init]


# Load climatological percentiles of forecast data:

# In[39]:


filename_pct_fcst = f'{data_dir}percentiles_{system}_{res_str}_{year_training_start}_{year_training_end}_{month_init_str}.nc'

data_load = xr.open_dataset(filename_pct_fcst, engine='netcdf4')
prcp_fcst_pct = data_load.percentile.values
data_load.close()


# Load climatological percentiles and lon/lat coordinates of gound truth data:

# In[40]:


filename_pct_target = f'{data_dir}percentiles_{ground_truth}_{res_str}_{year_training_start}_{year_training_end}.nc'

data_load = xr.open_dataset(filename_pct_target, engine='netcdf4')
lon_target = data_load.lon.values
lat_target = data_load.lat.values
prcp_target_pct = data_load.percentile.values
data_load.close()

nlon = len(lon_target)
nlat = len(lat_target)


# Select a pentad (relative to the forecast initialization time) and a percentile level, and plot the respective climatological percentiles:

# In[41]:


ipct = 85                            # percentile level (integer between 1 and 99)
ipt = 21                             # number of pentads after forecast initialization dates
jpt = (6*(month_init-1)+ipt) % 72    # corresponding number of pentads after January 1

month_valid = jpt//6 + 1
month_valid_str = month_names[month_valid].capitalize()

dd_pentad_start = 5*(jpt-6*(month_valid-1)) + 1
dd_pentad_end = 5*(jpt-6*(month_valid-1)) + 5

plot_fields (fields_list = [prcp_fcst_pct[ipt,ipct-1,:,:],prcp_target_pct[jpt,ipct-1,:,:]],
          lon = lon_target,
          lat = lat_target,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f'Climatological {ipct}th percentile of rainfall accumulation for {month_valid_str} {dd_pentad_start}-{dd_pentad_end}',
          subtitle_list = [f'{system.upper()} forecast system',f'{ground_truth.upper()} observation product'],
          vmin = 0,
          vmax = 15,
          unit = 'mm')


# Let's pick a particular location and look at the cumulative distribution functions (CDFs) of the forecast and ground truth climatology, represented via the 1st through 99th percentile. Note that the CDFs are specific to time of the year, i.e. they will change based on the pentad selected above. 

# In[42]:


lon_exmpl = 36.82       # longitude of example location
lat_exmpl = -1.29       # latitude of example location
name_exmpl = 'Nairobi'  # name of the selected location

ix, iy = get_nearest_grid_index(lon_exmpl, lat_exmpl, lon_target, lat_target)

pct_level = np.arange(1,100)

plt.figure(figsize=(8,5.5))
ax = plt.subplot(1, 1, 1)
ax.plot(prcp_fcst_pct[ipt,:,iy,ix], pct_level/100, color='darkblue', label='forecast CDF')
ax.plot(prcp_target_pct[jpt,:,iy,ix], pct_level/100, color='orange', label='observation CDF')
ax.set_xticks([*range(0,26,5)])
ax.set_xticklabels([f'{i}mm' for i in range(0,26,5)], fontsize=12)
ax.set_xlabel('5-day rainfall accumulation', fontsize=12)
ax.set_ylabel('probability', fontsize=12)
ax.legend(loc='lower right', fontsize=14)
ax.set_title(f'Climatological CDFs at {name_exmpl} ({month_valid_str} {dd_pentad_start}-{dd_pentad_end})', fontsize=16)


# Let's try to use these CDFs now to quantile-map a given forecast value. Can you guess the result based on the above plot?

# In[43]:


fcst_new = np.array([5., 10., 15.])     # our hypothetical new forecast values

quantile_mapping(fcst_new, prcp_fcst_pct[ipt,:,iy,ix], prcp_target_pct[jpt,:,iy,ix])


# Now, let's load a real forecast and bias-correct it.

# In[44]:


#filename_fcst = f'{fcst_dir}{month_init_str.capitalize()}{year_fcst}/Rscripts/precip/c3S/{system}ForecastV51.grib'
#data_load = xr.open_dataset(filename_fcst, engine='cfgrib')

filename_fcst = f'{data_dir}{system}ForecastV51.nc4'

data_load = xr.open_dataset(filename_fcst, engine='netcdf4')
data_subset = data_load.sel(latitude=slice(lat_bounds[1],lat_bounds[0]), longitude=slice(lon_bounds[0],lon_bounds[1]))
lon_fcst = data_subset.longitude.values
lat_fcst = data_subset.latitude.values[::-1]
#prcp_fcst = data_subset.tp.values[:,:,::-1,:]
prcp_fcst = data_subset.precipitation.values[:,:,::-1,:]
data_load.close()

nmbs, nlts, nlatf, nlonf = prcp_fcst.shape


# In[45]:


#filename_fcst = f'{fcst_dir}total_precipitation_ecmwf_51_2023_8.nc4'

#data_load = xr.open_dataset(filename_fcst, engine='netcdf4')
#data_subset = data_load.sel(g0_lat_2=slice(lat_bounds[1],lat_bounds[0]), g0_lon_3=slice(lon_bounds[0],lon_bounds[1]))
#lon_fcst = data_subset.g0_lon_3.values
#lat_fcst = data_subset.g0_lat_2.values[::-1]
#prcp_fcst = data_subset.TP_GDS0_SFC.values[:,:,::-1,:]
#data_load.close()

#nmbs, nlts, nlatf, nlonf = prcp_fcst.shape


# The raw data come in the form of cumulative rainfall amounts in $m/m^2$. We convert this to $mm/m^2$ and extract pentad accumulations.

# In[46]:


prcp_fcst_pentad = calculate_pentad_accumulations(prcp_fcst, year_fcst, month_init)


# Let's plot a few pentad forcasts for a given ensemble member and a given month:

# In[47]:


imb = 0    # ensemble member
ilm = 3    # forecast lead time in months (0 corresponds to initialization month)

pentad_selection = [3,4,5,6]   # pentads within that month to be plotted (number between 1 and 6)

month_valid = month_init + ilm
month_valid_str = month_names[month_valid].capitalize()

plot_fields (fields_list = [prcp_fcst_pentad[imb,6*ilm+ipt-1,:,:] for ipt in pentad_selection],
          lon = lon_fcst,
          lat = lat_fcst,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f'Rainfall accumulation forecast for member {imb} ({month_valid_str})',
          subtitle_list = [f'pentad {ipt}' for ipt in pentad_selection],
          vmin = 0,
          vmax = 15,
          unit = 'mm')


# Before bias-correction, we need to interpolate these forecasts to the target grid:

# In[48]:


prcp_fcst_pentad_itp = interpolate_forecasts(prcp_fcst_pentad, lat_fcst, lon_fcst, lat_target, lon_target)


# Here's the forecasts from above, now interpolated:

# In[49]:


plot_fields (fields_list = [prcp_fcst_pentad_itp[imb,6*ilm+ipt-1,:,:] for ipt in pentad_selection],
          lon = lon_target,
          lat = lat_target,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f'Interpolated rainfall accumulation forecast for member {imb} ({month_valid_str})',
          subtitle_list = [f'pentad {ipt}' for ipt in pentad_selection],
          vmin = 0,
          vmax = 15,
          unit = 'mm')


# Now, we are ready to apply the quantile mapping function to bias-correct the forecasts for the selected pentads:

# In[50]:


prcp_fcst_pentad_bc = np.full((nmbs,len(pentad_selection),nlat,nlon), np.nan, dtype=np.float32)

for ix in range(nlon):
    for iy in range(nlat):
        if np.any(np.isnan(prcp_target_pct[:,:,iy,ix])):
            continue
        for i in range(len(pentad_selection)):
            ipt = 6*ilm+pentad_selection[i]-1
            jpt = (6*(month_init-1) + ipt) % 72
            fcst_new = prcp_fcst_pentad_itp[:,ipt,iy,ix]
            prcp_fcst_pentad_bc[:,i,iy,ix] = quantile_mapping(fcst_new, prcp_fcst_pct[ipt,:,iy,ix], prcp_target_pct[jpt,:,iy,ix])


# And again we plot the result:

# In[51]:


plot_fields (fields_list = [prcp_fcst_pentad_bc[imb,i,:,:] for i in range(len(pentad_selection))],
          lon = lon_target,
          lat = lat_target,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f'Bias-corrected rainfall accumulation forecast for member {imb} ({month_valid_str})',
          subtitle_list = [f'pentad {ipt}' for ipt in pentad_selection],
          vmin = 0,
          vmax = 15,
          unit = 'mm')


# In[ ]:




