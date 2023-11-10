#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np                # library for mathematical operations with arrays
import pandas as pd               # library for data frames; includes useful functions for date arithmetic
import xarray as xr               # library for arrays, especially tailored to weather data
import matplotlib.pyplot as plt   # library for basic plotting

from plotting import plot_fields                    # function for visualizing spatial data in a map
from miscellaneous import get_nearest_grid_index    # helper function to find the grid indices for a selected location
from miscellaneous import find_onset_day            # function to identify the onset day based on time series of threshold exceedances
from miscellaneous import mark_wet_spells           # helper function to mark wet spells within a rainfall time series
from miscellaneous import mark_dry_spells           # helper function to mark 7-day dry spells within a rainfall time series
from miscellaneous import interpolate_forecasts     # function for bilinear interpolation of forecasts


# In[2]:


fcst_dir = '/home/michael/bigdisk2/pro/SFE/Systems_daily_nc/'
#fcst_dir = '/home/ghacof/COF65/gcm/'
data_dir = '/home/michael/Desktop/FTW2023/'


# We set a few parameters that are fixed for this exercise but can vary in general:

# In[39]:


season = 'ond'          # season for which rainy season onset is considered
month_start = 8         # month in which we start the onset date search for this season
day_start = 15          # day within this month when the onset date search starts
nwks = 22               # length (number of weeks) of the onset search window

ndts = 7*nwks+23        # number of lead time days needed to calculate an onset date within the chosen search window

month_names = {1:'jan', 2:'feb', 3:'mar', 4:'apr', 5:'may', 6:'jun', 7:'jul', 8:'aug', 9:'sep', 10:'oct', 11:'nov', 12:'dec'}


# Let's load some daily CHIRPS data so we can experiment with the onset identification:

# In[4]:


filename_chirps = f'{data_dir}chirps_example_{season}_2017-2022.nc'

data_load = xr.open_dataset(filename_chirps, engine='netcdf4')
lon_chirps = data_load.longitude.values
lat_chirps = data_load.latitude.values
prcp_chirps_daily = data_load.precipitation.values
data_load.close()


# We need 1-day and 3-day accumulations during the time period specified above:

# In[5]:


prcp_chirps_1d = prcp_chirps_daily[:,:ndts,:,:]
prcp_chirps_3d = prcp_chirps_daily[:,:ndts,:,:] + prcp_chirps_daily[:,1:(ndts+1),:,:] + prcp_chirps_daily[:,2:(ndts+2),:,:]


# As for the bias-correction script, you can choose your favorite location within GHA. Since we only prepared OND data, your location should have an expected rainy season onset in OND though for the example to be meaningful. You can also choose a year between 2017 and 2022.

# In[6]:


lon_exmpl = 36.82       # longitude of example location
lat_exmpl = -1.29       # latitude of example location
name_exmpl = 'Nairobi'  # name of the selected location
year_exmpl = 2020

ix, iy = get_nearest_grid_index(lon_exmpl, lat_exmpl, lon_chirps, lat_chirps)

iyr = {2017:0, 2018:1, 2019:2, 2020:3, 2021:4, 2022:5}[year_exmpl]

date_start = f'{year_exmpl}-{month_start}-{day_start}'

precp_series = pd.Series(prcp_chirps_1d[iyr,:,iy,ix], index=pd.date_range(start=date_start, periods=ndts))

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12,8))
precp_series[:f'{year_exmpl}-12-31'].plot(style='.', color='darkblue', ax=ax)
ax.set_ylabel('daily rainfall amount (mm)')
ax.set_title(f'Daily CHIRPS rainfall amounts at {name_exmpl} in OND {year_exmpl}', fontsize=16)


# Now, we set some key parameters of the rainy season onset definition:

# In[7]:


thr_wet = 20.      # the rainfall amount within 3 days required to consider these days a 'wet spell'
thr_dry = 1.       # the daily rainfall amount below which the day will be considered a 'dry day'


# We highlight the corresponding 'wet spells' and 'dry spells' of at least 7 days duration, and calculate the date identified as the rainy season onset according to these criteria. Try a few different values to see how it affects the onset date.

# In[8]:


exc1_ind = np.greater(prcp_chirps_1d[iyr,:,iy,ix], thr_dry)     # days with more than {thr_dry}mm rainfall
exc20_ind = np.greater(prcp_chirps_3d[iyr,:,iy,ix], thr_wet)    # 3-day periods with more than {thr_wet}mm rainfall

onset_day = find_onset_day(exc1_ind, exc20_ind)

precp_series_wet = precp_series.where(mark_wet_spells(exc1_ind, exc20_ind),np.nan)
precp_series_dry = precp_series.where(mark_dry_spells(exc1_ind),np.nan)
precp_series_onset = precp_series.where(np.arange(1,ndts+1)==onset_day,np.nan)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12,8))
precp_series[:f'{year_exmpl}-12-31'].plot(style='.', color='darkblue', ax=ax)
precp_series_onset[:f'{year_exmpl}-12-31'].plot(style='o', color='blueviolet', label='onset date', ax=ax)
precp_series_wet[:f'{year_exmpl}-12-31'].plot(style='.', color='orange', label='wet spell', ax=ax)
precp_series_dry[:f'{year_exmpl}-12-31'].plot(style='.', color='darkred', label='dry spell', ax=ax)
ax.legend(fontsize=14)
ax.set_ylabel('daily rainfall amount (mm)')
ax.set_title(f'OND {year_exmpl} onset date for {name_exmpl}', fontsize=16)


# Now, let's load a real forecast and derive an ensemble of onset dates from it. The following parameters are fixed for this exercise but can be changed in general:

# In[9]:


year_fcst = 2023        # year for which the bias-corrected forecasts should be generated

lon_bounds = [20, 53]   # longitude range of the domain of interest
lat_bounds = [-15, 23]  # latitude range of the domain of interest

system = 'ecmwf'        # forecast system on which the rainy season onset forecasts will be based


# In[ ]:


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


# In[10]:


#filename_fcst = f'{fcst_dir}total_precipitation_{system}_51_{year_fcst}_{month_start}.nc4'

#data_load = xr.open_dataset(filename_fcst, engine='netcdf4')
#data_subset = data_load.sel(g0_lat_2=slice(lat_bounds[1],lat_bounds[0]), g0_lon_3=slice(lon_bounds[0],lon_bounds[1]))
#lon_fcst = data_subset.g0_lon_3.values
#lat_fcst = data_subset.g0_lat_2.values[::-1]
#prcp_fcst = data_subset.TP_GDS0_SFC.values[:,:,::-1,:]
#data_load.close()

#nmbs, nlts, nlatf, nlonf = prcp_fcst.shape


# We calculate daily and 3-day rainfall amounts from cumulative precipitation amounts given in the raw forecast file, and as for the bias-correction exercise we interpolate to the target grid:

# In[11]:


# Calculate forecasts of daily and 3-day precipitation accumulations and convert to mm/m^2
istrt = day_start-1
prcp_fcst_1d = 1000*(prcp_fcst[:,istrt:(istrt+ndts),:,:]-prcp_fcst[:,(istrt-1):(istrt+ndts-1),:,:])
prcp_fcst_3d = 1000*(prcp_fcst[:,(istrt+2):(istrt+ndts+2),:,:]-prcp_fcst[:,(istrt-1):(istrt+ndts-1),:,:])

prcp_fcst_1d_itp = interpolate_forecasts(prcp_fcst_1d, lat_fcst, lon_fcst, lat_chirps, lon_chirps)
prcp_fcst_3d_itp = interpolate_forecasts(prcp_fcst_3d, lat_fcst, lon_fcst, lat_chirps, lon_chirps)


# Let's plot some of the interpolated, daily precipitation forecasts:

# In[12]:


imb = 0                           # ensemble member
lead_selection = [10,20,30,40]    # forecast lead times (days after the forecast initialization date)

plot_fields (fields_list = [prcp_fcst_1d_itp[imb,ilt,:,:] for ilt in lead_selection],
          lon = lon_chirps,
          lat = lat_chirps,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f'Interpolated daily rainfall forecast for member {imb}',
          subtitle_list = [(pd.Timestamp(f'{year_fcst}-{month_start}-{day_start}')+pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in lead_selection],
          vmin = 0,
          vmax = 25,
          unit = 'mm')


# We have seen that these forecasts often have systematic biases, but here, instead of bias-correcting the forecasts, we adjust the thresholds of the onset definition such that the forecasts exceed {thr_dry}mm per day and {thr_wet}mm in 3 days with the same climatological frequency as the CHIRPS data.

# For the choices thr_dry=1mm and thr_wet=20mm, we have pre-calculated these adjusted thresholds and can load them from disk:

# In[28]:


f1 = np.load(f'{data_dir}thresh_adj_nocv_{system}_0p25.npz')
thresh_adj_1d_1mm = f1['thresh_1d']
thresh_adj_3d_20mm = f1['thresh_3d']
f1.close()

f1 = np.load(f'{data_dir}onset_chirps_0p25_{season}.npz')
mask = f1['mask']
onset_day_chirps = f1['onset_day']
f1.close()

nxy = np.sum(mask)


# To save disk space, there thresholds were only calculated on a selected set of grid points over GHA (where we expect an OND onset), so we have to expand the values into a 2D array before plotting them:

# In[14]:


ilt = 93    # lead time to be plotted (recall that the adjusted thresholds vary with lead time)

date_str = (pd.Timestamp(f'{year_fcst}-{month_start}-{day_start}')+pd.Timedelta(days=ilt)).strftime("%Y-%m-%d")

nlat = len(lat_chirps)
nlon = len(lon_chirps)

field1 = np.full(nlat*nlon, np.nan, dtype=np.float32)
field1[mask.flatten()] = thresh_adj_1d_1mm[ilt,:]

field2 = np.full(nlat*nlon, np.nan, dtype=np.float32)
field2[mask.flatten()] = thresh_adj_3d_20mm[ilt,:]

plot_fields (fields_list = [field1.reshape(nlat,nlon), field2.reshape(nlat,nlon)],
          lon = lon_chirps,
          lat = lat_chirps,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f'Adjusted threshold values for {date_str}',
          subtitle_list = ['threshold for 1 mm / day','threshold for  20 mm / 3 days'],
          vmin = [0,0],
          vmax = [5,50],
          unit = 'mm')


# It may be easier to see the magnitude of adjustment if we depict the factor that needs to be multiplied to the original thresholds:

# In[15]:


plot_fields (fields_list = [100*(field1.reshape(nlat,nlon)-1), 100*((field2.reshape(nlat,nlon)/20)-1)],
          lon = lon_chirps,
          lat = lat_chirps,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f'Increase or decrease (%) relative to original threshold for {date_str}',
          subtitle_list = ['threshold for 1 mm / day','threshold for  20 mm / 3 days'],
          vmin = -100,
          vmax = 100,
          cmap = 'BrBG',
          unit = '%')


# Now it's time to generate our OND onset forecast:

# In[70]:


onset_day_fcst = np.full((nxy,nmbs), -1, dtype=int)

for imb in range(nmbs):
    prcp_fcst_1d_mb = prcp_fcst_1d_itp[imb,:,:,:].reshape((ndts,-1))[:,mask.flatten()]
    prcp_fcst_3d_mb = prcp_fcst_3d_itp[imb,:,:,:].reshape((ndts,-1))[:,mask.flatten()]
    exc1d_ind = np.greater(prcp_fcst_1d_mb, thresh_adj_1d_1mm)
    exc3d_ind = np.greater(prcp_fcst_3d_mb, thresh_adj_3d_20mm)
    for ixy in range(nxy):
        onset_day_fcst[ixy,imb] = find_onset_day(exc1d_ind[:,ixy], exc3d_ind[:,ixy])

onset_day_fcst_grid = np.full((nmbs,nlat*nlon), np.nan, dtype=np.float32)
for imb in range(nmbs):
    onset_day_fcst_grid[imb,mask.flatten()] = onset_day_fcst[:,imb]

onset_day_fcst_grid = onset_day_fcst_grid.reshape((nmbs,nlat,nlon))
onset_week_fcst_grid = np.where(onset_day_fcst_grid<0, np.nan, 1+(onset_day_fcst_grid-1)//7)


# Historic rainy season onset dates have already been loaded, and just need to be reformatted to the full grid:

# In[71]:


nyrs_clim = onset_day_chirps.shape[0]

onset_day_chirps_grid = np.full((nyrs_clim,nlat*nlon), np.nan, dtype=np.float32)
for iyr in range(nyrs_clim):
    onset_day_chirps_grid[iyr,mask.flatten()] = onset_day_chirps[iyr,:]

onset_day_chirps_grid = onset_day_chirps_grid.reshape((nyrs_clim,nlat,nlon))
onset_week_chirps_grid = np.where(onset_day_chirps_grid<0, np.nan, 1+(onset_day_chirps_grid-1)//7)


# The above calculation gives us an ensemble of rainy season onset dates at each gridpoint within the region with expected OND onset. Let's visualize a few selected ensemble members, with onset dates aggregated to weeks after the start date of the search window:

# In[72]:


member_selection = [0,1,2]    # selection of ensemble members to be plotted

plot_fields (fields_list = [onset_week_fcst_grid[imb,:,:] for imb in member_selection],
          lon = lon_chirps,
          lat = lat_chirps,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f'{system.upper()}-based forecast of rainy season onset date in {season.upper()} {year_fcst}',
          subtitle_list = [f'member {imb}' for imb in member_selection],
          cmap = 'tab20c',
          vmin = 0,
          vmax = 20,
          unit = f'weeks after {day_start} {month_names[month_start].capitalize()}')


# Let's plot some climatological onset dates for comparison:

# In[74]:


year_selection = [2014,2015,2016]    # selection of years (between 1993 and 2016) to be plotted

plot_fields (fields_list = [onset_week_chirps_grid[iyr-1993,:,:] for iyr in year_selection],
          lon = lon_chirps,
          lat = lat_chirps,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f'Historic (CHIRPS-based) {season.upper()} rainy season onset dates',
          subtitle_list = year_selection,
          cmap = 'tab20c',
          vmin = 0,
          vmax = 20,
          unit = f'weeks after {day_start} {month_names[month_start].capitalize()}')


# At a selected location, we can compare the cumulative distribution function (CDF) of historic and predicted rainy season onset dates:

# In[77]:


lon_exmpl = 36.82       # longitude of example location
lat_exmpl = -1.29       # latitude of example location
name_exmpl = 'Nairobi'  # name of the selected location

ix, iy = get_nearest_grid_index(lon_exmpl, lat_exmpl, lon_chirps, lat_chirps)

onset_day_exmpl_clm = np.where(onset_day_chirps_grid[:,iy,ix]==-1,1e5,onset_day_chirps_grid[:,iy,ix])
onset_day_exmpl_fcst = np.where(onset_day_fcst_grid[:,iy,ix]==-1,1e5,onset_day_fcst_grid[:,iy,ix])

days_grid = np.arange(1,154)

cdf_clm = np.mean(np.less_equal.outer(onset_day_exmpl_clm, days_grid), axis=0)
cdf_fcst = np.mean(np.less_equal.outer(onset_day_exmpl_fcst, days_grid), axis=0)

fig, ax = plt.subplots(1, 1, figsize=(10.,7))
ax.plot(days_grid, cdf_clm, c='b', label='Climatology')
ax.plot(days_grid, cdf_fcst, c='r', label=f'{system.upper()} forecast {year_fcst}')
ax.hlines(1.0, 0, 160, colors='k', linestyles='dashed')
ax.set_ylim(0.5, 153.05)
ax.set_ylim(0.0, 1.05)
ax.set_xticks([1,18,32,48,62,79,93,109,123,140,154])
ax.set_xticklabels([' ','Sep 1',' ','Oct 1',' ','Nov 1',' ','Dec 1',' ','Jan 1',' '])
ax.set_title(f'CDF for OND rainy season onset date in {name_exmpl}', fontsize=16)
ax.legend(loc=[0.55,0.16], fontsize=12)


# Finally, let's calculate the forecast and climatological probability that rainy season onset occurs before or at a given date:

# In[87]:


date_exmpl = '2023-11-20'

idt = [ilt for ilt in range(120) if (pd.Timestamp(f'{year_fcst}-{month_start}-{day_start}')+pd.Timedelta(days=ilt)).strftime("%Y-%m-%d")==date_exmpl][0]

prob_exmpl_clm = np.mean(np.logical_and(np.less_equal(onset_day_chirps_grid,idt),np.greater_equal(onset_day_chirps_grid,0)), axis=0)
prob_exmpl_fcst = np.mean(np.logical_and(np.less_equal(onset_day_fcst_grid,idt),np.greater_equal(onset_day_fcst_grid,0)), axis=0)

plot_fields (fields_list = [prob_exmpl_fcst, prob_exmpl_clm],
          lon = lon_chirps,
          lat = lat_chirps,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f'Probability for a rainy season onset before {date_exmpl}',
          subtitle_list = [f'{system.upper()} forecast','CHIRPS climatology'],
          cmap = 'Blues',
          vmin = 0,
          vmax = 1,
          unit = 'probability')


# Note: a low probability in these plots can either be due to a high probability of a later OR a high probability of a failed onset at the respective grid point. Do you have a better idea to visualize this? Then try changing the treatment of the 'onset_day = -1' category.

# In[ ]:




