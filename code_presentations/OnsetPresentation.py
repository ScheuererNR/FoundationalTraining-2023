
import numpy as np
import pandas as pd
import xarray as xr

import datetime
import seasonal.interface as sfe

import matplotlib.pyplot as plt

from datetime import date, datetime, timedelta

plt.ion()


# Dirty trick to deal with numpy deprecation issues
np.bool = np.bool_
np.int = np.int_


chirps_dir = '/media/datadisk/pro/SFE/CHIRPS_daily_nc/'           # directory where the CHIRPS data are stored
onset_dir = '/home/michael/data/CONFER/RainOnset/QM-CHIRPS/'      # directory where csv output files should be stored

start_year = 2017         # first year for which we have both forecast and observation data
end_year = 2022           # last year for which we have both forecast and observation data

lon_bounds = [20, 53]     # longitude range of the domain of interest
lat_bounds = [-14, 25]    # latitude range of the domain of interest

season = 'ond'

res = 0.25
res_str = '0p25'

start_month = 8           # forecast initialization month; here: August for OND rain onset
start_day = 15            # day within this month when the onset date search starts
nwks = 22                 # length (number of weeks) of the time period during which the onset date is sought (22 for OND) 

years = [iyr for iyr in range(start_year, end_year+1)]
nyrs = len(years)

ndts = 7*nwks+23      # number of days needed to calculate an onset date within the next <nwks> weeks



##  Load CHIRPS data and calculate 1-day and 3-day precipitation accumulations

dates_chirps = np.chararray((nyrs,ndts+2), 10)

for iyr in range(nyrs):
    start_date = datetime.strptime(str(years[iyr])+'-'+format(start_month,'02d')+'-15', "%Y-%m-%d")
    for idts in range(ndts+2):
        dates_chirps[iyr,idts] = (start_date+timedelta(days=idts)).strftime("%Y-%m-%d")

lat_chirps, lon_chirps, prcp_chirps_daily = sfe.load_chirps_data (data_dir = chirps_dir,
                                                          dates = dates_chirps,
                                                          aggregation = 'daily',
                                                          lon_bounds = lon_bounds,
                                                          lat_bounds = lat_bounds,
                                                          res = res)

ds = xr.Dataset(
        {
            'precipitation': (['year_start','day','latitude','longitude'], prcp_chirps_daily),
        },
        coords = {
            'year_start': years,
            'time': [*range(1,ndts+3)],
            'latitude': lat_chirps,
            'longitude': lon_chirps,
        },                
    )

ds.to_netcdf(f'{onset_dir}chirps_example_ond_{start_year}-{end_year}.nc')



f1 = np.load('/home/michael/data/CONFER/CHIRPS_onset_mask_'+res_str+'.npz')
mask = f1['mask_'+season]
f1.close()

coords_gha = np.array(np.meshgrid(lon_chirps,lat_chirps)).reshape((2,-1))[:,mask.flatten()].T
nxy = coords_gha.shape[0]


prcp_chirps_daily_gha = prcp_chirps_daily.reshape((nyrs,ndts+2,-1))[:,:,mask.flatten()]

prcp_chirps_1d = prcp_chirps_daily_gha[:,:ndts,:]
prcp_chirps_3d = prcp_chirps_daily_gha[:,:ndts,:] + prcp_chirps_daily_gha[:,1:(ndts+1),:] + prcp_chirps_daily_gha[:,2:(ndts+2),:]





##  Calculate rain onset date for CHIRPS data

def rainseason_onset_day(exc1_ind, exc20_ind):
    n = len(exc1_ind)
    wet_spell = np.logical_and(exc20_ind[:(n-2)], exc1_ind[:(n-2)])
    ind_lag7d = np.expand_dims(np.arange(7),0)+np.expand_dims(np.arange(n-6),0).T
    dry_spell = np.all(~exc1_ind[ind_lag7d], axis=1)
    ind_lag15d = np.expand_dims(np.arange(15),0)+np.expand_dims(np.arange(n-23),0).T
    onset = np.logical_and(wet_spell[:(n-23)], ~np.any(dry_spell[3:][ind_lag15d], axis=1))
    onset_day = -1 if np.all(~onset) else np.nonzero(onset)[0][0]+1
    return onset_day



lon_exmpl = 36.875  # Nairobi, res = 0.25
lat_exmpl = -1.375

lon_exmpl = 39.65  # Garissa, res = 0.25
lat_exmpl = -0.47

lon_exmpl = 41.84  # Mandera, res = 0.25
lat_exmpl = 3.93


lon_exmpl_gpt = np.round((lon_exmpl-0.125)*4)/4 + 0.125
lat_exmpl_gpt = np.round((lat_exmpl-0.125)*4)/4 + 0.125

ixy = np.where(np.logical_and(coords_gha[:,0].round(3)==lon_exmpl_gpt, coords_gha[:,1].round(3)==lat_exmpl_gpt))[0][0]

iyr = 3

thr_wet = 20.
thr_dry = 1.

exc1_ind = np.greater(prcp_chirps_1d[iyr,:,ixy], thr_dry)
exc20_ind = np.greater(prcp_chirps_3d[iyr,:,ixy], thr_wet)

onset_day = rainseason_onset_day(exc1_ind, exc20_ind)

precp_series = pd.Series(prcp_chirps_1d[iyr,:,ixy], index=pd.date_range(start=dates_chirps[iyr,:ndts].ljust(10).astype(str)[0], periods=ndts))
wet_spell_day1 = np.logical_and(exc20_ind, exc1_ind)
ind_lag7d = np.expand_dims(np.arange(7),0)+np.expand_dims(np.arange(ndts-6),0).T
dry_spell_day1 = np.append(np.all(~exc1_ind[ind_lag7d], axis=1), np.zeros(6, dtype=bool))

wet_spell = np.zeros(ndts, dtype=bool)
dry_spell = np.zeros(ndts, dtype=bool)

for idt in range(ndts-2):
    if wet_spell_day1[idt]:
        wet_spell[idt:(idt+3)] = True
    if dry_spell_day1[idt]:
        dry_spell[idt:(idt+7)] = True

precp_series_wet = precp_series.where(wet_spell,np.nan)
precp_series_dry = precp_series.where(dry_spell,np.nan)
precp_series_onset = precp_series.where(np.arange(1,ndts+1)==onset_day,np.nan)


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))
precp_series[:f'{years[iyr]}-12-31'].plot(style='.', color='darkblue', ax=ax)
precp_series_onset[:f'{years[iyr]}-12-31'].plot(style='o', color='blueviolet', label='onset date', ax=ax)
precp_series_wet[:f'{years[iyr]}-12-31'].plot(style='.', color='orange', label='wet spell', ax=ax)
precp_series_dry[:f'{years[iyr]}-12-31'].plot(style='.', color='darkred', label='dry spell', ax=ax)
ax.legend()
ax.set_ylabel('daily rainfall amount (mm)')
ax.set_title(f'OND {years[iyr]} onset date for Mandera', fontsize=12)
plt.tight_layout()
plt.savefig(f'/home/michael/Desktop/tmp/onset-plots/onset-Mandera_{years[iyr]}.png')





##  Maps illustrating the adjusted thresholds

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from mpl_toolkits.axes_grid1 import ImageGrid #, make_axes_locatable


def get_xticks(x_extent, inc = 1):
    x_inc = np.arange(-180,180,inc)
    return(x_inc[np.where(np.logical_and(x_inc >= x_extent[0], x_inc <= x_extent[1]))])

def get_yticks(y_extent, inc = 1):
    y_inc = np.arange(-90,90,inc)
    return(y_inc[np.where(np.logical_and(y_inc >= y_extent[0], y_inc <= y_extent[1]))])


res = 0.25
res_str = '0p25'


##  Load CHIRPS grid coordinates and GHA mask

f1 = np.load('/home/michael/data/CONFER/CHIRPS_mask_'+res_str+'.npz')
lon = f1['lon']
lat = f1['lat']
f1.close()

nlon = len(lon)
nlat = len(lat)

season = 'ond'
system = 'ecmwf'

lon_bounds = [20, 53]     # longitude range of the domain of interest
lat_bounds = [-14, 25]    # latitude range of the domain of interest

img_extent = lon_bounds + lat_bounds

unit = 'mm'
cmap = cm.get_cmap('BrBG', 11)

r = abs(lon[1]-lon[0])
lons_mat, lats_mat = np.meshgrid(lon, lat)
lons_matplot = np.hstack((lons_mat - r/2, lons_mat[:,[-1]] + r/2))
lons_matplot = np.vstack((lons_matplot, lons_matplot[[-1],:]))
lats_matplot = np.hstack((lats_mat, lats_mat[:,[-1]]))
lats_matplot = np.vstack((lats_matplot - r/2, lats_matplot[[-1],:] + r/2))     # assumes latitudes in ascending order

f1 = np.load(onset_dir+"onset_chirps_"+res_str+"_"+season+".npz")
mask = f1['mask']
f1.close()

f2 = np.load(onset_dir+'thresh-'+season+'/thresh_adj_'+system+'_'+res_str+'.npz')
#f2 = np.load(onset_dir+'thresh-'+season+'/thresh_adj_nocv_'+system+'_'+res_str+'.npz')
coords_gha = f2['coords']
thresh_adj_1d_1mm = f2['thresh_1d']
thresh_adj_3d_20mm = f2['thresh_3d']
f2.close()

field1 = np.full(nlat*nlon, np.nan, dtype=np.float32)
field1[mask.flatten()] = thresh_adj_1d_1mm[-1,107,:]

field2 = np.full(nlat*nlon, np.nan, dtype=np.float32)
field2[mask.flatten()] = thresh_adj_3d_20mm[-1,107,:]

fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(5,5))
cmesh = ax.pcolormesh(lons_matplot, lats_matplot, np.log10(field1.reshape(nlat,nlon)), cmap=cmap, vmin=-1.1, vmax=1.1)
ax.set_extent(img_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5)
cbar = plt.colorbar(cmesh, orientation='vertical', extend='both')
cbar.set_ticks([-1,np.log10(0.5),0,np.log10(2),1])
cbar.set_ticklabels(['0.1 mm','0.5 mm','1 mm','2 mm','10 mm'])
fig.suptitle('Adjusted'+' 1 mm threshold'+' for ECMWF, Nov 16', fontsize=14)
fig.canvas.draw()
plt.tight_layout()
plt.savefig(f'/home/michael/Desktop/tmp/onset-plots/threshold-1mm.png')


fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(5,5))
cmesh = ax.pcolormesh(lons_matplot, lats_matplot, np.log10(field2.reshape(nlat,nlon)/20), cmap=cmap, vmin=-0.65, vmax=0.65)
ax.set_extent(img_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5)
cbar = plt.colorbar(cmesh, orientation='vertical', extend='both')
cbar.set_ticks([np.log10(0.25),np.log10(0.5),0,np.log10(2),np.log10(4)])
cbar.set_ticklabels(['5 mm','10 mm','20 mm','40 mm','80 mm'])
fig.suptitle('Adjusted'+' 20 mm threshold'+' for ECMWF, Nov 16', fontsize=14)
fig.canvas.draw()
plt.tight_layout()
plt.savefig(f'/home/michael/Desktop/tmp/onset-plots/threshold-20mm.png')




##  Plot onset ensemble forecasts

import plotting.mapplot as sfep
import matplotlib.colors as pltc

from colorspace import qualitative_hcl
from matplotlib.colors import ListedColormap

# Define new colormap
a = qualitative_hcl(h=[60,360], c=95, l=90)(5)
b = qualitative_hcl(h=[60,360], c=80, l=70)(5)
c = qualitative_hcl(h=[60,360], c=65, l=50)(5)
d = qualitative_hcl(h=[60,360], c=50, l=30)(5)

onsetcol = np.ones((23,4))
onsetcol[0,:3] = np.array([242/256, 242/256, 242/256])
onsetcol[-2,:3] = np.array([162/256, 162/256, 162/256])
onsetcol[-1,:3] = np.array([102/256, 102/256, 102/256])

for i in range(5):
    onsetcol[1+0+4*i,:3] = pltc.to_rgb(a[i])
    onsetcol[1+1+4*i,:3] = pltc.to_rgb(b[i])
    onsetcol[1+2+4*i,:3] = pltc.to_rgb(c[i])
    onsetcol[1+3+4*i,:3] = pltc.to_rgb(d[i])

nwks = 22
#newcmp = ListedColormap(np.concatenate((onsetcol[(nwks-10):,:],onsetcol[:(nwks-10),:]), axis=0))
newcmp = ListedColormap(onsetcol)
labels = [(datetime.strptime('2000-08-15', "%Y-%m-%d")+timedelta(weeks=x)).strftime("%b %d") for x in range(nwks)]

labels.insert(0,'no onset')


# Load ECMWF onset date foracsts for 2017-2022 period and convert to onset week
f4 = np.load(onset_dir+'onsetfcst-'+season+'/onsetdate_ecmwf_2017-2022_'+res_str+'.npz')
onset_day_ecmwf = f4['onset_day_fcst']
f4.close()


iyr = 2

fields = np.full((6,nlat*nlon), np.nan, dtype=np.float32)
for imb in range(6):
    fields[imb,:][mask.flatten()] = 1 + (onset_day_ecmwf[iyr,:,imb]-1) // 7

fields.shape = (6,nlat,nlon)

fig, axs = plt.subplots(nrows=2, ncols=3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(18,12))
for k in range(6):
    j = k % 3
    i = k // 3
    cmesh = axs[i,j].pcolormesh(lons_matplot, lats_matplot, fields[k,:,:]-0.5, cmap=newcmp, vmin=0.0, vmax=nwks)
    axs[i,j].set_extent(lon_bounds+lat_bounds, crs=ccrs.PlateCarree())
    axs[i,j].set_yticks(get_yticks(lat_bounds,2), crs=ccrs.PlateCarree())
    axs[i,j].yaxis.set_major_formatter(LatitudeFormatter()) 
    axs[i,j].set_xticks(get_xticks(lon_bounds,4), crs=ccrs.PlateCarree())
    axs[i,j].xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    axs[i,j].add_feature(cfeature.COASTLINE)
    axs[i,j].add_feature(cfeature.BORDERS, linestyle='-', alpha=.5)
    axs[i,j].set_title(f'member {k+1}', fontsize=14)

fig.subplots_adjust(bottom=0.06, top=0.98, left=0.01, right=0.99, wspace=0.01, hspace=0.11)
cbar_ax = fig.add_axes([0.94, 0.2, 0.02, 0.6])
cbar = plt.colorbar(cmesh, cax=cbar_ax, orientation='vertical')
cbar.set_ticks([*range(nwks+1)])
cbar.set_ticklabels(labels)

fig.suptitle(f'ECMWF-based forecast of rainy season onset date in OND {[*range(2017,2023)][iyr]}', fontsize=16)
fig.canvas.draw()
plt.tight_layout()
plt.savefig(f'/home/michael/Desktop/tmp/onset-plots/onset-ensemble.png')



##  Plot onset CDF at Nairobi

lon_exmpl = 36.875  # Nairobi, res = 0.25
lat_exmpl = -1.375

lon_exmpl_gpt = np.round((lon_exmpl-0.125)*4)/4 + 0.125
lat_exmpl_gpt = np.round((lat_exmpl-0.125)*4)/4 + 0.125

ixy = np.where(np.logical_and(coords_gha[:,0].round(3)==lon_exmpl_gpt, coords_gha[:,1].round(3)==lat_exmpl_gpt))[0][0]

f1 = np.load(onset_dir+"onset_chirps_"+res_str+"_"+season+".npz")
onset_day_chirps = f1['onset_day']
f1.close()

onset_dates_clm = onset_day_chirps[:,ixy]
onset_dates_clm[onset_dates_clm==-1] = 1e5

onset_dates_fcst = onset_day_ecmwf[iyr,ixy,:]
onset_dates_fcst[onset_dates_fcst==-1] = 1e5

grid = np.arange(1,154)

cdf_clm = np.mean(np.less_equal.outer(onset_dates_clm, grid), axis=0)
cdf_fcst = np.mean(np.less_equal.outer(onset_dates_fcst, grid), axis=0)

fig, ax = plt.subplots(1, 1, figsize=(7.,4.5))
ax.plot(grid, cdf_clm, c='b', label='Climatology')
ax.plot(grid, cdf_fcst, c='r', label='ECMWF forecast '+str([*range(2017,2023)][iyr]))
ax.hlines(1.0, 0, 160, colors='k', linestyles='dashed')
#ax.fill_betweenx([0.,1.], x1=[grid[onset_week_median_start2]]*2, x2=[grid[onset_week_median_start2]+7]*2, alpha=.2, color='g')
ax.set_ylim(0.5, 153.05)
ax.set_ylim(0.0, 1.05)
ax.set_xticks([1,18,32,48,62,79,93,109,123,140,154])
ax.set_xticklabels([' ','Sep 1',' ','Oct 1',' ','Nov 1',' ','Dec 1',' ','Jan 1',' '])
ax.set_title('CDF for OND rainy season onset date in Nairobi', fontsize=16)
ax.legend(loc=[0.55,0.16], fontsize=12)
plt.tight_layout()
plt.savefig(f'/home/michael/Desktop/tmp/onset-plots/onset-cdf.png')



