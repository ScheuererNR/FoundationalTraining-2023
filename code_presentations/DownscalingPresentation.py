
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

import seasonal.interface as sfe
import plotting.mapplot as sfep

from netCDF4 import Dataset
from datetime import date, datetime, timedelta

# from cfgrib.xarray_to_grib import to_grib

from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline, interp1d


# Dirty trick to deal with netCDF4 deprecation issue
np.bool = np.bool_

plt.ion()


netcdf_dir = '/home/michael/bigdisk2/pro/SFE/Systems_daily_nc/'   # directory where the netcdf forecast files are stored
output_dir = '/home/michael/data/CONFER/Downscaling/'


start_year = 2000         # first year for which we have both forecast and observation data
end_year = 2021           # last year for which we have both forecast and observation data
year_fcst = 2023          # year for which the bias-corrected forecasts are sought


lon_bounds = [20, 53]     # longitude range of the domain of interest
lat_bounds = [-15, 23]    # latitude range of the domain of interest

years = [*range(start_year, end_year+1)]
nyrs = len(years)

nlm = 7                    # maximal forecast lead time in months
npts = 6*nlm               # number of pentads within the forecast lead range

system = 'ecmwf'           # forecast system to be used to predict rain onset
ground_truth = 'CHIRPS'    # data set against which the forecasts should be calibrated
res_str = '0p1'            # horizontal resolution to which forecasts should be downscaled

init_month = 8             # forecast initialization month

init_month_str = {1:'jan', 2:'feb', 3:'mar', 4:'apr', 5:'may', 6:'jun', 7:'jul', 8:'aug', 9:'sep', 10:'oct', 11:'nov', 12:'dec'}[init_month]




##  1. Load climatological percentiles of forecast and CHIRPS data

nc = Dataset(output_dir+'percentiles_'+system+'_'+res_str+'_'+str(start_year)+'_'+str(end_year)+'_'+init_month_str+'.nc')
prcp_fcst_pct = nc.variables["percentile"][:]
nc.close()

nc = Dataset(output_dir+'percentiles_'+ground_truth.lower()+'_'+res_str+'_'+str(start_year)+'_'+str(end_year)+'.nc')
lon_target = nc.variables["lon"][:]
lat_target = nc.variables["lat"][:]
prcp_target_pct = nc.variables["percentile"][:]
nc.close()

nlon = len(lon_target)
nlat = len(lat_target)

prcp_fcst_pct[prcp_fcst_pct<0.0] = 0.0            # just to be save ...
prcp_target_pct[prcp_target_pct<0.0] = 0.0




##  2. Load forecast data, calculate pentad precipitation amounts, and interpolate

lat_fcst, lon_fcst, dates_fcst, prcp_fcst_cum = sfe.load_cds_forecasts (variable = 'total_precipitation',
                                                          system = system,
                                                          system_nr = '51',
                                                          months = [init_month],
                                                          years = [year_fcst],
                                                          lon_bounds = lon_bounds,
                                                          lat_bounds = lat_bounds,
                                                          data_dir = netcdf_dir)

nyrs, nmts, nmbs, nlts, nlatf, nlonf = prcp_fcst_cum.shape


pentad_end_idx = np.zeros(npts, dtype=np.int32)

for imt in range(nlm):
    year_valid = str(year_fcst+(init_month+imt-1)//12)
    month_valid = format(1+(init_month+imt-1)%12,'02d')
    start_date_month = datetime.strptime(year_valid+'-'+month_valid+'-01', "%Y-%m-%d")
    for ipt in range(6):
        if imt == 0 and ipt == 0:
            continue
        end_date_prev_pentad = (start_date_month+timedelta(days=5*ipt-1)).strftime("%Y-%m-%d")
        pentad_end_idx[6*imt+ipt-1] = np.where(dates_fcst[0,0,:].astype(str)==end_date_prev_pentad)[0][0]

year_valid = str(year_fcst+(init_month+nlm-1)//12)
month_valid = format(1+(init_month+nlm-1)%12,'02d')
start_date_month = datetime.strptime(year_valid+'-'+month_valid+'-01', "%Y-%m-%d")
end_date_prev_pentad = (start_date_month+timedelta(days=-1)).strftime("%Y-%m-%d")
pentad_end_idx[-1] = np.where(dates_fcst[0,0,:].astype(str)==end_date_prev_pentad)[0][0]

prcp_fcst_pentad = np.full((nmbs,npts,nlatf,nlonf), np.nan, dtype=np.float32)

prcp_fcst_pentad[:,0,:,:] = 200.*prcp_fcst_cum[0,0,:,pentad_end_idx[0],:,:]
for ipt in range(1,npts):
    ilt0 = pentad_end_idx[ipt-1]
    ilt1 = pentad_end_idx[ipt]
    prcp_fcst_pentad[:,ipt,:,:] = np.maximum(0.,1000.*(prcp_fcst_cum[0,0,:,ilt1,:,:]-prcp_fcst_cum[0,0,:,ilt0,:,:])/(ilt1-ilt0))


prcp_fcst_pentad_itp = np.full((nmbs,npts,nlat,nlon), np.nan, dtype=np.float32)

for imb in range(nmbs):
    for ipt in range(npts):
        if np.all(np.isnan(prcp_fcst_pentad[imb,ipt,:,:])):
            continue
        itpfct_1d = RectBivariateSpline(lat_fcst, lon_fcst, prcp_fcst_pentad[imb,ipt,:,:], kx=1, ky=1, s=0)
        prcp_fcst_pentad_itp[imb,ipt,:,:] = itpfct_1d.__call__(lat_target, lon_target, grid=True)




##  3. Plot member 0 and ensemble mean forecast for the Nov 16-20 precipitation accumulation

ipt = 3*6 + 3

date_pentad_start = dates_fcst[0,0,:].astype(str)[pentad_end_idx[ipt_fcst]-4]
date_pentad_end = dates_fcst[0,0,:].astype(str)[pentad_end_idx[ipt_fcst]]

sfep.plot_fields (fields_list = [prcp_fcst_pentad[0,ipt,:,:],np.mean(prcp_fcst_pentad[:,ipt,:,:],axis=0)],
          lon = lon_fcst,
          lat = lat_fcst,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f"ECMWF forecast for precipitation accumulation between {date_pentad_start} and {date_pentad_end}",
          subtitle_list = [f'control member','ensemble mean'],
          vmin = 0,
          vmax = [15,15],
          unit = 'mm')


sfep.plot_fields (fields_list = [prcp_fcst_pentad_itp[0,ipt,:,:],np.mean(prcp_fcst_pentad_itp[:,ipt,:,:],axis=0)],
          lon = lon_target,
          lat = lat_target,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f"Spatially interpolated ECMWF forecast for precipitation accumulation between {date_pentad_start} and {date_pentad_end}",
          subtitle_list = [f'control member','ensemble mean'],
          vmin = 0,
          vmax = [15,15],
          unit = 'mm')




##  4. Plot climatological median of ECMWF forecasts and CHIRPS/IMERG data for this pentad

jpt = (6*(init_month-1) + ipt) % 72

sfep.plot_fields (fields_list = [prcp_fcst_pct[ipt,49,:,:],prcp_target_pct[jpt,49,:,:]],
          lon = lon_target,
          lat = lat_target,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f"Climatological median precipitation accumulation for November 16-20",
          subtitle_list = ['ECMWF forecast system',f'{ground_truth} observation product'],
          vmin = 0,
          vmax = 15,
          unit = 'mm')

sfep.plot_fields (fields_list = [prcp_fcst_pct[ipt,94,:,:],prcp_target_pct[jpt,94,:,:]],
          lon = lon_target,
          lat = lat_target,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f"Climatological 95th quantile of precipitation accumulation for November 16-20",
          subtitle_list = ['ECMWF forecast system',f'{ground_truth} observation product'],
          vmin = 0,
          vmax = 30,
          unit = 'mm')




##  5. CDFs at Nairobi and quantile mapping example

ix = 167   # 36.75
iy = 137   # -1.25

pct_level = np.arange(1,100)/100

plt.figure(figsize=(5,3.5))
ax1 = plt.subplot(1, 1, 1)
ax1.plot(prcp_fcst_pct[ipt,:,iy,ix], pct_level, color='darkblue', label='forecast CDF')
ax1.plot(prcp_target_pct[ipt,:,iy,ix], pct_level, color='orange', label='observation CDF')
ax1.set_xticks([*range(0,26,5)])
ax1.set_xticklabels([f'{i}mm' for i in range(0,26,5)])
ax1.set_xlabel('5-day rainfall accumulation')
ax1.set_ylabel('probability')
ax1.legend(loc='lower right')
ax1.set_title(f"Climatological CDFs at Nairobi (November 16-20)", fontsize=12)
plt.tight_layout()
plt.savefig(f'/home/michael/Desktop/tmp/quantile-mapping/cdfs-Nairobi.png')

idx = 85

plt.figure(figsize=(5,3.5))
ax1 = plt.subplot(1, 1, 1)
ax1.plot(prcp_fcst_pct[ipt,:,iy,ix], pct_level, color='darkblue', label='forecast CDF')
ax1.plot(prcp_target_pct[ipt,:,iy,ix], pct_level, color='orange', label='observation CDF')
ax1.arrow(prcp_fcst_pct[ipt,idx,iy,ix], 0, 0, pct_level[idx]-0.05, color='blue', head_length=0.05, head_width=0.5)
ax1.text(prcp_fcst_pct[ipt,idx,iy,ix]+1, 0.02, f'x = {np.round(prcp_fcst_pct[ipt,idx,iy,ix],1)} mm', color='blue')
ax1.set_xticks([*range(0,26,5)])
ax1.set_xticklabels([f'{i}mm' for i in range(0,26,5)])
ax1.set_xlabel('5-day rainfall accumulation')
ax1.set_ylabel('probability')
ax1.legend(loc='lower right')
ax1.set_title(f"Quantile mapping example at Nairobi", fontsize=12)
plt.tight_layout()
plt.savefig(f'/home/michael/Desktop/tmp/quantile-mapping/quantile-mapping-example-1.png')


plt.figure(figsize=(5,3.5))
ax1 = plt.subplot(1, 1, 1)
#ax1.plot(prcp_fcst_pct[ipt,:,iy,ix], pct_level, color='darkblue', label='forecast CDF')
#ax1.plot(prcp_target_pct[ipt,:,iy,ix], pct_level, color='orange', label='observation CDF')
ax1.scatter(prcp_fcst_pct[ipt,:,iy,ix], pct_level, color='darkblue', label='forecast percentiles', s=2)
ax1.scatter(prcp_target_pct[ipt,:,iy,ix], pct_level, color='orange', label='observation percentiles', s=2)
ax1.arrow(prcp_fcst_pct[ipt,idx,iy,ix], 0, 0, pct_level[idx]-0.05, color='blue', head_length=0.05, head_width=0.5)
ax1.arrow(prcp_fcst_pct[ipt,idx,iy,ix], pct_level[idx], prcp_target_pct[ipt,idx,iy,ix]-prcp_fcst_pct[ipt,idx,iy,ix]-0.7, 0, color='red', head_length=0.7, head_width=0.03)
ax1.text(prcp_fcst_pct[ipt,idx,iy,ix]+1, 0.02, f'x = {np.round(prcp_fcst_pct[ipt,idx,iy,ix],1)} mm', color='blue')
ax1.text(prcp_target_pct[ipt,idx,iy,ix], pct_level[idx]-0.08, r'$\tilde{x}$'+' = '+str(np.round(prcp_target_pct[ipt,idx,iy,ix],1))+' mm', color='red')
ax1.set_xticks([*range(0,26,5)])
ax1.set_xticklabels([f'{i}mm' for i in range(0,26,5)])
ax1.set_xlabel('5-day rainfall accumulation')
ax1.set_ylabel('probability')
ax1.legend(loc='lower right')
ax1.set_title(f"Quantile mapping example at Nairobi", fontsize=12)
plt.tight_layout()
#plt.savefig(f'/home/michael/Desktop/tmp/quantile-mapping/quantile-mapping-example-2a.png')
plt.savefig(f'/home/michael/Desktop/tmp/quantile-mapping/quantile-mapping-example-2b.png')


plt.figure(figsize=(5,3.5))
ax1 = plt.subplot(1, 1, 1)
ax1.scatter(prcp_fcst_pct[ipt,:,iy,ix], pct_level, color='darkblue', label='forecast percentiles', s=2)
ax1.scatter(prcp_target_pct[ipt,:,iy,ix], pct_level, color='orange', label='observation percentiles', s=2)
ax1.set_xticks([*range(0,26,5)])
ax1.set_xticklabels([f'{i}mm' for i in range(0,26,5)])
ax1.set_xlabel('5-day rainfall accumulation')
ax1.set_ylabel('probability')
ax1.legend(loc='lower right')
ax1.set_title('Climatological '+'\u0336'.join('CDF') +'\u0336'+' percentiles at Nairobi', fontsize=12)
plt.tight_layout()
plt.savefig(f'/home/michael/Desktop/tmp/quantile-mapping/pctls-Nairobi.png')


plt.figure(figsize=(4.5,4.5))
ax1 = plt.subplot(1, 1, 1)
ax1.scatter(prcp_fcst_pct[ipt,:,iy,ix], prcp_target_pct[ipt,:,iy,ix], color='darkgreen', s=2)
ax1.arrow(prcp_fcst_pct[ipt,idx,iy,ix], 0, 0, prcp_target_pct[ipt,idx,iy,ix]-1.5, color='blue', head_length=1.5, head_width=0.8, shape='left')
ax1.arrow(prcp_fcst_pct[ipt,idx,iy,ix]-0.1, 0, 0, prcp_target_pct[ipt,idx,iy,ix]-1.5, color='red', head_length=1.5, head_width=0.8, shape='right')
ax1.text(prcp_fcst_pct[ipt,idx,iy,ix]+1, 0.02, f'x = {np.round(prcp_fcst_pct[ipt,idx,iy,ix],1)} mm', color='blue')
ax1.text(prcp_fcst_pct[ipt,idx,iy,ix]+1, prcp_target_pct[ipt,idx,iy,ix], r'$\tilde{x}$'+' = '+str(np.round(prcp_target_pct[ipt,idx,iy,ix],1))+' mm', color='red')
ax1.set_xlim(-0.5,27.5)
ax1.set_xticks([*range(0,26,5)])
ax1.set_xticklabels([f'{i}mm' for i in range(0,26,5)])
ax1.set_xlabel('5-day rainfall accumulation (forecast)')
ax1.set_ylim(-0.5,27.5)
ax1.set_yticks([*range(0,26,5)])
ax1.set_yticklabels([f'{i}mm' for i in range(0,26,5)])
ax1.set_ylabel('5-day rainfall accumulation (observation)')
ax1.set_title(f"Quantile mapping example at Nairobi", fontsize=12)
plt.tight_layout()
plt.savefig(f'/home/michael/Desktop/tmp/quantile-mapping/quantile-mapping-example-3.png')




##  6. CDFs at Khartoum and quantile mapping example

ix = 126   # 32.65
iy = 305   # 15.55

pct_level = np.arange(1,100)/100

plt.figure(figsize=(5,3.5))
ax1 = plt.subplot(1, 1, 1)
ax1.scatter(prcp_fcst_pct[ipt,:,iy,ix], pct_level, color='darkblue', label='forecast percentiles', s=2)
ax1.scatter(prcp_target_pct[ipt,:,iy,ix], pct_level, color='orange', label='observation percentiles', s=2)
ax1.set_xticks([i/100 for i in range(0,16,5)])
ax1.set_xticklabels([f'{i/100}mm' for i in range(0,16,5)])
ax1.set_xlabel('5-day rainfall accumulation')
ax1.set_ylabel('probability')
ax1.legend(loc='lower right')
ax1.set_title(f"Climatological percentiles at Khartoum", fontsize=12)
plt.tight_layout()
plt.savefig(f'/home/michael/Desktop/tmp/quantile-mapping/cdfs-Khartoum.png')


plt.figure(figsize=(4.5,4.5))
ax1 = plt.subplot(1, 1, 1)
ax1.scatter(prcp_fcst_pct[ipt,:,iy,ix], prcp_target_pct[ipt,:,iy,ix], color='darkgreen', s=4)
ax1.set_xlim(-0.005,0.155)
ax1.set_xticks([i/100 for i in range(0,16,5)])
ax1.set_xticklabels([f'{i/100}mm' for i in range(0,16,5)])
ax1.set_xlabel('5-day rainfall accumulation (forecast)')
ax1.set_ylim(-0.005,0.155)
ax1.set_yticks([i/100 for i in range(0,16,5)])
ax1.set_yticklabels([f'{i/100}mm' for i in range(0,16,5)])
ax1.set_ylabel('5-day rainfall accumulation (observation)')
ax1.set_title(f"Quantile mapping example at Khartoum", fontsize=12)
plt.tight_layout()
plt.savefig(f'/home/michael/Desktop/tmp/quantile-mapping/quantile-mapping-example-4.png')


plt.figure(figsize=(4.5,4.5))
ax1 = plt.subplot(1, 1, 1)
ax1.scatter(prcp_fcst_pct[ipt,:,iy,ix], prcp_target_pct[ipt,:,iy,ix], color='darkgreen', s=4)
ax1.plot(np.arange(0,0.15,0.01), 0.03+0.9*np.arange(0,0.15,0.01), color='darkred', linestyle='--')
ax1.text(0.05, 0.05, 'linear approximation', color='darkred')
ax1.set_xlim(-0.005,0.155)
ax1.set_xticks([i/100 for i in range(0,16,5)])
ax1.set_xticklabels([f'{i/100}mm' for i in range(0,16,5)])
ax1.set_xlabel('5-day rainfall accumulation (forecast)')
ax1.set_ylim(-0.005,0.155)
ax1.set_yticks([i/100 for i in range(0,16,5)])
ax1.set_yticklabels([f'{i/100}mm' for i in range(0,16,5)])
ax1.set_ylabel('5-day rainfall accumulation (observation)')
ax1.set_title(f"Quantile mapping example at Khartoum", fontsize=12)
plt.tight_layout()
plt.savefig(f'/home/michael/Desktop/tmp/quantile-mapping/quantile-mapping-example-5.png')




##  7. Plot bias-corrected forecasts

prcp_fcst_pentad_bc = np.full((nmbs,npts,nlat,nlon), np.nan, dtype=np.float32)

for ix in range(nlon):
    print(ix,nlon)
    for iy in range(nlat):
        if np.any(np.isnan(prcp_target_pct[:,:,iy,ix])):
            continue
        for ipt in range(npts):
            jpt = (6*(init_month-1) + ipt) % 72         # which pentad of the year are we in
            nzi = (prcp_fcst_pct[ipt,:,iy,ix]>0.1)
            if np.sum(nzi) == 0:
                prcp_fcst_pentad_bc[:,ipt,iy,ix] = prcp_fcst_pentad_itp[:,ipt,iy,ix]    # no bias correction possible
            elif np.sum(nzi) == 1:
                x0 = prcp_fcst_pct[ipt,-1,iy,ix]
                y0 = prcp_target_pct[jpt,-1,iy,ix]
                prcp_fcst_pentad_bc[:,ipt,iy,ix] = np.where(prcp_fcst_pentad_itp[:,ipt,iy,ix]==0.,0.,np.maximum(0.,y0+prcp_fcst_pentad_itp[:,ipt,iy,ix]-x0))    # simple additive bias correction
            elif np.sum(nzi) == 2:
                x0 = prcp_fcst_pct[ipt,-2,iy,ix]
                y0 = prcp_target_pct[jpt,-2,iy,ix]
                itp_ind1 = np.logical_and(~np.isnan(prcp_fcst_pentad_itp[:,ipt,iy,ix]), prcp_fcst_pentad_itp[:,ipt,iy,ix]<=x0)
                itp_ind2 = np.logical_and(~np.isnan(prcp_fcst_pentad_itp[:,ipt,iy,ix]), prcp_fcst_pentad_itp[:,ipt,iy,ix]>x0)
                slope1 = y0/x0
                slope2 = (prcp_target_pct[jpt,-1,iy,ix]-y0)/(prcp_fcst_pct[ipt,-1,iy,ix]-x0)
                prcp_fcst_pentad_bc[:,ipt,iy,ix][itp_ind1] = slope1*prcp_fcst_pentad_itp[itp_ind1,ipt,iy,ix]              # linear interpolation
                prcp_fcst_pentad_bc[:,ipt,iy,ix][itp_ind2] = y0 + slope2*(prcp_fcst_pentad_itp[itp_ind2,ipt,iy,ix]-x0)    # linear extrapolation
            else:
                x0 = prcp_fcst_pct[ipt,-3,iy,ix]
                y0 = prcp_target_pct[jpt,-3,iy,ix]
                slope = np.sum(prcp_target_pct[jpt,-2:,iy,ix]-y0)/np.sum(prcp_fcst_pct[ipt,-2:,iy,ix]-x0)
                x = np.append(0.,prcp_fcst_pct[ipt,nzi,iy,ix])
                y = np.append(0.,prcp_target_pct[jpt,nzi,iy,ix])
                itp_fct = interp1d(x, y, kind='linear', fill_value='extrapolate')
                itp_ind = np.logical_and(~np.isnan(prcp_fcst_pentad_itp[:,ipt,iy,ix]), prcp_fcst_pentad_itp[:,ipt,iy,ix]<x0)
                prcp_fcst_pentad_bc[:,ipt,iy,ix][itp_ind] = itp_fct(prcp_fcst_pentad_itp[itp_ind,ipt,iy,ix])            # linear interpolation below the 85th percentile
                prcp_fcst_pentad_bc[:,ipt,iy,ix][~itp_ind] = y0 + slope*(prcp_fcst_pentad_itp[~itp_ind,ipt,iy,ix]-x0)   # linear extrapolation above 85th percentile, slope estimated with 90th/95th percentile


ipt = 3*6 + 3

sfep.plot_fields (fields_list = [prcp_fcst_pentad_bc[0,ipt,:,:],np.mean(prcp_fcst_pentad_bc[:,ipt,:,:],axis=0)],
          lon = lon_target,
          lat = lat_target,
          lon_bounds = [22, 52],
          lat_bounds = [-12, 19],
          main_title = f"Bias-corrected ECMWF forecast for precipitation accumulation between {date_pentad_start} and {date_pentad_end}",
          subtitle_list = [f'control member','ensemble mean'],
          vmin = 0,
          vmax = [15,15],
          unit = 'mm')



##  8. Illustration of PDF, CDF, PPF

x = np.arange(-3.5, 3.5, 1/20.)
y = norm.pdf(x)
z = norm.cdf(x)

plt.figure(figsize=(8,4))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(x, y, color='k')
ax1.set_ylabel('density')
ax1.set_title("Standard Gaussian distribution: PDF", fontsize=12)
ax2 = plt.subplot(1, 2, 2)
ax2.plot(x, z, color='k')
ax2.set_ylabel('probability')
ax2.set_title("Standard Gaussian distribution: CDF", fontsize=12)
plt.tight_layout()
plt.savefig(f'/home/michael/Desktop/tmp/quantile-mapping/pdf-cdf-0.png')

for i in range(1,5):
    idx = i*25
    plt.figure(figsize=(8,4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(x, y, color='k')
    ax1.axvline(x=x[idx], color='magenta', linestyle='--')
    ax1.fill_between(x[:(idx+1)],y[:(idx+1)])
    ax1.set_ylabel('density')
    ax1.set_title("Standard Gaussian distribution: PDF", fontsize=12)
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(x, z, color='k')
    ax2.scatter(x[idx], z[idx])
    ax2.axvline(x=x[idx], color='magenta', linestyle='--')
    ax2.set_ylabel('probability')
    ax2.set_title("Standard Gaussian distribution: CDF", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'/home/michael/Desktop/tmp/quantile-mapping/pdf-cdf-{i}.png')


idx = 103   # q95

plt.figure(figsize=(4,4))
ax1 = plt.subplot(1, 1, 1)
ax1.plot(x, z, color='k')
ax1.arrow(x[0], z[idx], x[idx]-x[0]-0.35, 0, color='red', head_length=0.3, head_width=0.03)
ax1.text(x[0]-1.2, z[idx]-0.02, 0.95, color='red')
ax1.arrow(x[idx], z[idx], 0, -(z[idx]-0.05), color='red', head_length=0.05, head_width=0.2)
ax1.text(x[idx]-0.25, -0.05, 'q95', color='red')
ax2.set_ylabel('probability')
plt.tight_layout()
plt.savefig(f'/home/michael/Desktop/tmp/quantile-mapping/example-q95.png')






















import numpy as np

import seasonal.automation as sfea
import seasonal.bookkeeping as sfeb


# Dirty trick to deal with netCDF4 deprecation issue
np.bool = np.bool_

grib_dir = '/home/michael/bigdisk2/pro/SFE/Systems_daily_grib/'      # directory where the grib files stored
netcdf_dir = '/home/michael/bigdisk2/pro/SFE/Systems_daily_nc/'      # directory where the netcdf forecast files are stored

year_fcst = 2023          # year for which the bias-corrected forecasts are sought
init_month = 8            # forecast initialization month


# Download ECMWF seasonal forecasts of daily precipitation accumulations
sfea.download_cds_daily_forecasts(variables = ['total_precipitation'],
                                  systems = ['ecmwf'],
                                  system_numbers = [51],
                                  years = [year_fcst],
                                  months = [init_month],
                                  grib_dir = grib_dir,
                                  netcdf_dir = netcdf_dir)


sfeb.check_inventory_cds ('total_precipitation', data_dir = netcdf_dir)




