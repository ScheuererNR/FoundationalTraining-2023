
# Functions for plotting. Some of this code was obtained from Joseph Bellier and then modified
#  and extended for the purposes of this project.

import numpy as np

from calendar import monthrange

from scipy.interpolate import RectBivariateSpline, interp1d


def get_nearest_grid_index(lon_exmpl, lat_exmpl, lon_grid, lat_grid):
    ix = np.argmin(abs(lon_grid-lon_exmpl))
    iy = np.argmin(abs(lat_grid-lat_exmpl))
    return ix, iy


def quantile_mapping(fcst_new, fcst_pct, target_pct):
    nzi = (fcst_pct>0.1)
    fcst_bc = fcst_new.copy()          # default: no bias correction
    if np.sum(nzi) == 1:
        x0 = fcst_pct[-1]
        y0 = target_pct[-1]
        fcst_bc = np.where(fcst_new==0.,0.,np.maximum(0.,y0+fcst_new-x0))    # simple additive bias correction
    elif np.sum(nzi) == 2:
        x0 = fcst_pct[-2]
        y0 = target_pct[-2]
        itp_ind1 = np.logical_and(~np.isnan(fcst_new), fcst_new<=x0)
        itp_ind2 = np.logical_and(~np.isnan(fcst_new), fcst_new>x0)
        slope1 = y0/x0
        slope2 = (target_pct[-1]-y0)/(fcst_pct[-1]-x0)
        fcst_bc[itp_ind1] = slope1*fcst_new[itp_ind1]              # linear interpolation
        fcst_bc[itp_ind2] = y0 + slope2*(fcst_new[itp_ind2]-x0)    # linear extrapolation
    elif np.sum(nzi) > 2:
        x0 = fcst_pct[-3]
        y0 = target_pct[-3]
        slope = np.sum(target_pct[-2:]-y0)/np.sum(fcst_pct[-2:]-x0)
        x = np.append(0.,fcst_pct[nzi])
        y = np.append(0.,target_pct[nzi])
        itp_fct = interp1d(x, y, kind='linear', fill_value='extrapolate')
        itp_ind = np.logical_and(~np.isnan(fcst_new), fcst_new<x0)
        fcst_bc[itp_ind] = itp_fct(fcst_new[itp_ind])              # linear interpolation below the 85th percentile
        fcst_bc[~itp_ind] = y0 + slope*(fcst_new[~itp_ind]-x0)     # linear extrapolation above 85th percentile, slope estimated with 90th/95th percentile
    return fcst_bc


def calculate_pentad_accumulations(prcp_fcst, year_fcst, month_init):
    nmbs, nlts, nlatf, nlonf = prcp_fcst.shape
    nlm = nlts // 30
    npts = 6*nlm
   # Find indices that delineate the pentads
    pentad_end_idx = np.zeros(npts, dtype=np.int32)
    pentad_end_idx[0] = 4
    for imt in range(nlm):
        year_valid = year_fcst + (month_init+imt-1)//12
        month_valid = 1 + (month_init+imt-1)%12
        days_this_month = 0
        for ipt in range(5):
            days_this_month += 5
            if imt == 0 and ipt == 0:
                continue
            pentad_end_idx[6*imt+ipt] = pentad_end_idx[6*imt+ipt-1] + 5
        pentad_end_idx[6*imt+5] =  pentad_end_idx[6*imt+4] + monthrange(year_valid, month_valid)[1] - days_this_month
   # Calculate accumulations between these delineations
    prcp_fcst_pentad = np.full((nmbs,npts,nlatf,nlonf), np.nan, dtype=np.float32)
    prcp_fcst_pentad[:,0,:,:] = 200.*prcp_fcst[:,pentad_end_idx[0],:,:]
    for ipt in range(1,npts):
        ilt0 = pentad_end_idx[ipt-1]
        ilt1 = pentad_end_idx[ipt]
        prcp_fcst_pentad[:,ipt,:,:] = np.maximum(0.,1000.*(prcp_fcst[:,ilt1,:,:]-prcp_fcst[:,ilt0,:,:])/(ilt1-ilt0))
    return prcp_fcst_pentad


def interpolate_forecasts(prcp_fcst, lat_fcst, lon_fcst, lat_target, lon_target):
    nmbs, nlts, nlatf, nlonf = prcp_fcst.shape
    prcp_fcst_itp = np.full((nmbs,nlts,len(lat_target),len(lon_target)), np.nan, dtype=np.float32)
    for imb in range(nmbs):
        for ilt in range(nlts):
            if np.all(np.isnan(prcp_fcst[imb,ilt,:,:])):
                continue
            itpfct = RectBivariateSpline(lat_fcst, lon_fcst, prcp_fcst[imb,ilt,:,:], kx=1, ky=1, s=0)
            prcp_fcst_itp[imb,ilt,:,:] = itpfct.__call__(lat_target, lon_target, grid=True)
    return prcp_fcst_itp


def find_onset_day(exc1_ind, exc20_ind):
    n = len(exc1_ind)
    wet_spell = np.logical_and(exc20_ind[:(n-2)], exc1_ind[:(n-2)])
    ind_lag7d = np.expand_dims(np.arange(7),0)+np.expand_dims(np.arange(n-6),0).T
    dry_spell = np.all(~exc1_ind[ind_lag7d], axis=1)
    ind_lag15d = np.expand_dims(np.arange(15),0)+np.expand_dims(np.arange(n-23),0).T
    onset = np.logical_and(wet_spell[:(n-23)], ~np.any(dry_spell[3:][ind_lag15d], axis=1))
    onset_day = -1 if np.all(~onset) else np.nonzero(onset)[0][0]+1
    return onset_day


def mark_wet_spells(exc1_ind, exc20_ind):
    n = len(exc1_ind)
    wet_spell = np.zeros(n, dtype=bool)
    wet_spell_day1 = np.logical_and(exc20_ind, exc1_ind)
    for idt in range(n-2):
        if wet_spell_day1[idt]:
            wet_spell[idt:(idt+3)] = True
    return wet_spell


def mark_dry_spells(exc1_ind):
    n = len(exc1_ind)
    dry_spell = np.zeros(n, dtype=bool)
    ind_lag7d = np.expand_dims(np.arange(7),0)+np.expand_dims(np.arange(n-6),0).T
    dry_spell_day1 = np.append(np.all(~exc1_ind[ind_lag7d], axis=1), np.zeros(6, dtype=bool))
    for idt in range(n-2):
        if dry_spell_day1[idt]:
            dry_spell[idt:(idt+7)] = True
    return dry_spell

























