"""
Created by Marion Bocquet
Date : 15/04/2024
Credits : LEGOS/CNES/CLS

The script aims to plot the time series maps
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import xarray as xr
import seaborn as sns
from scipy import stats
import pymannkendall as mk
import os.path
from mpl_toolkits.basemap import Basemap
import cmcrameri.cm as cmc
import dateutil
import plots_MAPS_function as pmf


# ------- LOAD DATA -------- #

# load dataset in zarr format with 'ice_conc', 'sit_interp', 'radar_freeboard_interp', 'mask_acq'
# 'ice_conc' :corresponds to the sea ice concentration
# 'sit_intepr' : corresponds to the sea ice thickness interpolated where ice_conc > 50
# 'radar_freeboard_interp' : same as for sit_interp
# 'mask_acq' : corresponds to the orbit mask (remove data above 81.5°)

ds_NH = xr.open_zarr('unc_all_sats_NH_199301_202304.zarr')[['ice_conc', 'sit_interp', 'radar_freeboard_interp', 'mask_acq']]
ds_SH = xr.open_zarr('all_sats_SH_199301_202306.zarr')[['ice_conc', 'sit_interp', 'radar_freeboard_interp', 'mask_acq']]
ds_NH_0 = ds_NH.fillna(0)
ds_SH_0 = ds_SH.fillna(0)

# Compute climatologies
clim_NH = ds_NH_0[['ice_conc', 'sit_interp', 'radar_freeboard_interp', 'mask_acq']].groupby('time.month').mean()
print('clim NH computed')
clim_SH = ds_SH_0[['ice_conc', 'sit_interp', 'radar_freeboard_interp', 'mask_acq']].groupby('time.month').mean()
print('clim SH computed')

var_NH = ds_NH_0[['ice_conc', 'sit_interp', 'radar_freeboard_interp', 'mask_acq']].groupby('time.month').std()
var_SH = ds_SH_0[['ice_conc', 'sit_interp', 'radar_freeboard_interp', 'mask_acq']].groupby('time.month').std()
mean_SH = ds_SH_0[['ice_conc', 'sit_interp', 'radar_freeboard_interp', 'mask_acq']].groupby('time.year').mean()

ds_SH = ds_SH.assign(month = ds_SH.time.dt.month)
ds_NH = ds_NH.assign(month = ds_NH.time.dt.month)
# make a year (full year starting by the begining of winter) to compute yearly mean
time_sh = pd.to_datetime(ds_SH.sit_interp.coords['time'].values)
time_vec_sh = [t - dateutil.relativedelta.relativedelta(months=2) for t in time_sh]
ds_SH_time_changes = ds_SH.assign(time=time_vec_sh)

# make a year (full year starting by the begining of summer) to compute yearly mean
time_nh = pd.to_datetime(ds_NH.sit_interp.coords['time'].values)
time_vec_nh = [t + dateutil.relativedelta.relativedelta(months=3) for t in time_nh]
ds_NH_time_changes = ds_NH.assign(time=time_vec_nh)

####### ---------- Maps for figure 1 ----------- #######
pmf.clim_trends_year(ds_NH_time_changes, ds_SH_time_changes, ds_NH, ds_SH, ds_NH_0, ds_SH_0)

# Computation and load trends for supplement and appendices
if os.path.isfile('NH_pval_01.npy'):
    nhpval_01 = np.load('NH_pval_01.npy')
    nhtrend_01 = np.load('NH_trend_01.npy')
    print('01 exist !')
else:
    time01=time.time()
    nhtrend_01, nhsign_mask_01, nhpval_01 = pmf.trend_month_map('01', ds_NH, 'sit_interp', 1994, 2023)
    nhtrend_01[ds_NH.mask_acq.values==0] = np.nan
    np.save('NH_pval_01.npy', nhpval_01)
    np.save('NH_trend_01.npy', nhtrend_01)
    print((time.time()-time01)/60)

print('NH 01')

if os.path.isfile('NH_pval_02.npy'):
    nhtrend_02 = np.load('NH_trend_02.npy')
    nhpval_02 = np.load('NH_pval_02.npy')
    print('02 exist !')
else:
    time02 = time.time()
    nhtrend_02, nhsign_mask_02, nhpval_02 = pmf.trend_month_map('02', ds_NH, 'sit_interp', 1994, 2023)
    nhtrend_02[ds_NH.mask_acq.values==0] = np.nan
    np.save('NH_pval_02.npy', nhpval_02)
    np.save('NH_trend_02.npy', nhtrend_02)
    print((time.time()-time02)/60)

print('NH 02')

if os.path.isfile('NH_pval_03.npy'):
    nhtrend_03 = np.load('NH_trend_03.npy')
    nhpval_03 = np.load('NH_pval_03.npy')
    print('03 exist !')
else:
    time03 = time.time()
    nhtrend_03, nhsign_mask_03, nhpval_03 = pmf.trend_month_map('03', ds_NH, 'sit_interp', 1994, 2023)
    nhtrend_03[ds_NH.mask_acq.values==0] = np.nan
    np.save('NH_pval_03.npy', nhpval_03)
    np.save('NH_trend_03.npy', nhtrend_03)
    print((time.time()-time03)/60)

print('NH 03')
if os.path.isfile('NH_pval_04.npy'):
    nhtrend_04 = np.load('NH_trend_04.npy')
    nhpval_04 = np.load('NH_pval_04.npy')
    print('04 exist !')
else:
    nhtrend_04, nhsign_mask_04, nhpval_04 = pmf.trend_month_map('04', ds_NH, 'sit_interp', 1994, 2023)
    nhtrend_04[ds_NH.mask_acq.values==0] = np.nan
    np.save('NH_pval_04.npy', nhpval_04)
    np.save('NH_trend_04.npy', nhtrend_04)
print('NH 04')

if os.path.isfile('NH_pval_10.npy'):
    nhtrend_10 = np.load('NH_trend_10.npy')
    nhpval_10 = np.load('NH_pval_10.npy')
    print('10 exist !')
else:
    nhtrend_10, nhsign_mask_10, nhpval_10 = pmf.trend_month_map('10', ds_NH, 'sit_interp', 1994, 2023)
    nhtrend_10[ds_NH.mask_acq.values==0] = np.nan
    np.save('NH_pval_10.npy', nhpval_10)
    np.save('NH_trend_10.npy', nhtrend_10)

print('NH 10')

if os.path.isfile('NH_pval_11.npy'):
    nhtrend_11 = np.load('NH_trend_11.npy')
    nhpval_11 = np.load('NH_pval_11.npy')
    print('11 exist !')
else:
    nhtrend_11, nhsign_mask_11, nhpval_11 = pmf.trend_month_map('11', ds_NH, 'sit_interp', 1994, 2023)
    nhtrend_11[ds_NH.mask_acq.values==0] = np.nan
    np.save('NH_pval_11.npy', nhpval_11)
    np.save('NH_trend_11.npy', nhtrend_11)
print('NH 11')

if os.path.isfile('NH_pval_12.npy'):
    nhtrend_12 = np.load('NH_trend_12.npy')
    nhpval_12 = np.load('NH_pval_12.npy')
    print('12 exist !')
else:
    nhtrend_12, nhsign_mask_12, nhpval_12 = pmf.trend_month_map('12', ds_NH, 'sit_interp', 1994, 2023)
    nhtrend_12[ds_NH.mask_acq.values==0] = np.nan
    np.save('NH_pval_12.npy', nhpval_12)
    np.save('NH_trend_12.npy', nhtrend_12)
print('NH 12')

# -------------------- SH ------------------ #

if os.path.isfile('SH_pval_01.npy'):
    shtrend_01 = np.load('SH_trend_01.npy')
    shpval_01 = np.load('SH_pval_01.npy')
    print('SH 01 exist !')
else:
    shtrend_01, shsign_mask_01, shpval_01 = pmf.trend_month_map('01', ds_SH, 'sit_interp', 1994, 2023)
    shtrend_01[ds_SH.mask_acq.values==0] = np.nan
    np.save('SH_pval_01.npy', shpval_01)
    np.save('SH_trend_01.npy', shtrend_01)
    print('SH 01')
print('SH 01')

if os.path.isfile('SH_pval_02.npy'):
    shtrend_02 = np.load('SH_trend_02.npy')
    shpval_02 = np.load('SH_pval_02.npy')
    print('02 exist !')
else:
    shtrend_02, shsign_mask_02, shpval_02 = pmf.trend_month_map('02', ds_SH, 'sit_interp', 1994, 2023)
    shtrend_02[ds_SH.mask_acq.values==0] = np.nan
    np.save('SH_pval_02.npy', shpval_02)
    np.save('SH_trend_02.npy', shtrend_02)
    print('SH 02')
print('SH 02')

if os.path.isfile('SH_pval_03.npy'):
    shtrend_03 = np.load('SH_trend_03.npy')
    shpval_03 = np.load('SH_pval_03.npy')
    print('03 exist !')
else:
    shtrend_03, shsign_mask_03, shpval_03 = pmf.trend_month_map('03', ds_SH, 'sit_interp', 1994, 2023)
    shtrend_03[ds_SH.mask_acq.values==0] = np.nan
    np.save('SH_pval_03.npy', shpval_03)
    np.save('SH_trend_03.npy', shtrend_03)
print('SH 03')

if os.path.isfile('SH_pval_04.npy'):
    shtrend_04 = np.load('SH_trend_04.npy')
    shpval_04 = np.load('SH_pval_04.npy')
    print('04 exist !')
else:
    shtrend_04, shsign_mask_04, shpval_04 = pmf.trend_month_map('04', ds_SH, 'sit_interp', 1994, 2023)
    shtrend_04[ds_SH.mask_acq.values==0] = np.nan
    np.save('SH_pval_04.npy', shpval_04)
    np.save('SH_trend_04.npy', shtrend_04)
print('SH 04')

if os.path.isfile('SH_pval_05.npy'):
    shtrend_05 = np.load('SH_trend_05.npy')
    shpval_05 = np.load('SH_pval_05.npy')
    print('05 exist !')
else:
    shtrend_05, shsign_mask_05, shpval_05 = pmf.trend_month_map('05', ds_SH, 'sit_interp', 1994, 2023)
    shtrend_05[ds_SH.mask_acq.values==0] = np.nan
    np.save('SH_pval_05.npy', shpval_05)
    np.save('SH_trend_05.npy', shtrend_05)

print('SH 05')

if os.path.isfile('SH_pval_06.npy'):
    shtrend_06 = np.load('SH_trend_06.npy')
    shpval_06 = np.load('SH_pval_06.npy')
    print('06 exist !')
else:
    shtrend_06, shsign_mask_06, shpval_06 = pmf.trend_month_map('06', ds_SH, 'sit_interp', 1994, 2023)
    shtrend_06[ds_SH.mask_acq.values==0] = np.nan
    np.save('SH_pval_06.npy', shpval_06)
    np.save('SH_trend_06.npy', shtrend_06)

print('SH 06')

if os.path.isfile('SH_pval_07.npy'):
    shtrend_07 = np.load('SH_trend_07.npy')
    shpval_07 = np.load('SH_pval_07.npy')
    print('07 exist !')
else:
    shtrend_07, shsign_mask_07, shpval_07 = pmf.trend_month_map('07', ds_SH, 'sit_interp', 1994, 2023)
    shtrend_07[ds_SH.mask_acq.values==0] = np.nan
    np.save('SH_pval_07.npy', shpval_07)
    np.save('SH_trend_07.npy', shtrend_07)

print('SH 07')

if os.path.isfile('SH_pval_08.npy'):
    shtrend_08 = np.load('SH_trend_08.npy')
    shpval_08 = np.load('SH_pval_08.npy')
    print('08 exist !')
else:
    shtrend_08, shsign_mask_08, shpval_08 = pmf.trend_month_map('08', ds_SH, 'sit_interp', 1994, 2023)
    shtrend_08[ds_SH.mask_acq.values==0] = np.nan
    np.save('SH_pval_08.npy', shpval_08)
    np.save('SH_trend_08.npy', shtrend_08)

print('SH 08')

if os.path.isfile('SH_pval_09.npy'):
    shtrend_09 = np.load('SH_trend_09.npy')
    shpval_09 = np.load('SH_pval_09.npy')
    print('09 exist !')
else:
    shtrend_09, shsign_mask_09, shpval_09 = pmf.trend_month_map('09', ds_SH, 'sit_interp', 1994, 2023)
    shtrend_09[ds_SH.mask_acq.values==0] = np.nan
    np.save('SH_pval_09.npy', shpval_09)
    np.save('SH_trend_09.npy', shtrend_09)

print('SH 09')

if os.path.isfile('SH_pval_10.npy'):
    shtrend_10 = np.load('SH_trend_10.npy')
    shpval_10 = np.load('SH_pval_10.npy')
    print('10 exist !')
else:
    shtrend_10, shsign_mask_10, shpval_10 = pmf.trend_month_map('10', ds_SH, 'sit_interp', 1994, 2023)
    shtrend_10[ds_SH.mask_acq.values==0] = np.nan
    np.save('SH_pval_10.npy', shpval_10)
    np.save('SH_trend_10.npy', shtrend_10)
print('SH 10')

if os.path.isfile('SH_pval_11.npy'):
    shtrend_11 = np.load('SH_trend_11.npy')
    shpval_11 = np.load('SH_pval_11.npy')
    print('11 exist !')
else:
    shtrend_11, shsign_mask_11, shpval_11 = pmf.trend_month_map('11', ds_SH, 'sit_interp', 1994, 2023)
    shtrend_11[ds_SH.mask_acq.values==0] = np.nan
    np.save('SH_pval_11.npy', shpval_11)
    np.save('SH_trend_11.npy', shtrend_11)
print('SH 11')

if os.path.isfile('SH_pval_12.npy'):
    shtrend_12 = np.load('SH_trend_12.npy')
    shpval_12 = np.load('SH_pval_12.npy')
    print('12 exist !')
else:
    shtrend_12, shsign_mask_12, shpval_12 = pmf.trend_month_map('12', ds_SH, 'sit_interp', 1994, 2023)
    shtrend_12[ds_SH.mask_acq.values==0] = np.nan
    np.save('SH_pval_12.npy', shpval_12)
    np.save('SH_trend_12.npy', shtrend_12)
print('SH 12')

sns.set_style('whitegrid')
fig = plt.figure(figsize=(12,15))

####### FIGURE B2 ########
plt.subplot(6,4,2)
pmf.plot_map_significance_no_cbar(ds_SH, shtrend_03*10, 1-shpval_03, -1, -0.75, 0.75, title='March')

plt.subplot(6,4,4)
pmf.plot_map_significance_no_cbar(ds_SH, shtrend_04*10, 1-shpval_04, -1, -0.75, 0.75, title='April')

plt.subplot(6,4,6)
pmf.plot_map_significance_no_cbar(ds_SH, shtrend_05*10, 1-shpval_05, -1, -0.75, 0.75, title='May')

plt.subplot(6,4,8)
pmf.plot_map_significance_no_cbar(ds_SH, shtrend_06*10, 1-shpval_06, -1, -0.75, 0.75, title='June')

plt.subplot(6,4,10)
pmf.plot_map_significance_no_cbar(ds_SH, shtrend_07*10, 1-shpval_07, -1, -0.75, 0.75, title='July')

plt.subplot(6,4,12)
pmf.plot_map_significance_no_cbar(ds_SH, shtrend_08*10, 1-shpval_08, -1, -0.75, 0.75, title='August')

plt.subplot(6,4,14)
pmf.plot_map_significance_no_cbar(ds_SH, shtrend_09*10, 1-shpval_09, -1, -0.75, 0.75, title='September')

plt.subplot(6,4,16)
pmf.plot_map_significance_no_cbar(ds_SH, shtrend_10*10, 1-shpval_10, -1, -0.75, 0.75, title='October')


plt.subplot(6,4,18)
pmf.plot_map_significance_no_cbar(ds_SH, shtrend_11*10, 1-shpval_11, -1, -0.75, 0.75, title='November')


plt.subplot(6,4,20)
pmf.plot_map_significance_no_cbar(ds_SH, shtrend_12*10, 1-shpval_12, -1, -0.75, 0.75, title='December')


plt.subplot(6,4,22)
pmf.plot_map_significance_no_cbar(ds_SH, shtrend_01*10, 1-shpval_01, -1, -0.75, 0.75, title='January')


plt.subplot(6,4,24)
cbar_trend = pmf.plot_map_significance_no_cbar(ds_SH, shtrend_02*10, 1-shpval_02, -1, -0.75, 0.75, title='February')

plt.subplot(6,4,1)
month = 3; clim_SH_month = clim_SH.sit_interp.sel(month=month).where(ds_SH.ice_conc.where(ds_SH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_SH_month, -1, 0, 4, cmc.roma_r ,'March')
plt.subplot(6,4,3)

month = 4; clim_SH_month = clim_SH.sit_interp.sel(month=month).where(ds_SH.ice_conc.where(ds_SH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_SH_month, -1, 0, 4, cmc.roma_r ,'April')

plt.subplot(6,4,5)
month = 5; clim_SH_month = clim_SH.sit_interp.sel(month=month).where(ds_SH.ice_conc.where(ds_SH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_SH_month, -1, 0, 4, cmc.roma_r, 'May')

plt.subplot(6,4,7)
month = 6; clim_SH_month = clim_SH.sit_interp.sel(month=month).where(ds_SH.ice_conc.where(ds_SH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_SH_month, -1, 0, 4, cmc.roma_r, 'June')

plt.subplot(6,4,9)
month = 7; clim_SH_month = clim_SH.sit_interp.sel(month=month).where(ds_SH.ice_conc.where(ds_SH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_SH_month, -1, 0, 4, cmc.roma_r,  'July')

plt.subplot(6,4,11)
month = 8; clim_SH_month = clim_SH.sit_interp.sel(month=month).where(ds_SH.ice_conc.where(ds_SH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_SH_month, -1, 0, 4, cmc.roma_r,  'August')

plt.subplot(6,4,13)
month = 9; clim_SH_month = clim_SH.sit_interp.sel(month=month).where(ds_SH.ice_conc.where(ds_SH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_SH_month, -1, 0, 4, cmc.roma_r,  'September')

plt.subplot(6,4,15)
month = 10; clim_SH_month = clim_SH.sit_interp.sel(month=month).where(ds_SH.ice_conc.where(ds_SH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_SH_month, -1, 0, 4, cmc.roma_r,  'October')

plt.subplot(6,4,17)
month = 11; clim_SH_month = clim_SH.sit_interp.sel(month=month).where(ds_SH.ice_conc.where(ds_SH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_SH_month, -1, 0, 4, cmc.roma_r,  'November')

plt.subplot(6,4,19)
month = 12; clim_SH_month = clim_SH.sit_interp.sel(month=month).where(ds_SH.ice_conc.where(ds_SH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_SH_month, -1, 0, 4, cmc.roma_r,  'December')

plt.subplot(6,4,21)
month = 1; clim_SH_month = clim_SH.sit_interp.sel(month=month).where(ds_SH.ice_conc.where(ds_SH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_SH_month, -1, 0, 4, cmc.roma_r, 'January')

plt.subplot(6,4,23)
month = 2; clim_SH_month = clim_SH.sit_interp.sel(month=month).where(ds_SH.ice_conc.where(ds_SH.month==month).mean(dim='time')>=1)

cbar_clim = pmf.plot_map_no_cbar(clim_SH_month, -1, 0, 4, cmc.roma_r, 'February')

cax = fig.add_axes([0.535, 0.035, 0.170, 0.01])
fig.add_axes(cax)
fig.colorbar(cbar_clim, cax = cax, orientation = 'horizontal', label='Mean SIT (m)')

caxt = fig.add_axes([0.7765, 0.035, 0.170, 0.01])
fig.add_axes(caxt)
fig.colorbar(cbar_trend, cax = caxt, orientation = 'horizontal', label='SIT trend (m/decade)')

fig.tight_layout()
plt.subplots_adjust(bottom=0.06)
plt.savefig('trend_SH_1995_2023.pdf')

####### FIGURE B1 ########
# --------------------- NH  ----------------------#

sns.set_style('whitegrid')
fig = plt.figure(figsize=(12,15))

plt.subplot(6,4,2)
pmf.plot_map_significance_no_cbar(ds_NH, nhtrend_10*10, 1-nhpval_10, 1, -0.75, 0.75, title='October')

plt.subplot(6,4,4)
pmf.plot_map_significance_no_cbar(ds_NH, nhtrend_11*10, 1-nhpval_11, 1, -0.75, 0.75, title='November')

plt.subplot(6,4,6)
pmf.plot_map_significance_no_cbar(ds_NH, nhtrend_12*10, 1-nhpval_12, 1, -0.75, 0.75, title='December')

plt.subplot(6,4,8)
pmf.plot_map_significance_no_cbar(ds_NH, nhtrend_01*10, 1-nhpval_01, 1, -0.75, 0.75, title='January')

plt.subplot(6,4,10)
cbar_trend = pmf.plot_map_significance_no_cbar(ds_NH, nhtrend_02*10, 1-nhpval_02, 1, -0.75, 0.75, title='February')

plt.subplot(6,4,12)
pmf.plot_map_significance_no_cbar(ds_NH, nhtrend_03*10, 1-nhpval_03, 1, -0.75, 0.75, title='March')

plt.subplot(6,4,14)
pmf.plot_map_significance_no_cbar(ds_NH, nhtrend_04*10, 1-nhpval_04, 1, -0.75, 0.75, title='April')

plt.subplot(6,4,1)
month = 10; clim_NH_month = clim_NH.sit_interp.sel(month=month).where(ds_NH.ice_conc.where(ds_NH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_NH_month, 1, 0, 4, cmc.roma_r,  'October')

plt.subplot(6,4,3)
month = 11; clim_NH_month = clim_NH.sit_interp.sel(month=month).where(ds_NH.ice_conc.where(ds_NH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_NH_month, 1, 0, 4, cmc.roma_r,  'November')

plt.subplot(6,4,5)
month = 12; clim_NH_month = clim_NH.sit_interp.sel(month=month).where(ds_NH.ice_conc.where(ds_NH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_NH_month, 1, 0, 4, cmc.roma_r,  'December')

plt.subplot(6,4,7)
month = 1; clim_NH_month = clim_NH.sit_interp.sel(month=month).where(ds_NH.ice_conc.where(ds_NH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_NH_month, 1, 0, 4, cmc.roma_r, 'January')

plt.subplot(6,4,9)
month = 2; clim_NH_month = clim_NH.sit_interp.sel(month=month).where(ds_NH.ice_conc.where(ds_NH.month==month).mean(dim='time')>=1)

cbar_clim = pmf.plot_map_no_cbar(clim_NH_month, 1, 0, 4, cmc.roma_r, 'February')

plt.subplot(6,4,11)
month = 3; clim_NH_month = clim_NH.sit_interp.sel(month=month).where(ds_NH.ice_conc.where(ds_NH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_NH_month, 1, 0, 4, cmc.roma_r ,'March')

plt.subplot(6,4,13)

month = 4; clim_NH_month = clim_NH.sit_interp.sel(month=month).where(ds_NH.ice_conc.where(ds_NH.month==month).mean(dim='time')>=1)
pmf.plot_map_no_cbar(clim_NH_month, 1, 0, 4, cmc.roma_r ,'April')


cax = fig.add_axes([0.535, 0.495, 0.170, 0.01])
fig.add_axes(cax)
fig.colorbar(cbar_clim, cax = cax, orientation = 'horizontal', label='Mean SIT (m)')

caxt = fig.add_axes([0.776, 0.495, 0.170, 0.01])
fig.add_axes(caxt)
fig.colorbar(cbar_trend, cax = caxt, orientation = 'horizontal', label='SIT trend (m/decade)')

fig.tight_layout()
plt.subplots_adjust(bottom=0.06, top=0.9764444444444446, left=0.0125, right=0.9875, hspace=0.1669132407197101, wspace=-0.071729957805907)
plt.savefig('trend_NH_1995_2023.pdf')


#Figure SUPLEMENT 4
fig, im = pmf.plot_pannels_month('09', ds_SH, clim_SH, 'sit_interp', -1.5, 1.5)

cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.015])
fig.colorbar(im, cax=cbar_ax, label = 'Sea Ice Thickness anomaly (m)', orientation='horizontal')
plt.subplots_adjust(left=0.01, bottom=0.07, right=0.99, top=0.975, wspace=0.01, hspace=0.07)
plt.savefig('SH_09_SIT_obs.pdf')

#Figure SUPLEMENT 3
fig2, im = pmf.plot_pannels_month('04', ds_SH, clim_SH, 'sit_interp', -1.5, 1.5)
axes = fig2.axes

cbar_ax = fig2.add_axes([0.25, 0.04, 0.5, 0.015])
fig2.colorbar(im, cax=cbar_ax, label = 'Sea Ice Thickness anomaly (m)', orientation='horizontal')
plt.subplots_adjust(left=0.01, bottom=0.07, right=0.99, top=0.975, wspace=0.01, hspace=0.07)
plt.savefig('SH_04_SIT_obs.pdf')

#Figure SUPLEMENT 2
fig, im = pmf.plot_pannels_month('04', ds_NH, clim_NH, 'sit_interp', -1.5, 1.5)
axes = fig.axes

cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.015])
fig.colorbar(im, cax=cbar_ax, label = 'Sea Ice Thickness anomaly (m)', orientation='horizontal')
plt.subplots_adjust(left=0.01, bottom=0.07, right=0.99, top=0.975, wspace=0.01, hspace=0.07)
plt.savefig('NH_04_SIT_obs.pdf')

#Figure SUPLEMENT 1

fig, im = pmf.plot_pannels_month('10', ds_NH, clim_NH, 'sit_interp', -1.5, 1.5)
axes = fig.axes

cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.015])
fig.colorbar(im, cax=cbar_ax, label = 'Sea Ice Thickness anomaly (m)', orientation='horizontal')
plt.subplots_adjust(left=0.01, bottom=0.07, right=0.99, top=0.975, wspace=0.01, hspace=0.07)

plt.savefig('NH_10_SIT_obs.pdf')
plt.show()
