"""
Created by Marion Bocquet
Date 15/05/2024
Credits : LEGOS/CNES/CLS

This script compares the snow depth products solutions with in situ sea ice thickness/draft -  Appendix number 2

"""
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import sys
import os
import matplotlib.pyplot as plt
import xarray as xr
import cmcrameri.cm as cmc                                                      
from mpl_toolkits.basemap import Basemap                                        
import plots_MAPS_function as pmf
def stat(datax, datay, ax, px=0.05, py=0.85):
    """

    :param datax:
    :param datay:
    :param ax: ax on wich the stats have to be printed
    :param px: position x along ax axis
    :param py: position y alonf ax axis
    :return: stat box on the axis
    """
    rmse = ((datay - datax) ** 2).mean() ** .5
    corr_coef = datax.corr(datay, method='pearson')
    dif = datay-datax
    props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.85)
    ax.text(px, py, ' N = %s\n Bias = %.3f\n Med = %.3f \n SD = %.3f\n RMSE = %.3f\n $r$ = %.3f'%(datax.shape[0], dif.mean(), dif.median(), dif.std(), rmse, corr_coef), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, bbox=props, fontsize=8)
    #plt.title('N = %s, Bias = %.3f, STD = %.3f\n, RMSE = %.3f, $R^2$ = %.3f'%(datax.shape[0], dif.mean(), dif.std(), rmse, corr_coef**2))


# There is 4 mooring locating in the fram strait
# F11 12 13 14

AWI_mooring_ers1 = pd.read_csv('VALIDATION/AWI/COMBINE_AWI_SH_ers1rSH_corr_interp.csv', names=['AWI', "sat", 'snow', "date"], header=0)
AWI_mooring_ers2 = pd.read_csv('VALIDATION/AWI/COMBINE_AWI_SH_ers2rSH_corr_interp.csv', names=['AWI', "sat", 'snow', "date"], header=0)
AWI_mooring_env3 = pd.read_csv('VALIDATION/AWI/COMBINE_AWI_SH_env3SH_corr_interp.csv', names=['AWI', "sat", 'snow', "date"], header=0)

AWI_mooring_ers1_AMSR_clim = pd.read_csv('VALIDATION/AWI/AMSR_clim_AWI_SH_ers1rSH_corr_interp.csv', names=['AWI', "sat", 'snow', "date"], header=0)
AWI_mooring_ers2_AMSR_clim = pd.read_csv('VALIDATION/AWI/AMSR_clim_AWI_SH_ers2rSH_corr_interp.csv', names=['AWI', "sat", 'snow', "date"], header=0)
AWI_mooring_env3_AMSR_clim = pd.read_csv('VALIDATION/AWI/AMSR_clim_AWI_SH_env3SH_corr_interp.csv', names=['AWI', "sat", 'snow', "date"], header=0)

AWI_mooring_ers1_ASD_clim = pd.read_csv('VALIDATION/AWI/ASD_clim_AWI_SH_ers1rSH_corr_interp.csv', names=['AWI', "sat", 'snow', "date"], header=0)
AWI_mooring_ers2_ASD_clim = pd.read_csv('VALIDATION/AWI/ASD_clim_AWI_SH_ers2rSH_corr_interp.csv', names=['AWI', "sat", 'snow', "date"], header=0)
AWI_mooring_env3_ASD_clim = pd.read_csv('VALIDATION/AWI/ASD_clim_AWI_SH_env3SH_corr_interp.csv', names=['AWI', "sat", 'snow', "date"], header=0)


TS = xr.open_zarr('../TS30/data/all_sats_SH_199301_202306.zarr')

AWI_mooring_ers1['time'] = pd.to_datetime(AWI_mooring_ers1['date'].astype('int').astype('str'), format="%Y%m")
AWI_mooring_ers2['time'] = pd.to_datetime(AWI_mooring_ers2['date'].astype('int').astype('str'), format="%Y%m")
AWI_mooring_env3['time'] = pd.to_datetime(AWI_mooring_env3['date'].astype('int').astype('str'), format="%Y%m")
AWI_mooring_ers1_AMSR_clim['time'] = pd.to_datetime(AWI_mooring_ers1_AMSR_clim['date'].astype('int').astype('str'), format="%Y%m")
AWI_mooring_ers2_AMSR_clim['time'] = pd.to_datetime(AWI_mooring_ers2_AMSR_clim['date'].astype('int').astype('str'), format="%Y%m")
AWI_mooring_env3_AMSR_clim['time'] = pd.to_datetime(AWI_mooring_env3_AMSR_clim['date'].astype('int').astype('str'), format="%Y%m")
AWI_mooring_ers1_ASD_clim['time'] = pd.to_datetime(AWI_mooring_ers1_ASD_clim['date'].astype('int').astype('str'), format="%Y%m")
AWI_mooring_ers2_ASD_clim['time'] = pd.to_datetime(AWI_mooring_ers2_ASD_clim['date'].astype('int').astype('str'), format="%Y%m")
AWI_mooring_env3_ASD_clim['time'] = pd.to_datetime(AWI_mooring_env3_ASD_clim['date'].astype('int').astype('str'), format="%Y%m")


cmap = plt.cm.get_cmap('inferno')
cmap_cut = cmap(np.arange(cmap.N))[:-40]
cmap = LinearSegmentedColormap.from_list('cut', cmap_cut, cmap.N)


fig = plt.figure('COMBINE scatter', figsize=(10,10))
sns.set_style('whitegrid')

ax1 = fig.add_subplot(3,3,1)
sns.set_style('whitegrid')

plt.scatter(AWI_mooring_ers2['AWI'], AWI_mooring_ers2['sat'], c = 'darkorange', vmin=0, vmax=1, label='ERS-2', marker='*')
plt.scatter(AWI_mooring_env3['AWI'], AWI_mooring_env3['sat'], c = 'darkred', vmin=0, vmax=1, marker='*', label='Envisat')
plt.scatter(AWI_mooring_ers1['AWI'], AWI_mooring_ers1['sat'], c = 'teal', vmin=0, vmax=1, label='ERS-1', marker='*')

x = np.linspace(-0.2, 4, 10)
ax1.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
ax1.set_ylabel('TS draft (m)')
ax1.set_xlabel('AWI draft (m)')
axes = plt.gca()
axes.set_aspect('equal', adjustable='box')
plt.plot(x, x, 'r')
plt.title('Weighted mean')
df_COMBINE = pd.concat([AWI_mooring_ers1, AWI_mooring_ers2, AWI_mooring_env3], ignore_index=False, axis=0)
stat(df_COMBINE['AWI'], df_COMBINE['sat'], ax1, px=0.05, py=0.85)

ax2 = fig.add_subplot(3,3,2)
sns.set_style('whitegrid')

plt.scatter(AWI_mooring_ers2_AMSR_clim['AWI'], AWI_mooring_ers2_AMSR_clim['sat'], c = 'darkorange', vmin=0, vmax=1, marker='*')#, label='ERS-2')
plt.scatter(AWI_mooring_env3_AMSR_clim['AWI'], AWI_mooring_env3_AMSR_clim['sat'], c = 'darkred', vmin=0, vmax=1, marker='*')#, label='Envisat')
plt.scatter(AWI_mooring_ers1_AMSR_clim['AWI'], AWI_mooring_ers1_AMSR_clim['sat'], c = 'teal', vmin=0, vmax=1, marker='*', label='ERS-1')#, marker='v')



x = np.linspace(-0.2, 4, 10)
ax2.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
ax2.set_ylabel('TS draft (m)')
ax2.set_xlabel('AWI draft (m)')

axes = plt.gca()
axes.set_aspect('equal', adjustable='box')
plt.plot(x, x, 'r')
df_AMSRW99 = pd.concat([AWI_mooring_ers1_AMSR_clim, AWI_mooring_ers2_AMSR_clim, AWI_mooring_env3_AMSR_clim], ignore_index=False, axis=0)
stat(df_AMSRW99['AWI'], df_AMSRW99['sat'], ax2, px=0.05, py=0.85)
plt.title('AMSR climatogoly snow depth')

ax3 = fig.add_subplot(3,3,3)
sns.set_style('whitegrid')

plt.scatter(AWI_mooring_ers2_ASD_clim['AWI'], AWI_mooring_ers2_ASD_clim['sat'], c = 'darkorange', vmin=0, vmax=1, marker='*')#, label='Envisat')
plt.scatter(AWI_mooring_env3_ASD_clim['AWI'], AWI_mooring_env3_ASD_clim['sat'], c = 'darkred', vmin=0, vmax=1, marker='*')#, label='CryoSat-2')
plt.scatter(AWI_mooring_ers1_ASD_clim['AWI'], AWI_mooring_ers1_ASD_clim['sat'], c = 'teal', vmin=0, vmax=1, marker='*')#, label='ERS-2')

x = np.linspace(-0.2, 4, 10)
ax3.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
ax3.set_ylabel('TS draft (m)')
ax3.set_xlabel('AWI draft (m)')
axes = plt.gca()
axes.set_aspect('equal', adjustable='box')
plt.plot(x, x, 'r')
df_LG = pd.concat([AWI_mooring_ers1_ASD_clim, AWI_mooring_ers2_ASD_clim, AWI_mooring_env3_ASD_clim], ignore_index=False, axis=0)
stat(df_LG['AWI'], df_LG['sat'], ax3, px=0.05, py=0.85)
plt.title('ASD climatology snow depth')



ax9 = fig.add_subplot(3, 3, 4)
sns.set_style('whitegrid')
im = pmf.plot_map(TS.sit_interp.mean(dim='time'), -1, 0, 5, cmc.roma_r, 'Sea ice thickness (m)', orientation='horizontal')

ax11 = fig.add_subplot(3, 3, 5)
sns.set_style('whitegrid')
pmf.plot_map(TS.sit_interp.mean(dim='time')-TS.SIT_AMSR_clim.mean(dim='time'), -1, -1, 1, cmc.vik, 'Sea ice thickness (m)', orientation='horizontal')

ax12 = fig.add_subplot(3, 3, 6)
sns.set_style('whitegrid')
pmf.plot_map(TS.sit_interp.mean(dim='time')-TS.SIT_ASD_clim.mean(dim='time'), -1, -1, 1, cmc.vik, 'Sea ice thickness (m)', orientation='horizontal')



ax13 = fig.add_subplot(3, 3, 7)
sns.set_style('whitegrid')
im_std = pmf.plot_map(TS.sit_interp.std(dim='time'), -1, 0, 2, cmc.davos, 'Sea ice thickness (m)', orientation='horizontal')

ax14 = fig.add_subplot(3, 3, 8)
sns.set_style('whitegrid')
im_nstd = pmf.plot_map(TS.SIT_AMSR_clim.std(dim='time')/TS.sit_interp.std(dim='time'), -1, 0.5, 1.5, cmc.cork, 'Sea ice thickness (m)', orientation='horizontal')

ax15 = fig.add_subplot(3, 3, 9)
sns.set_style('whitegrid')
pmf.plot_map(TS.SIT_ASD_clim.std(dim='time')/TS.sit_interp.std(dim='time'), -1, 0.5, 1.5, cmc.cork, 'Sea ice thickness (m)', orientation='horizontal')

from matplotlib import rcParams
rcParams['savefig.bbox'] = 'tight'
plt.savefig('insitu_improvement_cmap_SH_cbar.pdf')
plt.show()

