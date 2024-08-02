"""
Created by Marion Bocquet
Date : 15/05/2024
Credits : LEGOS/CNES/CLS

The script aims to plot the time series and create maps for uncertainties and sea ice thickness distribution
"""

import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import plots_MAPS_function as pmf
import cmcrameri.cm as cmc
from matplotlib import rcParams

def plot_stat_sat(nsat, dataset_sat, hem):
    """
    The function plot the several stats for the given dataset
    :param nsat: Nb of columns of the plot in the figure
    :param dataset_sat: the xarray dataset that contains the several 'params'
    :param hem: hemisphere
    :return: figure, im, im_std
    """
    params = ['STAT_sea_ice_thickness_q05', 'STAT_sea_ice_thickness_q1', 'STAT_sea_ice_thickness_median', 'STAT_sea_ice_thickness_q3', 'STAT_sea_ice_thickness_q95', 'STAT_sea_ice_thickness_mean', 'STAT_sea_ice_thickness_std']
    vmax = [5, 5, 5, 5, 5, 5, 1]
    for s in range(1,8):
        ax_temp = fig_all.add_subplot(7, 6, (s-1)*6+nsat)
        if s != 7:
            im = pmf.plot_map_no_cbar(dataset_sat[params[(s-1)]], hem, 0, vmax[s-1], cmc.roma_r)
        else:
            if hem==-1:
                im_std = pmf.plot_map_no_cbar(dataset_sat[params[(s-1)]], hem, 0, 2, cmc.roma_r)
            else:
                im_std = pmf.plot_map_no_cbar(dataset_sat[params[(s-1)]], hem, 0, vmax[s-1], cmc.roma_r)
    return(fig_all, im, im_std)

path = ''

# LOAD DATA
c2_nh = xr.open_dataset(path + 'NH/CryoSat2/SIT_nh_c2esaD1_201101_201112.nc').sel(time='2011-04-15')
c2_sh = xr.open_dataset(path + 'SH/CryoSat2/SIT_sh_c2esaD1_201101_201112.nc').sel(time='2011-09-15')

env1_sh = xr.open_dataset(path + 'SH/Envisat/SIT_sh_env3_201101_201112.nc').sel(time='2011-09-15')
env1_nh = xr.open_dataset(path + 'NH/Envisat/SIT_nh_env3_201101_201112.nc').sel(time='2011-04-15')

env2_sh = xr.open_dataset(path + 'SH/Envisat/SIT_sh_env3_200205_200212.nc').sel(time='2002-09-15')
env2_nh = xr.open_dataset(path + 'NH/Envisat/SIT_nh_env3_200301_200312.nc').sel(time='2003-04-15')

ers21_nh = xr.open_dataset(path + 'NH/ERS2/SIT_nh_ers2r_200301_200304.nc').sel(time='2003-04-15')
ers21_sh = xr.open_dataset(path + 'SH/ERS2/SIT_sh_ers2r_200201_200212.nc').sel(time='2002-09-15')

ers22_nh = xr.open_dataset(path + 'NH/ERS2/SIT_nh_ers2r_199601_199612.nc').sel(time='1996-04-15')
ers22_sh = xr.open_dataset(path + 'SH/ERS2/SIT_sh_ers2r_199505_199512.nc').sel(time='1995-09-15')

ers1_nh = xr.open_dataset(path + 'NH/ERS1/SIT_nh_ers1r_199601_199604.nc').sel(time='1996-04-15')
ers1_sh = xr.open_dataset(path + 'SH/ERS1/SIT_sh_ers1r_199501_199512.nc').sel(time='1995-09-15')


fig_all = plt.figure(figsize=(12, 12))
sns.set_style("whitegrid")
fig_all, im, im_std = pmf.plot_stat_sat(1, c2_nh, 1)
fig_all, im, im_std = pmf.plot_stat_sat(2, env1_nh, 1)
fig_all, im, im_std = pmf.plot_stat_sat(3, env2_nh, 1)
fig_all, im, im_std = pmf.plot_stat_sat(4, ers21_nh, 1)
fig_all, im, im_std = pmf.plot_stat_sat(5, ers22_nh, 1)
fig_all, im, im_std = pmf.plot_stat_sat(6, ers1_nh, 1)

axes = fig_all.axes
fig_all.colorbar(im, ax = list((axes[0], axes[7], axes[14], axes[21], axes[28], axes[35])), label='SIT (m)')
#fig_all.colorbar(im, ax = list((axes[1], axes[8], axes[15], axes[22], axes[29], axes[36])), label='SIT (m)')
#fig_all.colorbar(im, ax = list((axes[2], axes[9], axes[16], axes[23], axes[30], axes[37])), label='SIT (m)')
#fig_all.colorbar(im, ax = list((axes[3], axes[10], axes[17], axes[24], axes[31], axes[38])), label='SIT (m)')
#fig_all.colorbar(im, ax = list((axes[4], axes[11], axes[18], axes[25], axes[32], axes[39])), label='SIT (m)')
#fig_all.colorbar(im, ax = list((axes[5], axes[12], axes[19], axes[26], axes[33], axes[40])), label='SIT (m)')
fig_all.colorbar(im_std, ax = list((axes[6], axes[13], axes[20], axes[27], axes[34], axes[41])), label='SIT (m)')

axes[0].set_title('CryoSat-2')
axes[0].set_ylabel('5th percentile')
axes[1].set_ylabel('25th percentile')
axes[2].set_ylabel('Median')
axes[3].set_ylabel('75th percentile')
axes[4].set_ylabel('95th percentile')
axes[5].set_ylabel('Mean')
axes[6].set_ylabel('Standard Deviation')
axes[7].set_title('Envisat')

axes[14].set_title('Envisat')
axes[21].set_title('ERS-2')
axes[28].set_title('ERS-2')
axes[35].set_title('ERS-1')
axes[7].annotate('2011/04', (-0.1,1.27), xycoords=axes[7].transAxes, horizontalalignment = 'center', fontsize=13)
axes[21].annotate('2003/04', (-0.1,1.27), xycoords=axes[21].transAxes, horizontalalignment = 'center', fontsize=13)
axes[35].annotate('1996/04', (-0.1,1.27), xycoords=axes[35].transAxes, horizontalalignment = 'center', fontsize=13)
rcParams['savefig.bbox'] = 'tight'
plt.savefig('NH_SIT_unc.pdf')


fig_all = plt.figure(figsize=(12, 12))
sns.set_style("whitegrid")
fig_all, im, im_std = pmf.plot_stat_sat(1, c2_sh, -1)
fig_all, im, im_std = pmf.plot_stat_sat(2, env1_sh, -1)
fig_all, im, im_std = pmf.plot_stat_sat(3, env2_sh, -1)
fig_all, im, im_std = pmf.plot_stat_sat(4, ers21_sh, -1)
fig_all, im, im_std = pmf.plot_stat_sat(5, ers22_sh, -1)
fig_all, im, im_std = pmf.plot_stat_sat(6, ers1_sh, -1)

axes = fig_all.axes
fig_all.colorbar(im, ax = list((axes[0], axes[7], axes[14], axes[21], axes[28], axes[35])), label='SIT (m)')
fig_all.colorbar(im, ax = list((axes[1], axes[8], axes[15], axes[22], axes[29], axes[36])), label='SIT (m)')
fig_all.colorbar(im, ax = list((axes[2], axes[9], axes[16], axes[23], axes[30], axes[37])), label='SIT (m)')
fig_all.colorbar(im, ax = list((axes[3], axes[10], axes[17], axes[24], axes[31], axes[38])), label='SIT (m)')
fig_all.colorbar(im, ax = list((axes[4], axes[11], axes[18], axes[25], axes[32], axes[39])), label='SIT (m)')
fig_all.colorbar(im, ax = list((axes[5], axes[12], axes[19], axes[26], axes[33], axes[40])), label='SIT (m)')
fig_all.colorbar(im_std, ax = list((axes[6], axes[13], axes[20], axes[27], axes[34], axes[41])), label='SIT (m)')

axes[0].set_title('CryoSat-2')

axes[0].set_ylabel('5th percentile')
axes[1].set_ylabel('25th percentile')
axes[2].set_ylabel('Median')
axes[3].set_ylabel('75th percentile')
axes[4].set_ylabel('95th percentile')
axes[5].set_ylabel('Mean')
axes[6].set_ylabel('Standard Deviation')

axes[7].set_title('Envisat')
axes[7].annotate('2011/09', (-0.1,1.27), xycoords=axes[7].transAxes, horizontalalignment = 'center', fontsize=13)
axes[21].annotate('2002/09', (-0.1,1.27), xycoords=axes[21].transAxes, horizontalalignment = 'center', fontsize=13)
axes[35].annotate('1995/09', (-0.1,1.27), xycoords=axes[35].transAxes, horizontalalignment = 'center', fontsize=13)

axes[14].set_title('Envisat')
axes[21].set_title('ERS-2')
axes[28].set_title('ERS-2')
axes[35].set_title('ERS-1')
rcParams['savefig.bbox'] = 'tight'

plt.savefig('SH_SIT_unc.pdf')

