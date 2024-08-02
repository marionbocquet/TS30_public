"""
Created by Marion Bocquet
Date 15/05/2024
Credits : LEGOS/CNES/CLS

This script make the figure 2, ASPECT Validation

"""
import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import cmcrameri as cmc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import plots_MAPS_function as pmf
def stat(datax, datay, ax, px=0.05, py=0.15):
    """

    :param datax:
    :param datay:
    :param ax: ax on wich the stats have to be printed
    :param px: position x along ax axis
    :param py: position y alonf ax axis
    :return: stat box on the axis
    """
    rmse = ((datax - datay) ** 2).mean() ** .5
    corr_coef = datax.corr(datay, method='pearson')
    dif = datax-datay
    props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.85)
    ax.text(px, py, ' N = %s\n Bias = %.3f\n Med = %.3f \n SD = %.3f\n RMSE = %.3f\n $r$ = %.3f'%(datax.shape[0], dif.mean(), dif.median(), dif.std(), rmse, corr_coef), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, bbox=props, fontsize=8)


ds_ers1 = pd.read_csv('ASPECT/aspect_ASPeCT_ers1rSH_corr_interp_sd_climato_AMSR-AMSR2.csv')
ds_ers2 = pd.read_csv('ASPECT/aspect_ASPeCT_ers2rSH_corr_interp_sd_climato_AMSR-AMSR2.csv')
ds_env3 = pd.read_csv('ASPECT/aspect_ASPeCT_env3SH_corr_interp_sd_climato_AMSR-AMSR2.csv')
ds_c2 = pd.read_csv('ASPECT/aspect_ASPeCT_c2esaDSH1_SARpIN_interp_sd_climato_AMSR-AMSR2.csv')

map_c2 = xr.open_dataset('ASPECT/aspect_ASPeCT_sit_c2esaDSH1_SARpIN_interp_sit_sd_climato_AMSR-AMSR2_snow_depth.nc')
map_ers2 = xr.open_dataset('ASPECT/aspect_ASPeCT_sit_ers2rSH_corr_interp_sit_sd_climato_AMSR-AMSR2_snow_depth.nc')
map_ers1 = xr.open_dataset('ASPECT/aspect_ASPeCT_sit_ers1rSH_corr_interp_sit_sd_climato_AMSR-AMSR2_snow_depth.nc')
map_env3 = xr.open_dataset('ASPECT/aspect_ASPeCT_sit_env3SH_corr_interp_sit_sd_climato_AMSR-AMSR2_snow_depth.nc')

fig_all = plt.figure(figsize=(8, 7))
sns.set_style('whitegrid')

plt.subplot(224)
sns.set_style('whitegrid')

pmf.plot_map_no_cbar(map_c2['__xarray_dataarray_variable__'].values[1]-map_c2['__xarray_dataarray_variable__'].values[0], -1, -1.2, 1.2, 'cmc.vik')

diff_c2 = ds_c2.where((ds_c2['ASPeCT_sit']<10) & (ds_c2['ASPeCT_sit']>-10))['c2esaDSH1_SARpIN_interp_sit']-ds_c2.where((ds_c2['ASPeCT_sit']<10)&(ds_c2['ASPeCT_sit']>-10))['ASPeCT_sit']
ax1 = plt.gca()
ax1.set_title('(a)', loc='left')
ax1.set_title('CryoSat-2')
stat(ds_c2.where((ds_c2['ASPeCT_sit']<10) & (ds_c2['ASPeCT_sit']>-10))['c2esaDSH1_SARpIN_interp_sit'], ds_c2.where((ds_c2['ASPeCT_sit']<10)&(ds_c2['ASPeCT_sit']>-10))['ASPeCT_sit'], ax1)
sns.set_style('whitegrid')

plt.subplot(223)
pmf.plot_map_no_cbar(map_env3['__xarray_dataarray_variable__'].values[1]-map_env3['__xarray_dataarray_variable__'].values[0], -1, -1.2, 1.2, 'cmc.vik')
diff_env3 = ds_env3['env3SH_corr_interp_sit']-ds_env3['ASPeCT_sit']
ax2 = plt.gca()
ax2.set_title('(c)', loc='left')
ax2.set_title('Envisat')
stat(ds_env3['env3SH_corr_interp_sit'], ds_env3['ASPeCT_sit'], ax2)
sns.set_style('whitegrid')

plt.subplot(222)
pmf.plot_map_no_cbar(map_ers2['__xarray_dataarray_variable__'].values[1]-map_ers2['__xarray_dataarray_variable__'].values[0], -1, -1.2, 1.2, 'cmc.vik')
diff_ers2r = ds_ers2['ers2rSH_corr_interp_sit']-ds_ers2['ASPeCT_sit']
ax3 = plt.gca()
ax3.set_title('(b)', loc='left')
ax3.set_title('ERS-2')
stat(ds_ers2['ers2rSH_corr_interp_sit'], ds_ers2['ASPeCT_sit'], ax3)
sns.set_style('whitegrid')

plt.subplot(221)
im = pmf.plot_map_no_cbar(map_ers1['__xarray_dataarray_variable__'].values[1]-map_ers1['__xarray_dataarray_variable__'].values[0], -1, -1.2, 1.2, 'cmc.vik')
diff_ers1 = ds_ers1['ers1rSH_corr_interp_sit']-ds_ers1['ASPeCT_sit']
ax4 = plt.gca()
ax4.set_title('(a)', loc='left')
ax4.set_title('ERS-1')

stat(ds_ers1['ers1rSH_corr_interp_sit'], ds_ers1['ASPeCT_sit'], ax4)

cbar_ax = fig_all.add_axes([0.87, 0.03, 0.03, 0.94])
fig_all.colorbar(im, cax=cbar_ax, label = 'Sea Ice Thickness difference (Sat-ASPeCt) (m)', orientation='vertical')
plt.subplots_adjust(left=0.05, bottom=0.02, right=0.85, top=0.98, wspace=0.0351, hspace=0.03)
plt.savefig('ASPECT.pdf')
plt.show()
