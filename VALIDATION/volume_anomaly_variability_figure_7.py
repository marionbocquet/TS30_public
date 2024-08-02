"""
Created by Marion Bocquet
Date 15/05/2024
Credits : LEGOS/CNES/CLS

This script make the figure 7 volume anomaly decomposition

"""
import matplotlib.ticker as tck
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import seaborn as sns
import plots_MAPS_function as pmf


# LOAD NH
print('ds_NH')
ds_NH = xr.open_dataset('/Users/marionbocquet/PycharmProjects/TS30/data/NH_W99m/unc_all_sats_NH_199301_202304.zarr', engine="zarr", drop_variables=['FBi_AMSR2_NSIDC'])
ds_NH_red = ds_NH[['sit_interp', 'ice_conc', 'common_mask', 'volume_interp']]
print('ds is readed')
ds_NH_red = ds_NH_red.where((ds_NH_red.sit_interp>=0)&(ds_NH_red.ice_conc>=50)&(ds_NH_red.common_mask==0)).fillna(0)
ds_SH = xr.open_dataset('/Users/marionbocquet/PycharmProjects/TS30/data/all_sats_SH_199301_202306.zarr', engine="zarr")
ds_SH_red = ds_SH[['sit_interp', 'ice_conc', 'common_mask', 'volume_interp']]
ds_SH_red = ds_SH_red.where((ds_SH_red.sit_interp>=0)&(ds_SH_red.ice_conc>=50)&(ds_SH_red.common_mask==0)).fillna(0)

print('NH')

dataset_pos, df_volume_clim, df_volume_anomaly, ds_volume, df_volume,df_volume_av_av_clim, df_sitc_sicc, df_sitv_sicv, df_sitc_sicv, df_sitv_sicc = pmf.decomposed_obs_true_sit(ds_NH_red, 'volume_interp', 'ice_conc', 'sit_interp', '199410', '202304')
print('NH is done')

print('SH')
dataset_pos_sh, df_volume_clim_sh, df_volume_anomaly_sh, ds_volume_sh, df_volume_sh, df_volume_av_av_clim_sh, df_sitc_sicc_sh, df_sitv_sicv_sh, df_sitc_sicv_sh, df_sitv_sicc_sh = pmf.decomposed_obs_true_sit(ds_SH_red, 'volume_interp', 'ice_conc', 'sit_interp', '199405', '202306')
print('SH is done')

fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(12, 6))
sns.set_style("ticks")
sns.despine()
plt.tight_layout()
sns.set_style("ticks")
sns.despine()

ax1.plot(df_volume_anomaly_sh, label='Volume anomaly', color='k', linewidth = 1.5)

sns.set_style("ticks")
sns.despine()
ax1.plot(df_sitv_sicc_sh, label= r"$\int SIT' \cdot \overline{SIC} da$", linewidth = 1, color='#c44536')
ax1.plot(df_sitc_sicv_sh, label= r"$\int \overline{SIT} \cdot SIC' da$", linewidth = 1, color='#197278')
ax1.plot(df_sitv_sicv_sh, label= r"$\int SIT' \cdot SIC' da$", linewidth = 1, color='orange')
ax1.plot(df_sitc_sicc_sh-df_volume_clim_sh, label=r"$\int \overline{SIT} \cdot \overline{SIC} da - \overline{\int SIT \cdot SIC da}$", linewidth = 0.7, linestyle='--', color='#283d3b')
ax1.set_ylabel('Volume ($km^3$)')
ax1.set_xlabel('Year')
ax1.set_ylim(ymin=-4000, ymax=5700)
ax1.axhline(0, lw=0.5, color='k',zorder=0)
ax1.axhline(0, lw=0.5, color='k',zorder=0)
ax1.axvline(np.datetime64('2013-10-15'), lw=0.5, color='teal',zorder=0, ymin=0, ymax=0.97) #-> max 3 6/5/7/8
ax1.axvline(np.datetime64('2014-09-15'), lw=0.5, color='teal',zorder=0, ymin=0, ymax=0.97) #-> max 3 6/5/7/8
ax1.axvline(np.datetime64('2015-06-15'), lw=0.5, color='teal',zorder=0, ymin=0, ymax=0.97) #-> max 3 6/5/7/8
ax1.axvline(np.datetime64('2016-10-15'), lw=0.5, color='darkred',zorder=0, ymin=0, ymax=0.97) #-> max 3 6/5/7/8
ax1.axvline(np.datetime64('2017-09-15'), lw=0.5, color='darkred',zorder=0, ymin=0, ymax=0.97) #-> max 3 6/5/7/8
ax1.axvline(np.datetime64('2019-07-15'), lw=0.5, color='darkred',zorder=0, ymin=0, ymax=0.97) #-> max 3 6/5/7/8
ax1.axvline(np.datetime64('2023-06-15'), lw=0.5, color='darkred',zorder=0, ymin=0, ymax=0.97) #-> max 3 6/5/7/8
ax1.axvline(np.datetime64('2022-12-15'), lw=0.5, color='darkred',zorder=0, ymin=0, ymax=0.97) #-> max 3 6/5/7/8


sns.set_style("ticks")
sns.despine()
sns.set_style("ticks")
sns.despine()
plt.tight_layout()
sns.set_style("ticks")
sns.despine()
ax2.plot(df_volume_anomaly, label='Volume anomaly', color='k', linewidth = 1.5)

sns.set_style("ticks")
sns.despine()
ax2.plot(df_sitv_sicc, label= r"$\int SIT' \cdot \overline{SIC} da$", linewidth = 1, color='#c44536')
ax2.plot(df_sitc_sicv, label= r"$\int \overline{SIT} \cdot SIC' da$", linewidth = 1, color='#197278')
ax2.plot(df_sitv_sicv, label= r"$\int SIT' \cdot SIC' da$", linewidth = 1, color='orange')
ax2.plot(df_sitc_sicc-df_volume_clim, label=r"$\int \overline{SIT} \cdot \overline{SIC} da - \overline{\int SIT \cdot SIC da}$", linewidth = 0.75, linestyle='--', color='#283d3b')
ax2.set_ylabel('Volume ($km^3$)')
ax2.set_xlabel('Year')
ax2.axhline(0, lw=0.5, color='k',zorder=0)
ax2.set_ylim(ymin=-2500, ymax=4000)


ax2.legend(frameon=False, ncol=5)
ax1.annotate('2013-10', xy=(pd.to_datetime(np.datetime64('2013-09-01')), 5700), xycoords='data', rotation=50)
ax1.annotate('2014-09', xy=(pd.to_datetime(np.datetime64('2014-08-01')), 5700), xycoords='data', rotation=50)
ax1.annotate('2015-06', xy=(pd.to_datetime(np.datetime64('2015-05-01')), 5700), xycoords='data', rotation=50)
ax1.annotate('2016-09', xy=(pd.to_datetime(np.datetime64('2016-08-01')), 5700), xycoords='data', rotation=50)
ax1.annotate('2017-10', xy=(pd.to_datetime(np.datetime64('2017-09-01')), 5700), xycoords='data', rotation=50)
ax1.annotate('2019-07', xy=(pd.to_datetime(np.datetime64('2019-06-01')), 5700), xycoords='data', rotation=50)
ax1.annotate('2022-12', xy=(pd.to_datetime(np.datetime64('2022-11-01')), 5700), xycoords='data', rotation=50)
ax1.annotate('2023-06', xy=(pd.to_datetime(np.datetime64('2023-05-01')), 5700), xycoords='data', rotation=50)


ax1.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax1.xaxis.set_minor_locator(tck.AutoMinorLocator())

ax2.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax2.xaxis.set_minor_locator(tck.AutoMinorLocator())

plt.tight_layout()

ax2.set_title('(a)', loc='left')
ax1.set_title('(b)', loc='left')
plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.07)
plt.savefig('Anomaly_regions_reynolds.pdf')

plt.show()