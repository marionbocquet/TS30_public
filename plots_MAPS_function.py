"""
Created by Marion Bocquet
Date : 15/04/2024
Credits : LEGOS/CNES/CLS
The script aims to plot the time series and create pdfs
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
import plots_TS_function as ptf

def plot_map_no_cbar(data, fact, vmin, vmax, ccolor, title=None):
    """
    Plot maps without colorbar
    :param data: Array to plot
    :param fact: hemisphere : -1=SH and 1:NH
    :param vmin: min value
    :param vmax: max value
    :param ccolor: colormar
    :param title: plot title
    :return:
    """
    MAP_PARAM_DESC = {
        "projection": "laea",
        "lat_ts": 0.,
        "lon_0": 0,
        "lat_0": 90,
        "resolution": "l",
    }

    mapframe = Basemap(projection=MAP_PARAM_DESC['projection'],
                       lat_0=MAP_PARAM_DESC['lat_0'] * fact,
                       lat_ts=MAP_PARAM_DESC['lat_ts'],
                       lon_0=MAP_PARAM_DESC['lon_0'],
                       width=data.shape[1] * 12500,
                       height=data.shape[0] * 12500,
                       resolution=MAP_PARAM_DESC['resolution'],
                       round=False)

    # draw coastline, meridians, parallels, ...
    mapframe.fillcontinents(color='lightgray')
    mapframe.drawparallels(np.arange(-90, 90, 20), linewidth=0.4, color="gray")
    mapframe.drawmeridians(np.arange(0, 360, 30), linewidth=0.4, color="gray")
    im = mapframe.imshow(np.flipud(data.squeeze()), cmap=ccolor, vmin=vmin, vmax=vmax)
    plt.title(title)
    return(im)

def plot_map(data, fact, vmin, vmax, cmap, label, title=None, orientation = 'vertical'):
    """
    Same as for plot_map_no_cbar
    :param data: ""
    :param fact: ""
    :param vmin: ""
    :param vmax: ""
    :param cmap: ""
    :param label: label of the colorbar
    :param title: ""
    :return:
    """
    MAP_PARAM_DESC = {
        "projection": "laea",
        "lat_ts": 0.,
        "lon_0": 0,
        "lat_0": 90,
        "resolution": "l",
    }

    mapframe = Basemap(projection=MAP_PARAM_DESC['projection'],
                       lat_0=MAP_PARAM_DESC['lat_0'] * fact,
                       lat_ts=MAP_PARAM_DESC['lat_ts'],
                       lon_0=MAP_PARAM_DESC['lon_0'],
                       width=data.shape[1] * 12500,
                       height=data.shape[0] * 12500,
                       resolution=MAP_PARAM_DESC['resolution'],
                       round=False)

    # draw coastline, meridians, parallels, ...
    mapframe.fillcontinents(color='lightgray')
    mapframe.drawparallels(np.arange(-90, 90, 20), linewidth=0.4, color="gray")
    mapframe.drawmeridians(np.arange(0, 360, 30), linewidth=0.4, color="gray")
    im = mapframe.imshow(np.flipud(data.squeeze()), cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(orientation=orientation)
    cbar.set_label('%s' % label)
    plt.title(title)

def plot_map_significance_no_cbar(dataset, data, pval, fact, vmin, vmax, title=None):
    """
    Plots maps of trends with significance (test result)
    :param dataset: xarray dataset to get lon and lat values
    :param pval: array to get the shape of the grid
    :param fact: hemisphere -1:SH and 1:NH
    :param vmin: min value
    :param vmax: max value
    :param title: title of the plot
    :return:
    """
    MAP_PARAM_DESC = {
        "projection": "laea",
        "lat_ts": 0.,
        "lon_0": 0,
        "lat_0": 90,
        "resolution": "l",
    }

    mapframe = Basemap(projection=MAP_PARAM_DESC['projection'],
                       lat_0=MAP_PARAM_DESC['lat_0'] * fact,
                       lat_ts=MAP_PARAM_DESC['lat_ts'],
                       lon_0=MAP_PARAM_DESC['lon_0'],
                       width=data.shape[1] * 12500,
                       height=data.shape[0] * 12500,
                       resolution=MAP_PARAM_DESC['resolution'],
                       round=False)

    # draw coastline, meridians, parallels, ...
    mapframe.fillcontinents(color='lightgray')
    lon_m, lat_m = mapframe(dataset.lon.values, dataset.lat.values)
    mapframe.drawparallels(np.arange(-90, 90, 20), linewidth=0.4, color="gray")
    mapframe.drawmeridians(np.arange(0, 360, 30), linewidth=0.4, color="gray")
    im = mapframe.imshow(np.flipud(data.squeeze()), cmap='RdBu_r', vmin=vmin, vmax=vmax)
    levels = [0, 0.95, 1]
    si = mapframe.contourf(lon_m, lat_m, pval.squeeze(), levels=levels, hatches=["", "///"], alpha=0)
    plt.title(title)
    return(im)
def trend_month_map_year(dataset, param, year1, year2):
    """

    :param dataset: dataset
    :param param: param to plot in dataset (int)
    :param year1: first year of the TS
    :param year2: last year of the TS
    :return:
    """

    dataset = dataset.sel(year=((dataset.year >= year1) & (dataset.year <= year2)))
    dataset_sel = dataset.fillna(0)

    time_array = dataset_sel.year.data.reshape(-1, 1)

    def trend_map(data_array, time_array):
        # X = ds.time.dt.year.data.reshape(-1,1)
        # time_array = data_array.time.dt.year.data.reshape(-1, 1)
        if np.isnan(data_array.sum()):
            aa = np.nan
            hyp = False
            p_val = np.nan

        else:
            res = stats.theilslopes(data_array, time_array)
            aa = res[0]
            bb = res[1]
            yr = aa * time_array + bb
            test = mk.original_test(data_array, alpha=0.05)
            hyp = test.h
            p_val = test.p
        # print("Test for to obs trend at 99p9 percent:", '\n', test)

        return (aa, hyp, p_val)

    print('Trend computing in progress along all the matrix')

    trend_m, res_m, p_value_m = np.apply_along_axis(trend_map, 0, dataset_sel[param].values,
                                                    dataset_sel.year.data.reshape(-1, 1))
    return (trend_m, res_m, p_value_m)


def trend_month_map(m, dataset, param, year1, year2):
    """

    :param m: Month of the year (string)
    :param dataset: xarray dataset
    :param param: param to compute the trend
    :param year1: first year of the Time series to be considered
    :param year2: last year of the time series to be considered
    :return:
    """
    dataset = dataset.sel(time=((dataset.time.dt.year >= year1) & (dataset.time.dt.year <= year2)))
    mask = dataset.sel(time=(dataset['time.month'] == int(m))).mean(dim=['time'])
    dataset_sel = dataset.fillna(0).sel(time=(dataset['time.month'] == int(m)))
    dataset_sel = dataset_sel.where(~mask.isnull())
    time_array = dataset_sel.time.dt.year.data.reshape(-1, 1)

    def trend_map(data_array, time_array):
        if np.isnan(data_array.sum()):
            aa = np.nan
            hyp = False
            p_val = np.nan

        else:
            res = stats.theilslopes(data_array, time_array)
            aa = res[0]
            bb = res[1]
            yr = aa * time_array + bb
            test = mk.original_test(data_array, alpha=0.05)
            hyp = test.h
            p_val = test.p
        # print("Test for to obs trend at 99p9 percent:", '\n', test)

        return (aa, hyp, p_val)

    print('Trend computing in progress along all the matrix')
    trend_m, res_m, p_value_m = np.apply_along_axis(trend_map, 0, dataset_sel[param].values, dataset_sel.time.dt.year.data.reshape(-1, 1))
    return (trend_m, res_m, p_value_m)

def clim_trends_year(ds_NH_time_changes, ds_SH_time_changes, ds_NH, ds_SH, ds_NH_0, ds_SH_0):
    """

    :param ds_NH_time_changes: Dataset NH with the adapted time
    :param ds_SH_time_changes: Dataset SH with the adapted time
    :param ds_NH: Dataset NH
    :param ds_SH: Dataset SH
    :param ds_NH_0: Dataset NH replaced nan by 0
    :param ds_SH_0: Dataset S replaced nan by 0
    :return: None
    """
    print('Let us compute yearly trends')
    if os.path.isfile('NH_pvaly.npy'):
        nhtrendy = np.load('NH_trendy.npy')
        nhpvaly = np.load('NH_pvaly.npy')
        shtrendy = np.load('SH_trendy.npy')
        shpvaly = np.load('SH_pvaly.npy')
        print('nh y exist !')
    else:
        timeynh = time.time()
        nhtrendy, nhsign_masky, nhpvaly = trend_month_map_year(ds_NH_time_changes.groupby('time.year').mean(dim='time'),
                                                               'sit_interp', 1994, 2023)
        nhtrendy[ds_NH.mask_acq.values == 0] = np.nan

        print((time.time() - timeynh) / 60)
        print('NH trend computed')
        timeysh = time.time()
        shtrendy, shsign_masky, shpvaly = trend_month_map_year(ds_SH_time_changes.groupby('time.year').mean(dim='time'),
                                                               'sit_interp', 1994, 2023)
        shtrendy[ds_SH.mask_acq.values == 0] = np.nan
        print((time.time() - timeysh) / 60)
        np.save('NH_pvaly.npy', nhpvaly)
        np.save('NH_trendy.npy', nhtrendy)
        np.save('SH_pvaly.npy', shpvaly)
        np.save('SH_trendy.npy', shtrendy)
        print('SH trend computed')

    sns.set_style('whitegrid')
    plt.figure(); plot_map_significance_no_cbar(ds_NH, nhtrendy*10, 1-nhpvaly, 1, -0.6, 0.6)
    plt.savefig('SIT_year_trend_NH.pdf')
    plt.figure(); plot_map_significance_no_cbar(ds_SH, shtrendy*10, 1-shpvaly, -1, -0.6, 0.6)
    plt.savefig('SIT_year_trend_SH.pdf')



    plt.figure(); plot_map(ds_NH_0.sit_interp.mean(dim='time').where(ds_NH.ice_conc.mean(dim='time')>=1), 1, 0, 4, cmc.roma_r, 'Sea ice thickness (m)')
    plt.savefig('SIT_MEAN_NH.pdf')

    plt.figure(); plot_map(ds_SH_0.sit_interp.mean(dim='time').where(ds_SH.ice_conc.mean(dim='time')>=1), -1, 0, 4, cmc.roma_r, 'Sea ice thickness (m)')
    plt.savefig('SIT_MEAN_SH.pdf')
def plot_pannels_month(m, dataset, clim, param, vmin, vmax):

    """

    :param m: month (int)
    :param dataset: dataset (xarray dataset)
    :param clim: dataset climato
    :param param: param in dataset to plot
    :param vmin: value min
    :param vmax: values max
    :return:
    """

    id_month = np.where(dataset.time.dt.month == int(m))
    list_date = dataset.time.data[id_month[0]]
    if dataset.lat.values.max() > 0:
        hem = 1
    else:
        hem = -1
    fig_all = plt.figure(figsize=(12, 12))
    sns.set_style("whitegrid")
    years = pd.to_datetime(list_date).year.values
    props = dict(boxstyle='round', facecolor='white', alpha=0.85)
    for d in range(len(list_date)):
        if hem == 1:
            ax_temp = fig_all.add_subplot(6, 5, (d + 1))
        else:
            ax_temp = fig_all.add_subplot(6, 5, (d + 1))
        im = plot_map_no_cbar(dataset.sel(time=list_date[d])[param] - clim.sel(month=int(m))[param], hem, vmin, vmax, cmc.vik)
        ax_temp.text(0.05, 0.05, '%s' % (str(years[d])), transform=ax_temp.transAxes, fontsize=11, bbox=props)
    return(fig_all, im)



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

def decomposed_obs_true_sit(datasetm, volume_var, ice_conc_var, sit_var, start_date, end_date):
    """

    :param datasetm:
    :param volume_var:
    :param ice_conc_var:
    :param sit_var:
    :param start_date:
    :param end_date:
    :return:
    """
    datasetm = datasetm.assign(area=datasetm[ice_conc_var] * (12500 ** 2) * 1e-9 * 0.01)
    groupped_by_month = datasetm.groupby('time.month')
    clim = groupped_by_month.mean('time')
    anom = groupped_by_month - clim

    # datasetm = datasetm.assign(volume_sitv_sicc = (('time', 'y', 'x'), datasetm[sit_var].groupby("time.month")*clim.area))
    # datasetm = datasetm.assign(volume_sitc_sicv = clim[sit_var]*datasetm['area'].groupby("time.month"))
    datasetm = datasetm.assign(volume_anomsitv_sicc=anom[sit_var].groupby("time.month") * clim.area)
    datasetm = datasetm.assign(volume_sitc_anomsicv=clim[sit_var] * anom.area.groupby("time.month"))
    datasetm = datasetm.assign(volume_anomsitv_anomsicv=anom[sit_var] * anom.area)
    datasetm = datasetm.assign(volume_sitc_sicc=clim[sit_var] * clim.area)
    selector = xr.DataArray(datasetm.time.dt.month, dims=["time"], coords=[datasetm.time])
    datasetm = datasetm.assign(
        volume_sitc_sicc=(datasetm.volume_sitc_sicc.sel(month=selector)))  # allows to repeat values
    print('OK for volumes clim and anom computing')

    print('df_sitv_sicc')
    df_sitv_siccm = datasetm.volume_anomsitv_sicc.sum(dim=["x", "y"]).to_dataframe()
    df_sitv_siccm = ptf.fill_df_nan_anom(df_sitv_siccm, start_date, end_date).volume_anomsitv_sicc

    print('df_sitc_sicv')
    df_sitc_sicvm = datasetm.volume_sitc_anomsicv.sum(dim=["x", "y"]).to_dataframe()
    df_sitc_sicvm = ptf.fill_df_nan_anom(df_sitc_sicvm, start_date, end_date).volume_sitc_anomsicv

    print('df_sitv_sicv')
    df_sitv_sicvm = datasetm.volume_anomsitv_anomsicv.sum(dim=["x", "y"]).to_dataframe()
    df_sitv_sicvm = ptf.fill_df_nan_anom(df_sitv_sicvm, start_date, end_date).volume_anomsitv_anomsicv

    print('df_sitc_sicc')
    df_sitc_siccm = datasetm.volume_sitc_sicc.sum(dim=["x", "y"]).to_dataframe()
    df_sitc_siccm = ptf.fill_df_nan_anom(df_sitc_siccm, start_date, end_date).volume_sitc_sicc

    ds_volume_av_avm = datasetm.volume_anomsitv_anomsicv.sum(dim=['x', 'y'])
    df_volume_av_av_climm = ds_volume_av_avm.groupby('time.month').mean('time')
    df_volume_av_av_climm = df_volume_av_av_climm.sel(month=selector).to_dataframe()
    df_volume_av_av_climm = ptf.fill_df_nan_anom(df_volume_av_av_climm, start_date, end_date).volume_anomsitv_anomsicv

    print('ds_volume')
    ds_volumem = datasetm[volume_var].sum(dim=['x', 'y'])
    df_volumem = ds_volumem.to_dataframe()
    df_volumem = ptf.fill_df_nan_anom(df_volumem, start_date, end_date)[volume_var]

    print('df_volume_anomaly')
    df_volume_anomalym = (
                ds_volumem.groupby('time.month') - ds_volumem.groupby('time.month').mean('time')).to_dataframe()
    df_volume_anomalym = ptf.fill_df_nan_anom(df_volume_anomalym, start_date, end_date)[volume_var]

    print('df_volume_clim')
    df_volume_climm = ds_volumem.groupby('time.month').mean('time')
    df_volume_climm = df_volume_climm.sel(month=selector).to_dataframe()
    df_volume_climm = ptf.fill_df_nan_anom(df_volume_climm, start_date, end_date)[volume_var]

    return (datasetm, df_volume_climm,
            df_volume_anomalym, ds_volumem, df_volumem,
            df_volume_av_av_climm, df_sitc_siccm,
            df_sitv_sicvm, df_sitc_sicvm, df_sitv_siccm)
