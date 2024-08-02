"""
Created by Marion Bocquet
Date 15/04/2024
Credit to CNES/LEGOS/CLS

This script gather the main functions used to plot the time series in several ways
"""

import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.ticker as tck
from matplotlib import gridspec
from mpl_toolkits.basemap import Basemap
import pymannkendall as mk
from scipy import stats

# Functions plots_volumes_stack_cat, plots_volumes_stack_cat_per, plot_pannels_per_cat_glob build the figure 8

def plots_volumes_stack_cat(df_vol, columns, labels, tcmap, title, ax, vmin, vmax):

    """
    The function plot for the concerned variable (column) a stackplot on the given ax
    :param df_vol: pandas dataframe containing the volumes,
    :param columns: name of the variable in df_vol to plot
    :param labels: Labels for the stackplot
    :param tcmap: The colormap
    :param title: Title of the plot
    :param ax: ax on wich to plot
    :param vmin: minimum value ylims
    :param vmax: maximum value for ylims
    :return:
    """

    cmap = plt.get_cmap(tcmap, 10)
    custom_palette = [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    df_vol = df_vol[columns]
    plt.xticks(rotation=45)
    ax.stackplot(df_vol.index, df_vol.T.values, labels=labels, alpha=0.9, colors=custom_palette, lw=0)
    ax.set_ylim(ymin=vmin, ymax=vmax)
    ax.set_title('%s' % title)
    ax.set_ylabel('Volumes ($km^3$)')
def plots_volumes_stack_cat_per(df_vol, ref, columns, labels, tcmap, title, ax):

    """
    Same as for plots_volumes_stack_cat but the volume are given according to a reference 'ref'
    :param df_vol: pandas dataframe in which the volume are saved
    :param ref: the reference to compute the %
    :param columns: the columns to plot
    :param labels: the labels of the stackplot
    :param tcmap: the colorbar
    :param title: the title of the plot
    :param ax: the ax on which to plot
    :return:
    """

    cmap = plt.get_cmap(tcmap, 10)
    custom_palette = [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    df_vol = df_vol[columns]
    ax.stackplot(df_vol.index, (df_vol.T / ref).values, labels=labels, alpha=0.9, colors=custom_palette, lw=0)
    ax.set_title('%s' % title)
    ax.set_ylabel(r'Volume fraction (%)')
def plot_pannels_per_cat_glob(df_volumes_NH, df_volumes_SH, var, cmap, cols_cat, labels_cat, vmin, vmax_NH, vmax_SH):

    """
    The function gather the previous function to make the big plot with NH and SH volums per sea ice thickness categories
    :param df_volumes_NH: Dataframe that contains NH sea ice volumes
    :param df_volumes_SH: Dataframe that contains SH sea ice volumes
    :param var: Variable to plot
    :param cmap: Colormap
    :param cols_cat: List of name of the categories in df_volumes*
    :param labels_cat: Label of each of the previous categories
    :param vmin: Min y value
    :param vmax_NH: Max y value for NH
    :param vmax_SH: Max y value for SH
    :return: figure and axs
    """


    fig, axs = plt.subplots(7, 4, figsize=(12, 14))

    # Select each dataframe according to the month for NH
    df_volumes_NH_10 = df_volumes_NH[df_volumes_NH.index.month == 10]
    df_volumes_NH_11 = df_volumes_NH[df_volumes_NH.index.month == 11]
    df_volumes_NH_12 = df_volumes_NH[df_volumes_NH.index.month == 12]
    df_volumes_NH_01 = df_volumes_NH[df_volumes_NH.index.month == 1]
    df_volumes_NH_02 = df_volumes_NH[df_volumes_NH.index.month == 2]
    df_volumes_NH_03 = df_volumes_NH[df_volumes_NH.index.month == 3]
    df_volumes_NH_04 = df_volumes_NH[df_volumes_NH.index.month == 4]

    # Plot the volume and the percentage of volume for each month
    plots_volumes_stack_cat(df_volumes_NH_10, cols_cat, labels_cat, cmap, 'October', axs[0, 0], vmin, vmax_NH)
    plots_volumes_stack_cat_per(df_volumes_NH_10, df_volumes_NH_10['%s' % var], cols_cat, labels_cat, cmap,
                                'October', axs[0, 1])
    plots_volumes_stack_cat(df_volumes_NH_11, cols_cat, labels_cat, cmap, 'November', axs[1, 0], vmin, vmax_NH)
    plots_volumes_stack_cat_per(df_volumes_NH_11, df_volumes_NH_11['%s' % var], cols_cat, labels_cat, cmap,
                                'November', axs[1, 1])
    plots_volumes_stack_cat(df_volumes_NH_12, cols_cat, labels_cat, cmap, 'December', axs[2, 0], vmin, vmax_NH)
    plots_volumes_stack_cat_per(df_volumes_NH_12, df_volumes_NH_12['%s' % var], cols_cat, labels_cat, cmap,
                                'December', axs[2, 1])
    plots_volumes_stack_cat(df_volumes_NH_01, cols_cat, labels_cat, cmap, 'January', axs[3, 0], vmin, vmax_NH)
    plots_volumes_stack_cat_per(df_volumes_NH_01, df_volumes_NH_01['%s' % var], cols_cat, labels_cat, cmap,
                                'January', axs[3, 1])
    plots_volumes_stack_cat(df_volumes_NH_02, cols_cat, labels_cat, cmap, 'February', axs[4, 0], vmin, vmax_NH)
    plots_volumes_stack_cat_per(df_volumes_NH_02, df_volumes_NH_02['%s' % var], cols_cat, labels_cat, cmap,
                                'February', axs[4, 1])
    plots_volumes_stack_cat(df_volumes_NH_03, cols_cat, labels_cat, cmap, 'March', axs[5, 0], vmin, vmax_NH)
    plots_volumes_stack_cat_per(df_volumes_NH_03, df_volumes_NH_03['%s' % var], cols_cat, labels_cat, cmap, 'March',
                                axs[5, 1])
    plots_volumes_stack_cat(df_volumes_NH_04, cols_cat, labels_cat, cmap, 'April', axs[6, 0], vmin, vmax_NH)
    plots_volumes_stack_cat_per(df_volumes_NH_04, df_volumes_NH_04['%s' % var], cols_cat, labels_cat, cmap, 'April',
                                axs[6, 1])

    # Select each dataframe according to the month for SH

    df_volumes_SH_05 = df_volumes_SH[df_volumes_SH.index.month == 5]
    df_volumes_SH_06 = df_volumes_SH[df_volumes_SH.index.month == 6]
    df_volumes_SH_07 = df_volumes_SH[df_volumes_SH.index.month == 7]
    df_volumes_SH_08 = df_volumes_SH[df_volumes_SH.index.month == 8]
    df_volumes_SH_09 = df_volumes_SH[df_volumes_SH.index.month == 9]
    df_volumes_SH_10 = df_volumes_SH[df_volumes_SH.index.month == 10]
    df_volumes_SH_11 = df_volumes_SH[df_volumes_SH.index.month == 11]

    # Plot the volume and the percentage of volume for each month
    plots_volumes_stack_cat(df_volumes_SH_05, cols_cat, labels_cat, cmap, 'May', axs[0, 2], vmin, vmax_SH)
    plots_volumes_stack_cat_per(df_volumes_SH_05, df_volumes_SH_05['%s' % var], cols_cat, labels_cat, cmap, 'May',
                                axs[0, 3])
    plots_volumes_stack_cat(df_volumes_SH_06, cols_cat, labels_cat, cmap, 'June', axs[1, 2], vmin, vmax_SH)
    plots_volumes_stack_cat_per(df_volumes_SH_06, df_volumes_SH_06['%s' % var], cols_cat, labels_cat, cmap, 'June',
                                axs[1, 3])
    plots_volumes_stack_cat(df_volumes_SH_07, cols_cat, labels_cat, cmap, 'July', axs[2, 2], vmin, vmax_SH)
    plots_volumes_stack_cat_per(df_volumes_SH_07, df_volumes_SH_07['%s' % var], cols_cat, labels_cat, cmap, 'July',
                                axs[2, 3])
    plots_volumes_stack_cat(df_volumes_SH_08, cols_cat, labels_cat, cmap, 'August', axs[3, 2], vmin, vmax_SH)
    plots_volumes_stack_cat_per(df_volumes_SH_08, df_volumes_SH_08['%s' % var], cols_cat, labels_cat, cmap,
                                'August', axs[3, 3])
    plots_volumes_stack_cat(df_volumes_SH_09, cols_cat, labels_cat, cmap, 'September', axs[4, 2], vmin, vmax_SH)
    plots_volumes_stack_cat_per(df_volumes_SH_09, df_volumes_SH_09['%s' % var], cols_cat, labels_cat, cmap,
                                'September', axs[4, 3])
    plots_volumes_stack_cat(df_volumes_SH_10, cols_cat, labels_cat, cmap, 'October', axs[5, 2], vmin, vmax_SH)
    plots_volumes_stack_cat_per(df_volumes_SH_10, df_volumes_SH_10['%s' % var], cols_cat, labels_cat, cmap,
                                'October', axs[5, 3])
    plots_volumes_stack_cat(df_volumes_SH_11, cols_cat, labels_cat, cmap, 'November', axs[6, 2], vmin, vmax_SH)
    plots_volumes_stack_cat_per(df_volumes_SH_11, df_volumes_SH_11['%s' % var], cols_cat, labels_cat, cmap,
                                'November', axs[6, 3])
    return (fig, axs)
def per_cat(df_volumes_NH, df_volumes_SH):
    """
    Gather the previous 3 functions to make the final plot
    :param df_volumes_NH: Pandas dataframe that gathers NH sea ice volumes
    :param df_volumes_SH: Pandas dataframe that gathers SH sea ice volumes
    :return:
    """

    print("Trends need to be computed first")
    xnh_10, ynh_10, trendnh_10, l10, h10, xnh_11, ynh_11, trendnh_11, l11, h11, xnh_12, ynh_12, trendnh_12, l12, h12, xnh_01, ynh_01, trendnh_01, l01, h01, xnh_02, ynh_02, trendnh_02, l02, h02, xnh_03, ynh_03, trendnh_03, l03, h03, xnh_04, ynh_04, trendnh_04, l04, h04 = trend_xr_months_winter(
        df_volumes_NH, 'evolume', 'NH')
    trendnh_10 = trendnh_10 * 60 * 60 * 24 * 365.25
    trendunh_10 = abs(abs(l10) - abs(h10)) / 2 * 60 * 60 * 24 * 365.25

    trendunh_11 = abs(abs(l11) - abs(h11)) / 2 * 60 * 60 * 24 * 365.25
    trendunh_12 = abs(abs(l12) - abs(h12)) / 2 * 60 * 60 * 24 * 365.25
    trendunh_01 = abs(abs(l01) - abs(h01)) / 2 * 60 * 60 * 24 * 365.25
    trendunh_02 = abs(abs(l02) - abs(h02)) / 2 * 60 * 60 * 24 * 365.25
    trendunh_03 = abs(abs(l03) - abs(h03)) / 2 * 60 * 60 * 24 * 365.25
    trendunh_04 = abs(abs(l04) - abs(h04)) / 2 * 60 * 60 * 24 * 365.25

    trendnh_11 = trendnh_11 * 60 * 60 * 24 * 365.25
    trendnh_12 = trendnh_12 * 60 * 60 * 24 * 365.25
    trendnh_01 = trendnh_01 * 60 * 60 * 24 * 365.25
    trendnh_02 = trendnh_02 * 60 * 60 * 24 * 365.25
    trendnh_03 = trendnh_03 * 60 * 60 * 24 * 365.25
    trendnh_04 = trendnh_04 * 60 * 60 * 24 * 365.25
    xnh_10 = pd.to_timedelta(xnh_10, unit='s') + np.datetime64('1970')
    xnh_11 = pd.to_timedelta(xnh_11, unit='s') + np.datetime64('1970')
    xnh_12 = pd.to_timedelta(xnh_12, unit='s') + np.datetime64('1970')
    xnh_01 = pd.to_timedelta(xnh_01, unit='s') + np.datetime64('1970')
    xnh_02 = pd.to_timedelta(xnh_02, unit='s') + np.datetime64('1970')
    xnh_03 = pd.to_timedelta(xnh_03, unit='s') + np.datetime64('1970')
    xnh_04 = pd.to_timedelta(xnh_04, unit='s') + np.datetime64('1970')

    cols_cat = ['evolume_cat_1', 'evolume_cat_2', 'evolume_cat_3', 'evolume_cat_4', 'evolume_cat_5', 'evolume_cat_6',
                'evolume_cat_7', 'evolume_cat_7.1']
    cols_cat.reverse()
    labels_cat = ['0-0.5m', '0.5-1m', '1-1.5m', '1.5-2m', '2-3m', '3-4m', '4-5m', '>5m']
    labels_cat.reverse()
    cmap = 'gist_earth'
    tcmap = plt.cm.gist_earth
    cat = [1, 2, 3, 4, 5, 6, 7, 8]
    props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
    ybbox = 0.85

    sns.set_style("ticks")
    sns.despine()

    sns.set_style("ticks")
    sns.despine()
    fig, axs = plot_pannels_per_cat_glob(df_volumes_NH, df_volumes_SH, 'evolume', cmap, cols_cat, labels_cat, 0, 16000, 28000)
    sns.set_style("ticks")
    sns.despine()
    axs[0, 0].plot(xnh_10, ynh_10, 'r')
    axs[0, 0].text(0.5, ybbox, '%.2f $\pm$ %.2f $km³/year$' % (trendnh_10, trendunh_10), horizontalalignment='center',
                   verticalalignment='center', transform=axs[0, 0].transAxes, bbox=props)
    axs[1, 0].plot(xnh_11, ynh_11, 'r')
    axs[1, 0].text(0.5, ybbox, '%.2f $\pm$ %.2f $km³/year$' % (trendnh_11, trendunh_11), horizontalalignment='center',
                   verticalalignment='center', transform=axs[1, 0].transAxes, bbox=props)
    axs[2, 0].plot(xnh_12, ynh_12, 'r')
    axs[2, 0].text(0.5, ybbox, '%.2f $\pm$ %.2f $km³/year$' % (trendnh_12, trendunh_12), horizontalalignment='center',
                   verticalalignment='center', transform=axs[2, 0].transAxes, bbox=props)
    axs[3, 0].plot(xnh_01, ynh_01, 'r')
    axs[3, 0].text(0.5, ybbox, '%.2f $\pm$ %.2f $km³/year$' % (trendnh_01, trendunh_01), horizontalalignment='center',
                   verticalalignment='center', transform=axs[3, 0].transAxes, bbox=props)
    axs[4, 0].plot(xnh_02, ynh_02, 'r')
    axs[4, 0].text(0.5, ybbox, '%.2f $\pm$ %.2f $km³/year$' % (trendnh_02, trendunh_02), horizontalalignment='center',
                   verticalalignment='center', transform=axs[4, 0].transAxes, bbox=props)
    axs[5, 0].plot(xnh_03, ynh_03, 'r')
    axs[5, 0].text(0.5, ybbox, '%.2f $\pm$ %.2f $km³/year$' % (trendnh_03, trendunh_03), horizontalalignment='center',
                   verticalalignment='center', transform=axs[5, 0].transAxes, bbox=props)
    axs[6, 0].plot(xnh_04, ynh_04, 'r')
    axs[6, 0].text(0.5, ybbox, '%.2f $\pm$ %.2f $km³/year$' % (trendnh_04, trendunh_04), horizontalalignment='center',
                   verticalalignment='center', transform=axs[6, 0].transAxes, bbox=props)


    sns.set_style("ticks")
    sns.despine()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = fig.legend( by_label.values(), by_label.keys(), loc='lower center', ncol=8, frameon = False, reverse=True, title='Thickness categories :', alignment='left', draggable=True)
    title = legend.get_title()
    title.set_x(-110)
    title.set_y(-13.5)

    for axxx in axs.ravel():
        axxx.tick_params(axis='x', labelrotation=25)
        axxx.xaxis.set_minor_locator(tck.AutoMinorLocator())

    sns.set_style("ticks")
    sns.despine()
    axcol1 = fig.add_subplot(1,2,1, frameon=False)
    axcol1.set_xticks([])
    axcol1.set_yticks([])
    axcol1.set_title("Northern Hemisphere (NH)", pad=27, fontsize=15)
    axcol1.set_title('(a)', loc='left', pad=27)

    axcol2 = fig.add_subplot(1,2,2, frameon=False)
    axcol2.set_xticks([])
    axcol2.set_yticks([])
    axcol2.set_title("Southern Hemisphere (SH)", pad=27, fontsize=15)
    axcol2.set_title('(b)', loc='left', pad=27)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, top=0.95, wspace=0.33)

    plt.savefig('vol_per_thick_cat_percentage_cumul.pdf')


# Function to plots appendice A5 et A6
# comparing sea ice density parametrizations and snow depth
def figure_S1_SH(df_volumes_SH, SH_vol_cs2_cci, SH_vol_env_cci, SH_vol_ers1r, SH_vol_ers2r, SH_vol_env3, SH_vol_c2):
    """
    Plot sea ice volume depending on snow depth products for two months (beginning and end of winter)
    :param df_volumes_SH: Pandas dataframe that present the sea ice volume for the entire time series
    :param SH_vol_cs2_cci: Pandas dataframe that present the sea ice volume from cs2 CCI
    :param SH_vol_env_cci: Pandas dataframe that present the sea ice volume from env3 CCI
    :param SH_vol_ers1r: Pandas dataframe that gather ERS1 sea ice volume
    :param SH_vol_ers2r: Pandas dataframe that gather ERS2 sea ice volume
    :param SH_vol_env3: Pandas dataframe that gather Envisat sea ice volume
    :param SH_vol_c2: Pandas dataframe that gather CS2 sea ice volume
    :return:
    """
    ##### PLOTS TS SH #####
    sns.set_style("ticks")
    sns.despine()
    figure = plt.figure('Time series SH - October', figsize=(12, 4))
    sns.set_style("ticks")
    sns.despine()
    axes = figure.add_subplot(122)
    sns.despine()

    df_volumes_SH_red = fill_df_nan(
        df_volumes_SH[['time', 'VOL_ASD_clim', 'VOL_AMSR_clim', 'VOL_AMSRE_NSIDC', 'VOL_AMSR2_NSIDC', 'VOL_SSMI']],
        '199401', '202306')
    df_volumes_SH_red['time'] = df_volumes_SH_red.index

    SH_vol_cs2_cci['time'] = pd.to_datetime(SH_vol_cs2_cci.time)
    SH_vol_env_cci['time'] = pd.to_datetime(SH_vol_env_cci.time)
    df_volumes_SH_10 = df_volumes_SH_red[df_volumes_SH_red.time.dt.month == 10]
    df_volumes_SH_04 = df_volumes_SH_red[df_volumes_SH_red.time.dt.month == 4]
    SH_vol_cs2_cci_10 = SH_vol_cs2_cci[SH_vol_cs2_cci.time.dt.month == 10]
    SH_vol_cs2_cci_04 = SH_vol_cs2_cci[SH_vol_cs2_cci.time.dt.month == 4]
    SH_vol_env_cci_10 = SH_vol_env_cci[SH_vol_env_cci.time.dt.month == 10]
    SH_vol_env_cci_04 = SH_vol_env_cci[SH_vol_env_cci.time.dt.month == 4]
    SH_vol_ers1r['time'] = pd.to_datetime(SH_vol_ers1r.index)
    SH_vol_ers2r['time'] = pd.to_datetime(SH_vol_ers2r.index)
    SH_vol_env3['time'] = pd.to_datetime(SH_vol_env3.index)
    SH_vol_c2['time'] = pd.to_datetime(SH_vol_c2.index)

    plt.plot(SH_vol_ers1r[SH_vol_ers1r.time.dt.month == 10].index, SH_vol_ers1r[SH_vol_ers1r.time.dt.month == 10].evolume,
             '.-', color='black', markersize=10, label=r'Volume from weighted mean')
    plt.plot(SH_vol_ers2r[SH_vol_ers2r.time.dt.month == 10].index, SH_vol_ers2r[SH_vol_ers2r.time.dt.month == 10].evolume,
             '.-', color='black', markersize=10)

    plt.plot(SH_vol_env3[SH_vol_env3.time.dt.month == 10].index, SH_vol_env3[SH_vol_env3.time.dt.month == 10].evolume,
             '.-',
             color='black', markersize=10)

    plt.plot(SH_vol_c2[SH_vol_c2.time.dt.month == 10].index, SH_vol_c2[SH_vol_c2.time.dt.month == 10].evolume, '.-',
             color='black', markersize=10)
    plt.plot(pd.to_datetime(df_volumes_SH_10.time), df_volumes_SH_10['VOL_SSMI'].replace(0,np.nan).values, color='tab:olive',
             lw=2, linestyle='--', label='SSMI')
    plt.plot(pd.to_datetime(df_volumes_SH_10.time), df_volumes_SH_10['VOL_AMSRE_NSIDC'].replace(0,np.nan).values, color='tab:pink', lw=2,
             linestyle='--', label='AMSR-E')
    plt.plot(pd.to_datetime(df_volumes_SH_10.time), df_volumes_SH_10['VOL_AMSR2_NSIDC'].values, color='tab:orange', lw=2,
             linestyle='--', label='AMSR-2')
    plt.plot(pd.to_datetime(df_volumes_SH_10.time), df_volumes_SH_10['VOL_ASD_clim'].values, color='tab:blue', lw=2,
             linestyle='--', label='ASD clim')
    plt.plot(pd.to_datetime(df_volumes_SH_10.time), df_volumes_SH_10['VOL_AMSR_clim'].values, color='tab:red', lw=2,
             linestyle='--', label='AMSR clim')
    plt.plot(SH_vol_cs2_cci_10.time, SH_vol_cs2_cci_10.volume, color='tab:green', alpha=0.75, lw=2, label='SI-CCI')

    plt.plot(SH_vol_env_cci_10.time, SH_vol_env_cci_10.volume, color='tab:green', alpha=0.75, lw=2)


    axes.yaxis.set_minor_locator(tck.AutoMinorLocator())
    axes.xaxis.set_minor_locator(tck.AutoMinorLocator())
    axes.set_ylim(ymin=0)
    plt.title('October')
    plt.title('(b)', loc='left')

    plt.xlabel('Year')
    plt.ylabel('Volume $km^3$')
    df_volumes_SH['time'] = df_volumes_SH.index

    sns.set_style("ticks")
    sns.despine()
    axes = figure.add_subplot(121)
    sns.despine()

    plt.plot(pd.to_datetime(df_volumes_SH_04.time), df_volumes_SH_04['VOL_SSMI'].values, color='tab:olive', lw=2,
             linestyle='--')
    plt.plot(pd.to_datetime(df_volumes_SH_04.time), df_volumes_SH_04['VOL_AMSRE_NSIDC'].replace(0,np.nan).values, color='tab:pink',
             lw=2, linestyle='--')
    plt.plot(pd.to_datetime(df_volumes_SH_04.time), df_volumes_SH_04['VOL_AMSR2_NSIDC'].replace(0,np.nan).values,
             color='tab:orange', lw=2, linestyle='--')
    plt.plot(pd.to_datetime(df_volumes_SH_04.time), df_volumes_SH_04['VOL_ASD_clim'].values, color='tab:blue', lw=2,
             linestyle='--')
    plt.plot(pd.to_datetime(df_volumes_SH_04.time), df_volumes_SH_04['VOL_AMSR_clim'].values, color='tab:red', lw=2,
             linestyle='--')

    plt.plot(SH_vol_ers1r[SH_vol_ers1r.time.dt.month == 4].index, SH_vol_ers1r[SH_vol_ers1r.time.dt.month == 4].evolume,
             '.-', color='black', markersize=10)
    plt.plot(SH_vol_cs2_cci_04.time, SH_vol_cs2_cci_04.volume, color='tab:green', lw=2, alpha=0.75)
    plt.plot(SH_vol_env_cci_04.time, SH_vol_env_cci_04.volume, color='tab:green', lw=2, alpha=0.75)
    plt.plot(SH_vol_ers2r[SH_vol_ers2r.time.dt.month == 4].index, SH_vol_ers2r[SH_vol_ers2r.time.dt.month == 4].evolume,
             '.-', color='black', markersize=10)
    plt.plot(SH_vol_env3[SH_vol_env3.time.dt.month == 4].index, SH_vol_env3[SH_vol_env3.time.dt.month == 4].evolume, '.-',
             color='black', markersize=10)

    plt.plot(SH_vol_c2[SH_vol_c2.time.dt.month == 4].index, SH_vol_c2[SH_vol_c2.time.dt.month == 4].evolume, '.-',
             color='black', markersize=10)
    axes.set_ylim(ymin=0)

    plt.title('April')
    plt.title('(a)', loc='left')
    axes.yaxis.set_minor_locator(tck.AutoMinorLocator())
    axes.xaxis.set_minor_locator(tck.AutoMinorLocator())
    plt.xlabel('Year')
    plt.ylabel('Volume $km^3$')
    figure.legend(loc='upper left', ncol=7, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.1)

    plt.savefig('SH_TS_comps.pdf')
def figure_S1_NH(df_volumes_NH, df_volumes_NH_rft, NH_vol_cs2smos, NH_vol_cs2_cci, NH_vol_env_cci, NH_vol_ers1r, NH_vol_ers2r, NH_vol_env3, NH_vol_c2, NH_vol_ers1r_rft, NH_vol_ers2r_rft, NH_vol_env3_rft, NH_vol_c2_rft):

    """
    Plot sea ice volume depending on snow depth products for two months (beginning and end of winter)
    :param df_volumes_NH: Pandas dataframe that present the sea ice volume for the entire time series
    :param df_volumes_NH_rft: Pandas dataframe that present the sea ice volume for the entire time series if Alexandrov et al 2010 parametrization
    :param NH_vol_cs2smos: Pandas dataframe that present the sea ice volume from CS2SMOS
    :param NH_vol_cs2_cci: Pandas dataframe that present the sea ice volume from cs2 CCI
    :param NH_vol_env_cci: Pandas dataframe that present the sea ice volume from Envisat CCI
    :param NH_vol_ers1r: Pandas dataframe that gather ERS1 sea ice volume
    :param NH_vol_ers2r: Pandas dataframe that gather ERS2 sea ice volume
    :param NH_vol_env3: Pandas dataframe that gather Envisat sea ice volume
    :param NH_vol_c2: Pandas dataframe that gather CS2 sea ice volume
    :param NH_vol_ers1r_rft: same as NH_vol_ers1r but for Alexandrov et al 2010 parametrization
    :param NH_vol_ers2r_rft: same as NH_vol_ers1r but for Alexandrov et al 2010 parametrization
    :param NH_vol_env3_rft: same as NH_vol_ers1r but for Alexandrov et al 2010 parametrization
    :param NH_vol_c2_rft: same as NH_vol_ers1r but for Alexandrov et al 2010 parametrization
    :param NH_vol_piomas: 
    :param NH_vol_nextsim: 
    :return: 
    """

    ##### PLOTS TS NH #####
    sns.set_style("ticks")
    sns.despine()
    figure = plt.figure('Time series NH - October', figsize=(12, 4))
    sns.set_style("ticks")
    sns.despine()
    axes = figure.add_subplot(121)
    sns.despine()

    df_volumes_NH_red = fill_df_nan(df_volumes_NH[['time', 'VOL_ASD_clim', 'VOL_LG_ERA', 'VOL_AMSRW99', 'VOL_NESOSIM']], '199401', '202304')
    df_volumes_NH_red['time'] = df_volumes_NH_red.index
    NH_vol_cs2smos['time'] = pd.to_datetime(NH_vol_cs2smos.time)

    NH_vol_cs2_cci['time'] = NH_vol_cs2_cci.index
    NH_vol_env_cci['time'] = NH_vol_env_cci.index
    df_volumes_NH_10 = df_volumes_NH_red[df_volumes_NH_red.time.dt.month == 10]
    df_volumes_NH_04 = df_volumes_NH_red[df_volumes_NH_red.time.dt.month == 4]
    cs2smos_monthly = NH_vol_cs2smos.groupby(NH_vol_cs2smos.time.dt.to_period("M")).mean()
    NH_vol_cs2smos_10 = cs2smos_monthly.loc[cs2smos_monthly.index.month == 10]
    NH_vol_cs2smos_04 = cs2smos_monthly.loc[cs2smos_monthly.index.month == 4]
    NH_vol_cs2_cci_10 = NH_vol_cs2_cci[NH_vol_cs2_cci.index.month == 10]
    NH_vol_cs2_cci_04 = NH_vol_cs2_cci[NH_vol_cs2_cci.index.month == 4]
    NH_vol_env_cci_10 = NH_vol_env_cci[NH_vol_env_cci.index.month == 10]
    NH_vol_env_cci_04 = NH_vol_env_cci[NH_vol_env_cci.index.month == 4]

    NH_vol_ers1r['time'] = pd.to_datetime(NH_vol_ers1r.index)
    NH_vol_ers2r['time'] = pd.to_datetime(NH_vol_ers2r.index)
    NH_vol_env3['time'] = pd.to_datetime(NH_vol_env3.index)
    NH_vol_c2['time'] = pd.to_datetime(NH_vol_c2.index)

    plt.plot(NH_vol_ers1r[NH_vol_ers1r.time.dt.month == 10].index,
             NH_vol_ers1r[NH_vol_ers1r.time.dt.month == 10].evolume, '.-', color='black', markersize=10,
             label=r'Volume from weighted mean - $\rho_i$ J21')
    plt.plot(NH_vol_ers1r_rft[NH_vol_ers1r.time.dt.month == 10].index,
             NH_vol_ers1r_rft[NH_vol_ers1r.time.dt.month == 10].evolume, '.-', color='grey', markersize=10,
             label=r'Volume from weighted mean - $\rho_i$ A10')

    plt.plot(NH_vol_ers2r[NH_vol_ers2r.time.dt.month == 10].index,
             NH_vol_ers2r[NH_vol_ers2r.time.dt.month == 10].evolume, '.-', color='black', markersize=10)
    plt.plot(NH_vol_ers2r_rft[NH_vol_ers2r.time.dt.month == 10].index,
             NH_vol_ers2r_rft[NH_vol_ers2r.time.dt.month == 10].evolume, '.-', color='grey', markersize=10)

    plt.plot(NH_vol_env3[NH_vol_env3.time.dt.month == 10].index, NH_vol_env3[NH_vol_env3.time.dt.month == 10].evolume,
             '.-', color='black', markersize=10)
    plt.plot(NH_vol_env3_rft[NH_vol_env3.time.dt.month == 10].index,
             NH_vol_env3_rft[NH_vol_env3.time.dt.month == 10].evolume, '.-', color='grey', markersize=10)
    plt.plot(NH_vol_c2[NH_vol_c2.time.dt.month == 10].index, NH_vol_c2[NH_vol_c2.time.dt.month == 10].evolume, '.-',
             color='black', markersize=10)
    plt.plot(NH_vol_c2_rft[NH_vol_c2.time.dt.month == 10].index, NH_vol_c2_rft[NH_vol_c2.time.dt.month == 10].evolume,
             '.-', color='grey', markersize=10)

    trend_J21 = mk.original_test(df_volumes_NH[df_volumes_NH.time.dt.month == 10].evolume, alpha=0.05)
    trend_A10 = mk.original_test(df_volumes_NH_rft[df_volumes_NH_rft.time.dt.month == 10].evolume, alpha=0.05)
    trend_LG = mk.original_test(df_volumes_NH_10['VOL_LG_ERA'])
    trend_ASD = mk.original_test(df_volumes_NH_10['VOL_ASD_clim'])
    trend_AMSRW99 = mk.original_test(df_volumes_NH_10['VOL_AMSRW99'])
    trend_NESOSIM = mk.original_test(df_volumes_NH_10['VOL_NESOSIM'])

    print('J21', trend_J21)
    print('A10', trend_A10)
    print('LG 10', trend_LG)
    print('trend_ASD 10', trend_ASD)
    print('trend_AMSRW99 10', trend_AMSRW99)
    print('trend_NESOSIM 10', trend_NESOSIM)

    print(mk.original_test(df_volumes_NH_10['VOL_NESOSIM']))

    plt.plot(pd.to_datetime(df_volumes_NH_10.time).iloc[0:-3], df_volumes_NH_10['VOL_LG_ERA'].iloc[0:-3].values,
             color='tab:orange', lw=2, linestyle='--', label='SnowLG-ERA')
    plt.plot(pd.to_datetime(df_volumes_NH_10.time), df_volumes_NH_10['VOL_ASD_clim'].values, color='tab:blue', lw=2,
             linestyle='--', label='ASD clim')
    plt.plot(pd.to_datetime(df_volumes_NH_10.time), df_volumes_NH_10['VOL_AMSRW99'].values, color='tab:red', lw=2,
             linestyle='--', label='W99-AMSR clim')
    plt.plot(pd.to_datetime(df_volumes_NH_10.time), df_volumes_NH_10['VOL_NESOSIM'].values, color='tab:olive', lw=2,
             linestyle='--', label='NESOSIM')
    plt.plot(NH_vol_cs2smos_10.index.to_timestamp(), NH_vol_cs2smos_10['volume'], color='pink', lw=2, label='CS2SMOS')
    plt.plot(NH_vol_cs2_cci_10.time, NH_vol_cs2_cci_10.volume, color='tab:green', lw=2, alpha=0.75, label='SI-CCI')
    plt.plot(NH_vol_env_cci_10.time, NH_vol_env_cci_10.volume, color='tab:green', lw=2, alpha=0.75)
    axes.yaxis.set_minor_locator(tck.AutoMinorLocator())
    axes.xaxis.set_minor_locator(tck.AutoMinorLocator())
    plt.xlabel('Year')
    plt.ylabel('Volume $km^3$')
    plt.title('October')
    plt.title('(a)', loc='left')
    axes.set_ylim(ymin=0)

    sns.set_style("ticks")
    sns.despine()
    axes = figure.add_subplot(122)
    sns.despine()
    plt.plot(pd.to_datetime(df_volumes_NH_04.time).iloc[0:-2], df_volumes_NH_04['VOL_LG_ERA'].iloc[0:-2].values,
             color='tab:orange', lw=2, linestyle='--')
    plt.plot(pd.to_datetime(df_volumes_NH_04.time), df_volumes_NH_04['VOL_ASD_clim'].values, color='tab:blue', lw=2,
             linestyle='--')
    plt.plot(pd.to_datetime(df_volumes_NH_04.time), df_volumes_NH_04['VOL_AMSRW99'].values, color='tab:red', lw=2,
             linestyle='--')
    plt.plot(pd.to_datetime(df_volumes_NH_04.time), df_volumes_NH_04['VOL_NESOSIM'].values, color='tab:olive', lw=2,
             linestyle='--')
    plt.plot(NH_vol_cs2smos_04.index.to_timestamp(), NH_vol_cs2smos_04['volume'], color='pink', lw=2)

    plt.plot(NH_vol_ers1r[NH_vol_ers1r.time.dt.month == 4].index, NH_vol_ers1r[NH_vol_ers1r.time.dt.month == 4].evolume)

    plt.plot(NH_vol_ers1r_rft[NH_vol_ers1r.time.dt.month == 4].index,
             NH_vol_ers1r_rft[NH_vol_ers1r.time.dt.month == 4].evolume, '.-', color='grey', markersize=10)
    plt.plot(NH_vol_cs2_cci_04.time, NH_vol_cs2_cci_04.volume, color='tab:green', lw=2, alpha=0.75)

    plt.plot(NH_vol_env_cci_04.time, NH_vol_env_cci_04.volume, color='tab:green', lw=2, alpha=0.75)

    plt.plot(NH_vol_ers2r[NH_vol_ers2r.time.dt.month == 4].index, NH_vol_ers2r[NH_vol_ers2r.time.dt.month == 4].evolume,
             '.-', color='black', markersize=10)
    plt.plot(NH_vol_ers2r_rft[NH_vol_ers2r.time.dt.month == 4].index,
             NH_vol_ers2r_rft[NH_vol_ers2r.time.dt.month == 4].evolume, '.-', color='grey', markersize=10)
    plt.plot(NH_vol_env3[NH_vol_env3.time.dt.month == 4].index, NH_vol_env3[NH_vol_env3.time.dt.month == 4].evolume,
             '.-', color='black', markersize=10)
    plt.plot(NH_vol_env3_rft[NH_vol_env3.time.dt.month == 4].index,
             NH_vol_env3_rft[NH_vol_env3.time.dt.month == 4].evolume, '.-', color='grey', markersize=10)

    plt.plot(NH_vol_c2[NH_vol_c2.time.dt.month == 4].index, NH_vol_c2[NH_vol_c2.time.dt.month == 4].evolume, '.-',
             color='black', markersize=10)
    plt.plot(NH_vol_c2_rft[NH_vol_c2.time.dt.month == 4].index, NH_vol_c2_rft[NH_vol_c2.time.dt.month == 4].evolume,
             '.-', color='grey', markersize=10)

    trend_J21 = mk.original_test(df_volumes_NH[df_volumes_NH.time.dt.month == 4].evolume, alpha=0.05)
    trend_A10 = mk.original_test(df_volumes_NH_rft[df_volumes_NH_rft.time.dt.month == 4].evolume, alpha=0.05)
    trend_LG = mk.original_test(df_volumes_NH_04['VOL_LG_ERA'])
    trend_ASD = mk.original_test(df_volumes_NH_04['VOL_ASD_clim'])
    trend_AMSRW99 = mk.original_test(df_volumes_NH_04['VOL_AMSRW99'])
    trend_NESOSIM = mk.original_test(df_volumes_NH_04['VOL_NESOSIM'])
    print('J21 04', trend_J21)
    print('A10 04', trend_A10)
    print('LG 04', trend_LG)
    print('trend_ASD 04', trend_ASD)
    print('trend_AMSRW99 04', trend_AMSRW99)
    print('trend_NESOSIM 04', trend_NESOSIM)

    plt.title('April')
    plt.title('(b)', loc='left')
    axes.set_ylim(ymin=0)
    axes.yaxis.set_minor_locator(tck.AutoMinorLocator())
    axes.xaxis.set_minor_locator(tck.AutoMinorLocator())
    plt.xlabel('Year')
    plt.ylabel('Volume $km^3$')

    figure.legend(loc='upper left', ncol=5, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.1)

    plt.savefig('NH_TS_comps.pdf')


# Auxilliary functions

def diff_month(d1, d2):
    """
    Compute the number of month between two dates (minus one month)
    :param d1: date1
    :param d2: date3
    :return:
    """
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def fill_df_nan(df, start_date, end_date):
    """
    Complete dataframe with nans for each month there are no values
    :param df: dataframe to fill
    :param start_date: first date to start filling
    :param end_date: last date to fill
    :return: pandas dataframe filled
    """
    # Complete df with nan
    nb_month = diff_month(datetime(int(end_date[0:4]), 12, 31), datetime(int(start_date[0:4]), 1, 1)) + 1
    df_time_complete = pd.date_range(np.datetime64('%s-%s-01' % (start_date[0:4], start_date[4::])), freq='MS',
                                     periods=nb_month + 1) + pd.DateOffset(days=14)
    data_df_time = np.zeros((df_time_complete.shape[0], df.shape[1])) + np.nan
    df_time = pd.DataFrame(columns=df.columns, data=data_df_time, index=df_time_complete)
    df_time = df_time.rename_axis('time', axis=0)
    df = df.set_index(pd.to_datetime((df.set_index('time').index).values))
    df_red = pd.concat([df, df_time])
    df_red = df_red[~df_red.index.duplicated(keep='first')].sort_index()
    return df_red

def fill_df_nan_anom(df, start_date, end_date):
    # Complete df with nan
    nb_month = diff_month(datetime(int(end_date[0:4]), 12, 31), datetime(int(start_date[0:4]), 1, 1)) + 1
    df_time_complete = pd.date_range(np.datetime64('%s-%s-01' % (start_date[0:4], start_date[4::])), freq='MS',
                                     periods=nb_month + 1) + pd.DateOffset(days=14)
    data_df_time = np.zeros((df_time_complete.shape[0], df.shape[1])) + np.nan
    df_time = pd.DataFrame(columns=df.columns, data=data_df_time, index=df_time_complete)
    df_time = df_time.rename_axis('time', axis=0)

    df_red = pd.concat([df, df_time])
    df_red = df_red[~df_red.index.duplicated(keep='first')].sort_index()
    return df_red
def concat_df_dealing_overlaps(dates_drop_overlap_ers1, dates_drop_overlap_ers2, dates_drop_overlap_env3, dates_drop_overlap_cs2, list_df):
    """
    :param dates_drop_overlap_ers1: list of dates to remove in ERS1
    :param dates_drop_overlap_ers2: list of dates to remove in ERS2
    :param dates_drop_overlap_env3: list of dates to remove in Envisat
    :param dates_drop_overlap_cs2: list of dates to remove in CS2
    :param list_df: list of dataframe to drop dates and to concat (ers1, ers2, envisat and cs2)
    :return: dataframe concatenate and fill by nan where needed
    """
    # Drop dates overlapping
    df_ers1_dropped = list_df[0].drop(dates_drop_overlap_ers1, axis=0)
    df_ers2_dropped = list_df[1].drop(dates_drop_overlap_ers2, axis=0)
    df_env3_dropped = list_df[2].drop(dates_drop_overlap_env3, axis=0)
    df_cs2_dropped = list_df[3].drop(dates_drop_overlap_cs2, axis=0)

    # concat
    df_dropped = pd.concat([df_ers1_dropped, df_ers2_dropped, df_env3_dropped, df_cs2_dropped])
    # fill
    df_dropped_fill = fill_df_nan(df_dropped, '%s%s' %(list_df[3].index.min().year, str(list_df[3].index.min().month).zfill(2)), '%s%s' %(str(list_df[3].index.max().year), str(list_df[3].index.max().month).zfill(2)))

    return df_dropped_fill

def app_total_seconds(val):
    """
    Compute seconds since 1970
    :param val: datetime
    :return: int, seconds
    """
    import datetime
    date = datetime.datetime(val.year, val.month, val.day, 0, 0, 0)
    ref = datetime.datetime(1970, 1, 1, 0, 0, 0)
    return((date-ref).total_seconds())
def month_of_year_to_month_of_season_NH(month):
    if month>=10:
        mm = month-9
    else:
        mm = month+3
    return mm
def month_of_year_to_month_of_season_NH_change_year(month):
    if month>=10:
        yy =  1
    else:
        yy = 0
    return yy

# Function that computes trends


def trend(tot, label):
    X = tot['second'].values.reshape(-1,1)
    res = stats.theilslopes(tot['%s' %label].values, tot.second.values)
    aa = res[0]
    bb = res[1]
    laa = res[2]
    haa = res[3]
    yr = aa*tot['second'].values+bb
    return(tot['second'].values, yr, res[0], laa, haa)
def trend_xr_months_winter(df, variable, hemisphere):
    if hemisphere == 'NH':

        x_10, y_10, trend_10, l10, h10 = trend(df[df.index.month == 10], variable)
        x_11, y_11, trend_11, l11, h11 = trend(df[df.index.month == 11], variable)
        x_12, y_12, trend_12, l12, h12 = trend(df[df.index.month == 12], variable)
        x_01, y_01, trend_01, l01, h01 = trend(df[df.index.month == 1], variable)
        x_02, y_02, trend_02, l02, h02 = trend(df[df.index.month == 2], variable)
        x_03, y_03, trend_03, l03, h03 = trend(df[df.index.month == 3], variable)
        x_04, y_04, trend_04, l04, h04 = trend(df[df.index.month == 4], variable)

        return (x_10, y_10, trend_10, l10, h10,
                x_11, y_11, trend_11, l11, h11,
                x_12, y_12, trend_12, l12, h12,
                x_01, y_01, trend_01, l01, h01,
                x_02, y_02, trend_02, l02, h02,
                x_03, y_03, trend_03, l03, h03,
                x_04, y_04, trend_04, l04, h04)
    else:
        x_01, y_01, trend_01, l01, h01 = trend(df[df.index.month == 1], variable)
        x_02, y_02, trend_02, l02, h02 = trend(df[df.index.month == 2], variable)
        x_03, y_03, trend_03, l03, h03 = trend(df[df.index.month == 3], variable)
        x_04, y_04, trend_04, l04, h04 = trend(df[df.index.month == 4], variable)
        x_05, y_05, trend_05, l05, h05 = trend(df[df.index.month == 5], variable)
        x_06, y_06, trend_06, l06, h06 = trend(df[df.index.month == 6], variable)
        x_07, y_07, trend_07, l07, h07 = trend(df[df.index.month == 7], variable)
        x_08, y_08, trend_08, l08, h08 = trend(df[df.index.month == 8], variable)
        x_09, y_09, trend_09, l09, h09 = trend(df[df.index.month == 9], variable)
        x_10, y_10, trend_10, l10, h10 = trend(df[df.index.month == 10], variable)
        x_11, y_11, trend_11, l11, h11 = trend(df[df.index.month == 11], variable)
        x_12, y_12, trend_12, l12, h12 = trend(df[df.index.month == 12], variable)

        return (x_01, y_01, trend_01, l01, h01,
                x_02, y_02, trend_02, l02, h02,
                x_03, y_03, trend_03, l03, h03,
                x_04, y_04, trend_04, l04, h04,
                x_05, y_05, trend_05, l05, h05,
                x_06, y_06, trend_06, l06, h06,
                x_07, y_07, trend_07, l07, h07,
                x_08, y_08, trend_08, l08, h08,
                x_09, y_09, trend_09, l09, h09,
                x_10, y_10, trend_10, l10, h10,
                x_11, y_11, trend_11, l11, h11,
                x_12, y_12, trend_12, l12, h12)


# Functions that compute anomalies and climatologies

def plot_TS_unc_clim(NH_vol_ers1r, NH_vol_ers1r_q025, NH_vol_ers1r_q975, NH_vol_ers2r, NH_vol_ers2r_q025, NH_vol_ers2r_q975,
                     NH_vol_env3, NH_vol_env3_q025, NH_vol_env3_q975, NH_vol_c2, NH_vol_c2_q025, NH_vol_c2_q975,
                     SH_vol_ers1r, SH_vol_ers1r_q025, SH_vol_ers1r_q975, SH_vol_ers2r, SH_vol_ers2r_q025,
                     SH_vol_ers2r_q975, SH_vol_env3, SH_vol_env3_q025, SH_vol_env3_q975, SH_vol_c2, SH_vol_c2_q025, SH_vol_c2_q975,
                     df_volumes_NH, mean_omip2, NH_vol_piomas, NH_vol_env_cci, NH_vol_cs2_cci, NH_vol_nextsim, month_NH,
                     df_volumes_SH, mean_omip2_SH, SH_vol_giomas, SH_vol_env_cci, SH_vol_cs2_cci, month_SH):
    figure = plt.figure('Time series', figsize=(12, 6))
    sns.set_style("ticks")
    sns.despine()
    spec = gridspec.GridSpec(ncols=2, nrows=2,
                             width_ratios=[5, 3], height_ratios=[1, 1])
    ax0 = figure.add_subplot(spec[0])
    ax1 = figure.add_subplot(spec[2])
    ax2 = figure.add_subplot(spec[1])
    ax3 = figure.add_subplot(spec[3])

    ax0.fill_between(NH_vol_ers1r.index, NH_vol_ers1r_q025.evolume, NH_vol_ers1r_q975.evolume, alpha=0.5, color='gray',
                     linewidth=0, label='90% confidence')
    ax0.fill_between(NH_vol_ers2r.index, NH_vol_ers2r_q025.evolume, NH_vol_ers2r_q975.evolume, alpha=0.5, color='gray',
                     linewidth=0)
    ax0.fill_between(NH_vol_env3.index, NH_vol_env3_q025.evolume, NH_vol_env3_q975.evolume, alpha=0.5, color='gray',
                     linewidth=0)
    ax0.fill_between(NH_vol_c2.index, NH_vol_c2_q025.evolume, NH_vol_c2_q975.evolume, alpha=0.5, color='gray',
                     linewidth=0)
    ax0.plot(NH_vol_c2.index, NH_vol_c2.evolume, '.-', color='#011f4b', markersize=6.5, lw=1, label='CryoSat-2')
    ax0.plot(NH_vol_env3.index, NH_vol_env3.evolume, '.-', color='#005b96', markersize=6.5, lw=1, label='Envisat')
    ax0.plot(NH_vol_ers2r.index, NH_vol_ers2r.evolume, '.-', color='#6497b1', markersize=6.5, lw=1, label='ERS-2')
    ax0.plot(NH_vol_ers1r.index, NH_vol_ers1r.evolume, '.-', color='#9AC0CD', markersize=6.5, lw=1, label='ERS-1')

    ax0.set_ylim(ymin=0)


    ax0.legend(ncol=5, frameon=False, loc='upper left', reverse=True)
    ax0.set_ylabel('Volume $km^3$')
    ax0.set_xlabel('Year')

    ax1.fill_between(SH_vol_ers1r.index, SH_vol_ers1r_q025.evolume, SH_vol_ers1r_q975.evolume, alpha=0.5, color='gray',
                     linewidth=0)
    ax1.plot(SH_vol_ers1r.index, SH_vol_ers1r.evolume, '.-', color='#9AC0CD', markersize=6.5, lw=1, label='ERS-1')
    ax1.fill_between(SH_vol_ers2r.index, SH_vol_ers2r_q025.evolume, SH_vol_ers2r_q975.evolume, alpha=0.5, color='gray',
                     linewidth=0)
    ax1.plot(SH_vol_ers2r.index, SH_vol_ers2r.evolume, '.-', color='#6497b1', markersize=6.5, lw=1, label='ERS-2')
    ax1.fill_between(SH_vol_env3.index, SH_vol_env3_q025.evolume, SH_vol_env3_q975.evolume, alpha=0.5, color='gray',
                     linewidth=0)
    ax1.plot(SH_vol_env3.index, SH_vol_env3.evolume, '.-', color='#005b96', markersize=6.5, lw=1, label='Envisat')
    ax1.fill_between(SH_vol_c2.index, SH_vol_c2_q025.evolume, SH_vol_c2_q975.evolume, alpha=0.5, color='gray',
                     linewidth=0)
    ax1.plot(SH_vol_c2.index, SH_vol_c2.evolume, '.-', color='#011f4b', markersize=6.5, lw=1, label='CryoSat-2')
    ax1.set_ylim(ymin=0)

    ax1.set_ylabel('Volume $km^3$')
    ax1.set_xlabel('Year')

    sns.lineplot(data=df_volumes_NH, x='month_of_season', y='evolume', hue='year_of_winter', lw=1.5, palette="Blues", ax=ax2)


    env3_cci = ax2.plot(NH_vol_env_cci.groupby(NH_vol_env_cci.month_of_season).volume.mean(), lw=3, linestyle='--', marker='*',
             color='darkblue', markersize=0, label='Envisat-CCI')
    cs2_cci = ax2.plot(NH_vol_cs2_cci.groupby(NH_vol_cs2_cci.month_of_season).volume.mean(), lw=3, linestyle='--', marker='*',
             color='seagreen', markersize=0, label='CS2-CCI')
    omip = ax2.plot(mean_omip2.groupby(mean_omip2.month_of_season).volume.mean(), lw=3, label='OMIP2 ensemble mean',
                    color='darkorange')
    nextsim = ax2.plot(NH_vol_nextsim.groupby(NH_vol_nextsim.month_of_season).volume.mean(), lw=3, marker='o', markersize=0,
             color='darkred', label='OPA-Nextsim')
    piomas = ax2.plot(NH_vol_piomas.groupby(NH_vol_piomas.month_of_season).volume.mean(), lw=3, linestyle='-',
                      marker='*',
                      color='red', markersize=0, label='PIOMAS')
    ax2.set_xticks(np.arange(13), month_NH, rotation='vertical')
    ax2.legend()

    sns.set_style("ticks")
    sns.despine()
    sns.lineplot(data=df_volumes_SH, x='month', y='evolume', hue='year', lw=1.5, palette="Blues", ax=ax3, legend=None)
    ax3.plot(mean_omip2_SH.groupby(mean_omip2_SH.month).volume.mean(), lw=3,
             color='darkorange')
    ax3.plot(SH_vol_env_cci.groupby(SH_vol_env_cci.month).volume.mean(), lw=3, linestyle='--', marker='*',
             color='darkblue', markersize=0)
    ax3.plot(SH_vol_cs2_cci.groupby(SH_vol_cs2_cci.month).volume.mean(), lw=3, linestyle='--', marker='*',
             color='seagreen', markersize=0)
    giomas = ax3.plot(SH_vol_giomas.groupby(SH_vol_giomas.month).volume.mean(), lw=3, linestyle='-', marker='*', color='mediumvioletred',
             markersize=0, label='GIOMAS')
    ax3.set_ylim(ymin=0, ymax=30000)
    ax2.set_ylim(ymin=0)

    plt.xticks(np.arange(13), month_SH, rotation='vertical')
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax2, ax3]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax3.legend(lines[6::], labels[6::], frameon=False, ncols=1, loc='upper left')
    ax2.legend(lines[0:6], labels[0:6], frameon=False, ncols=1, loc='upper right')

    ax0.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax0.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax1.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax3.yaxis.set_minor_locator(tck.AutoMinorLocator())

    ax3.set_xlabel('Month')
    ax3.set_ylabel('Volume $km^3$')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Volume $km^3$')
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, bottom=0.085, wspace=0.2)

    ax0.set_title('(a)', loc='left')
    ax1.set_title('(c)', loc='left')
    ax2.set_title('(b)', loc='left')
    ax3.set_title('(d)', loc='left')

    plt.savefig('vol_TS.pdf')



def anom_NH(df_volumes_NH_anom):
    """
    Plot the sea ice volume anomaly for each region, appendix B3
    :param df_volumes_NH_anom: Pandas dataframe NH with volume anomalies
    :return: fig, ax
    """
    sns.set_style("ticks")
    sns.despine()
    plt.tight_layout()
    fig, ax = plt.subplots(5, 2, figsize=(12, 12))
    sns.set_style("ticks")
    sns.despine()

    ax[4, 0].set_title('Hudson bay and Baffin bay')
    ax[4, 0].plot(df_volumes_NH_anom.evolume_region_4, label='Hudson bay')
    ax[4, 0].plot(df_volumes_NH_anom.evolume_region_6, label='Baffin bay')
    ax[4, 0].set_ylabel('Volume $km^3$')

    ax[4, 0].axhline(0, lw=0.5, color='k', zorder=0)
    ax[4, 0].legend(frameon = False)

    ax[2, 0].set_title('East Siberian Sea')
    ax[2, 0].plot(df_volumes_NH_anom.evolume_region_11, label='East Siberian Sea')
    ax[2, 0].axhline(0, lw=0.5, color='k', zorder=0)
    ax[2, 0].set_ylabel('Volume $km^3$')

    ax[1, 0].set_title('East Greeland and Barent Sea')
    ax[1, 0].plot(df_volumes_NH_anom.evolume_region_7, label='East Greenland Sea')
    ax[1, 0].axhline(0, lw=0.5, color='k', zorder=0)
    ax[1, 0].set_ylabel('Volume $km^3$')

    ax[1, 0].plot(df_volumes_NH_anom.evolume_region_8, label='Barent Sea')
    ax[1, 0].legend(frameon = False)

    ax[3, 0].set_title('Kara and Laptev sea')
    ax[3, 0].plot(df_volumes_NH_anom.evolume_region_9, label='Kara Sea')
    ax[3, 0].plot(df_volumes_NH_anom.evolume_region_10, label='Laptev Sea')
    ax[3, 0].axhline(0, lw=0.5, color='k', zorder=0)
    ax[3, 0].set_ylabel('Volume $km^3$')
    ax[4, 0].set_xlabel('Year')

    ax[3, 0].legend(frameon = False)

    ax[0, 0].set_title('Chukchi Sea and Beaufort Sea')
    ax[0, 0].plot(df_volumes_NH_anom.evolume_region_12, label='Chukchi Sea')
    ax[0, 0].plot(df_volumes_NH_anom.evolume_region_13, label='Beaufort Sea')
    ax[0, 0].set_ylabel('Volume $km^3$')

    ax[0, 0].legend(frameon = False)

    ax[0, 0].axhline(0, lw=0.5, color='k', zorder=0)
    ax[0, 0].legend(frameon = False)

    ax[0, 0].set_ylim(ymin=-1000, ymax=1000)
    ax[1, 0].set_ylim(ymin=-1000, ymax=1000)
    ax[2, 0].set_ylim(ymin=-1500, ymax=1500)
    ax[3, 0].set_ylim(ymin=-1000, ymax=1000)
    ax[4, 0].set_ylim(ymin=-500, ymax=500)

    ax[0, 0].xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax[1, 0].xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax[2, 0].xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax[3, 0].xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax[4, 0].xaxis.set_minor_locator(tck.AutoMinorLocator())


    plt.tight_layout()
    return(fig, ax)
def anom_SH(df_volumes_SH_anom, ax):
    """
    Plot the sea ice volume anomaly for each region, appendix B3
    :param df_volumes_SH_anom: Pandas dataframe SH with volume anomalies
    :return: fig, ax
    """
    sns.set_style("ticks")
    sns.despine()
    plt.tight_layout()

    ax[0, 1].set_title('Weddell Sea')
    ax[0, 1].plot(df_volumes_SH_anom.evolume_region_7, label='West Wedell Sea')
    ax[0, 1].plot(df_volumes_SH_anom.evolume_region_1, label='East Wedell Sea')
    ax[0, 1].legend(frameon = False)
    ax[0, 1].axhline(0, lw=0.5, color='k',zorder=0)
    ax[0, 1].axvline(np.datetime64('2016-01-01'), lw=0.5, color='k',zorder=0)
    ax[0, 1].set_ylabel('Volume $km^3$')

    ax[1, 1].set_title('Indian Ocean Sector')
    ax[1, 1].plot(df_volumes_SH_anom.evolume_region_2)

    ax[1, 1].set_ylabel('Volume $km^3$')

    ax[1, 1].axhline(0, lw=0.5, color='k',zorder=0)
    ax[1, 1].axvline(np.datetime64('2016-01-01'), lw=0.5, color='k',zorder=0)

    ax[2, 1].axhline(0, lw=0.5, color='k',zorder=0)
    ax[2, 1].axvline(np.datetime64('2016-01-01'), lw=0.5, color='k',zorder=0)
    ax[2, 1].set_title('Pacific Ocean Sector')
    ax[2, 1].plot(df_volumes_SH_anom.evolume_region_3)
    ax[2, 1].set_ylabel('Volume $km^3$')

    ax[3, 1].axhline(0, lw=0.5, color='k',zorder=0)
    ax[3, 1].axvline(np.datetime64('2016-01-01'), lw=0.5, color='k',zorder=0)
    ax[3, 1].set_title('Ross Sea')
    ax[3, 1].plot(df_volumes_SH_anom.evolume_region_4)
    ax[3, 1].set_ylabel('Volume $km^3$')

    ax[4, 1].axhline(0, lw=0.5, color='k',zorder=0)
    ax[4, 1].axvline(np.datetime64('2016-01-01'), lw=0.5, color='k',zorder=0)
    ax[4, 1].set_title('Amundsen-Bellingshausen')
    ax[4, 1].plot(df_volumes_SH_anom.evolume_region_5, label='AB')
    ax[4, 1].plot(df_volumes_SH_anom.evolume_region_6, label='Coastal AB')
    ax[4, 1].set_ylabel('Volume $km^3$')
    ax[4, 1].set_xlabel('Year')
    plt.legend(frameon = False)

    ax[1, 1].set_ylim(ymin = -1500, ymax = 1500)
    ax[0, 1].set_ylim(ymin = -2000, ymax = 2000)
    ax[2, 1].set_ylim(ymin = -1500, ymax = 1500)
    ax[3, 1].set_ylim(ymin = -1500, ymax = 1500)
    ax[4, 1].set_ylim(ymin = -1500, ymax = 1500)

    ax[0, 1].xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax[1, 1].xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax[2, 1].xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax[3, 1].xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax[4, 1].xaxis.set_minor_locator(tck.AutoMinorLocator())

