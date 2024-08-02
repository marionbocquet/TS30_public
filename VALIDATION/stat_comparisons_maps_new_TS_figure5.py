"""
Created by Marion Bocquet
Date 15/05/2024
Credits : LEGOS/CNES/CLS

This script make the figure 5, OIB

"""
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch
from matplotlib.dates import date2num, AutoDateLocator

MAP_PARAM_DESC = {
    "projection" : "laea",
    "lat_ts" : 0.,
    "lon_0" : 0,
    "lat_0" : 90,
    "resolution" : "l",
}


list_color = ['coral', 'teal', 'steelblue', 
              'darkred', 'orangered', 'forestgreen',
              'cornflowerblue', 'slategrey', 'mediumseagreen',
              'cadetblue', 'crimson', 'palevioletred',
              'doggerblue', 'darkslateblue', 'gold']   



regions_360 = {'lon_west' : {'WS' : 300, 'IO' : 20, 'PO' : 90, 'RS' : 160, 'BA' : 230},
               'lon_east' : {'WS' : 20, 'IO' : 90, 'PO' : 160, 'RS' : 230, 'BA' : 300},} 
time_sat = {'start' : {'ers1corr': np.datetime64('1992-05-02'), 'ers2corr': np.datetime64('1995-07-31') , 'env3corr': np.datetime64('2002-08-09'), 'c2esaDE' : np.datetime64('2012-09-22')}, 
            'end' : {'ers1corr': np.datetime64('1995-11-20'), 'ers2corr': np.datetime64('2003-04-02'), 'env3corr': np.datetime64('2010-12-19'), 'c2esaDE' : np.datetime64('2017-06-01')}}

def plot_mission_along_track(df, mission_name, fig, ax, sat, window, tolerance_seconds, title=None):
    """

    :param df: dataframe with freeboards and radar freeboards
    :param mission_name: name of OIB mission
    :param fig: figure
    :param ax: ax
    :param sat: mission to consider
    :param window: window for the rolling mean
    :param tolerance_seconds:
    :param title:
    :return:
    """
    df_mission = df[df.mission=='%s' %mission_name]
    sns.set_style('whitegrid')

    if df_mission.shape[0] != 0:
        list_variables = ['time_dt',
                          'ATM_fb', 
                          'FBt_%s_MB_sh'%(sat),
                          'eFBt_q025',
                          'eFBt_q975',
                          'FBt_%s_MB_sh_CASSIS'%(sat),
                          'FBt_%s_MB_sh_AMSRE_NSIDC'%(sat),
                          'FBt_%s_MB_sh_AMSR_clim'%(sat),
                          'FBt_%s_MB_sh_ASD_clim'%(sat),
                          'FBi_%s_MB_sh'%(sat),
                          'radar_freeboard_%s_MB_sh'%(sat),
                          ]
        if window!='':
            df_rolled = df_mission[list_variables].rolling(window, min_periods=10).mean()
        else:
            df_rolled = df_mission[list_variables]
        df_mission = df_mission.sort_values(by='time_dt')
        where_hole =  np.where(df_mission.index.to_series().diff().dt.seconds>tolerance_seconds)[0]
        where_hole = np.concatenate([np.array([0]), where_hole])
        ins = ax.inset_axes([0.792, 0.697, 0.3, 0.3])
        
        for s,segment in enumerate(where_hole):
            if segment == where_hole[-1]:
                ib = segment
                ie = -1
            else:
                ib = segment
                ie = where_hole[s+1] 
                """
                ib=0
                ie=-1
                """
            ax.plot(df_mission.index.values[ib:ie], df_mission['ATM_fb'][ib:ie], '-', color='teal', alpha=0.35, label='OIB ATM total freeboard')
            ax.plot(df_rolled.index.values[ib:ie], df_rolled['ATM_fb'][ib:ie], '-', color='teal', label='Smoothed OIB ATM total freeboard')
            ax.plot(df_mission.index.values[ib:ie], df_mission['FBt_%s_MB_sh'%(sat)][ib:ie], '-', color='darkred', label = 'Envisat NN FBtot ')
            ax.plot(df_mission.index.values[ib:ie], df_mission['eFBt_05'][ib:ie], ls='dotted', lw=0.5, color='darkred', label = 'Envisat NN FBtot quantile 5%')
            ax.plot(df_mission.index.values[ib:ie], df_mission['eFBt_q95'][ib:ie], ls='dotted', lw=0.5, color='darkred', label = 'Envisat NN FBtot quantile 95%')
            ax.plot(df_mission.index.values[ib:ie], df_mission['radar_freeboard_%s_MB_sh'%(sat)][ib:ie], '-', color='dimgray', label = 'Envisat NN FBr')

        mean = (df_mission['FBt_%s_MB_sh'%(sat)]-df_rolled['ATM_fb']).mean()
        median = (df_mission['FBt_%s_MB_sh'%(sat)]-df_rolled['ATM_fb']).median()
        std = (df_mission['FBt_%s_MB_sh'%(sat)]-df_rolled['ATM_fb']).std()
        rmse = ((df['FBt_%s_MB_sh'%(sat)] - df_rolled['ATM_fb']) ** 2).mean() ** .5
        corr_coef = df_mission['FBt_%s_MB_sh'%(sat)].corr(df_rolled['ATM_fb'])
        m = plot_map_trajectories_mission(df, df_mission, ins)
        props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.85)
        px = 0.035
        py = 0.82
        ax.text(px, py, ' N = %s\n Bias = %.3f\n Med = %.3f \n SD = %.3f\n RMSE = %.3f\n $r$ = %.3f' % (
        df_mission['FBt_%s_MB_sh'%(sat)].shape[0], mean, median, std, rmse, corr_coef), horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, bbox=props, fontsize=8)
        sns.set_style('whitegrid')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        con1 = ConnectionPatch(xyA=(date2num(df_mission.index[0]), 1.5), xyB=m(df_mission.lon[0], df_mission.lat[0]), coordsA='data', coordsB='data', axesA=ax, axesB=ins, color='black', linestyle='--', zorder=1001)
        con2 = ConnectionPatch(xyA=(date2num(df_mission.index[-5]), 1.5), xyB=m(df_mission.lon[-1], df_mission.lat[-1]), coordsA='data', coordsB='data', axesA=ax, axesB=ins, color='black', linestyle='--', zorder=1000)

        ax.add_artist(con1)
        ax.add_artist(con2)
        ax.set_ylim([-0.5, 3])
        ax.set_xticklabels(ax.get_xticks(), rotation = 45, horizontalalignment='right')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H-%M'))
        return by_label
    else:
        return

def plot_map_trajectories_mission(df, df_mission, ax):
    """
    The function aims to make the map of the top right of the time series plot
    :param df: dataframe with all OIB missions
    :param df_mission: dataframe with the mission to plot
    :param ax: ax
    :return:
    """
    fact=-1
    m = Basemap(llcrnrlon=-105, llcrnrlat=-55, urcrnrlon=-10, urcrnrlat=-67, projection = MAP_PARAM_DESC['projection'],
                           lat_0 = MAP_PARAM_DESC['lat_0']*fact, 
                           lat_ts = MAP_PARAM_DESC['lat_ts'], 
                           lon_0 = MAP_PARAM_DESC['lon_0'],
                           #width = 500*12500,
                           #height = 500*12500,
                           resolution = MAP_PARAM_DESC['resolution'],

                round = False, ax = ax)
    m.fillcontinents(color='lightgray')
    m.drawparallels(np.arange(-90, 90, 20), linewidth=0.4, color="gray")
    m.drawmeridians(np.arange(0, 360, 30), linewidth=0.4, color="gray")
    lon_m, lat_m = m(df.lon.values, df.lat.values)
    lon_m_mission, lat_m_mission = m(df_mission.lon.values, df_mission.lat.values)
    sc = m.scatter(lon_m, lat_m, c='#EBEBEB', alpha=0.15, s=1)
    sc = m.scatter(lon_m_mission, lat_m_mission, c='black', s=1)
    plt.setp(ax.spines.values(), color='black', linewidth=1.5)
    return m




# --------------------------------------------------
# Main 
#

if __name__ == '__main__':
    directory = 'OIB_ANT'
    i=0
    rhow = 1024

    # Loading dataset and merging into one big dataset with label mission
    df = pd.DataFrame()
    for file in os.listdir(directory):                                                            
        i+=1
        filename = os.path.join(directory, file)                                    
        ds = xr.open_dataset(filename)
        df_temp = ds.to_dataframe()
        name = file[:-3]
        df_temp['mission'] = np.repeat(np.array('%s' %name), df_temp.shape[0])
        if i>1:
            df = pd.concat([df, df_temp], axis=0)
        else:
            df = df_temp.copy()

    print('-------------------------df is fully loaded-------------------------')  

    if (df.lon<0).values.any():
        df.lon[(df.lon<0)] = df.lon[(df.lon<0)] + 360
        print('Negative longitude converted to positive >180')
    else:
        print('No negative longitude')

    
    df.loc[(df.lon>regions_360['lon_west']['PO']) & (df.lon<=regions_360['lon_east']['PO']), 'region'] = 'PO'
    df.loc[(df.lon>regions_360['lon_west']['WS']) & (df.lon<=regions_360['lon_east']['WS']), 'region'] = 'WS' 
    df.loc[(df.lon>regions_360['lon_west']['RS']) & (df.lon<=regions_360['lon_east']['RS']), 'region'] = 'RS' 
    df.loc[(df.lon>regions_360['lon_west']['BA']) & (df.lon<=regions_360['lon_east']['BA']), 'region'] = 'BA' 
    df.loc[(df.lon>regions_360['lon_west']['IO']) & (df.lon<=regions_360['lon_east']['IO']), 'region'] = 'IO'

    df['time_dt'] = df.index
    sns.set_style('whitegrid')

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))  # layout='constrained')

    sns.set_style('whitegrid')
    plot_mission_along_track(df, 'IDCSI4_20091021', fig, axes[0,0], 'env3corr', '1min', 60, title='IDCSI4_20091021')
    axes[0, 0].set_ylabel('Total freeboard (m)')

    sns.set_style('whitegrid')
    plot_mission_along_track(df, 'IDCSI4_20091024', fig, axes[0,1], 'env3corr', '1min', 60, title='IDCSI4_20091024')

    sns.set_style('whitegrid')
    plot_mission_along_track(df, 'IDCSI4_20091030', fig, axes[1,0], 'env3corr', '1min', 60, title='IDCSI4_20091030')
    axes[1, 0].set_ylabel('Total freeboard (m)')
    sns.set_style('whitegrid')
    
    
    sns.set_style('whitegrid')
    plot_mission_along_track(df, 'IDCSI4_20101026', fig, axes[1,1], 'env3corr', '1min', 60, title='IDCSI4_20101021')


    sns.set_style('whitegrid')
    plot_mission_along_track(df, 'IDCSI4_20101028', fig, axes[2,0], 'env3corr', '1min', 60, title='IDCSI4_20101024')
    axes[2, 0].set_xlabel('Time')
    axes[2, 0].set_ylabel('Total freeboard (m)')
    


    sns.set_style('whitegrid')
    by_label = plot_mission_along_track(df, 'IDCSI4_20101030', fig, axes[2,1], 'env3corr', '1min', 60, title='IDCSI4_20101030')
    axes[2, 1].set_xlabel('Time')
    legend = fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=5, frameon = False, draggable=True)

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.13, bottom=0.11, top=0.94)

    plt.savefig('OIB_ANT.pdf')
    plt.savefig('OIB_ANT.png')


    
