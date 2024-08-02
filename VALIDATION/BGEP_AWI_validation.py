"""
Created by Marion Bocquet
Date 15/05/2024
Credits : LEGOS/CNES/CLS

This script make the figure 2, BGEP and AWI mooring validation
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import datetime
from mpl_toolkits.basemap import Basemap
import matplotlib.dates as mdates
import plots_TS_function as pts
MAP_PARAM_DESC = {
    "projection" : "laea",
    "lat_ts" : 0.,
    "lon_0" : 0,
    "lat_0" : 90,
    "resolution" : "l",
}

def plot_map_trajectories_mission(lon, lat, ax, i, hem):
    fact = hem
    m = Basemap(llcrnrlon=-100, llcrnrlat=-60, urcrnrlon=15, urcrnrlat=-50, projection=MAP_PARAM_DESC['projection'],
                lat_0=MAP_PARAM_DESC['lat_0'] * fact,
                lat_ts=MAP_PARAM_DESC['lat_ts'],
                lon_0=MAP_PARAM_DESC['lon_0'],
                # width = 500*12500,
                # height = 500*12500,
                resolution=MAP_PARAM_DESC['resolution'],

                round=False, ax=ax)
    m.drawcoastlines(color='gray', linewidth=0.3)
    m.fillcontinents(color='lightgray')
    m.drawparallels(np.arange(-90, 90, 20), linewidth=0.4, color="gray")
    m.drawmeridians(np.arange(0, 360, 30), linewidth=0.4, color="gray")
    lon_m, lat_m = m(lon[i], lat[i])
    lon_m_mission, lat_m_mission = m(lon, lat)

    sc = m.scatter(lon_m_mission, lat_m_mission, c='lightgrey', s=10, )#, alpha=0.15, s=10)
    sc = m.scatter(lon_m, lat_m, c='red', s=10)

    plt.setp(ax.spines.values(), color='black', linewidth=0.5)
    return m

def plot_map_trajectories_mission_NH(lon, lat, ax, i, hem):
    fact = hem
    m = Basemap(llcrnrlon=-70, llcrnrlat=70, urcrnrlon=160, urcrnrlat=70, projection=MAP_PARAM_DESC['projection'],
                lat_0=MAP_PARAM_DESC['lat_0'] * fact,
                lat_ts=MAP_PARAM_DESC['lat_ts'],
                lon_0=MAP_PARAM_DESC['lon_0'],
                # width = 500*12500,
                # height = 500*12500,
                resolution=MAP_PARAM_DESC['resolution'],

                round=False, ax=ax)
    m.drawcoastlines(color='gray', linewidth=0.3)
    m.fillcontinents(color='lightgray')
    m.drawparallels(np.arange(-90, 90, 20), linewidth=0.4, color="gray")
    m.drawmeridians(np.arange(0, 360, 30), linewidth=0.4, color="gray")
    lon_m, lat_m = m(lon[i], lat[i])
    lon_m_mission, lat_m_mission = m(lon, lat)

    sc = m.scatter(lon_m_mission, lat_m_mission, c='lightgrey', s=10, )#, alpha=0.15, s=10)
    sc = m.scatter(lon_m, lat_m, c='red', s=10)

    plt.setp(ax.spines.values(), color='black', linewidth=0.5)

    return m


def fill_df_nan(df, start_date, end_date):
    # Complete df with nan
    nb_month = pts.diff_month(datetime.datetime(int(end_date[0:4]),12,31), datetime.datetime(int(start_date[0:4]),1,1))+1
    df_time_complete = pd.date_range(np.datetime64('%s-%s-01' %(start_date[0:4], start_date[4::])), freq='MS', periods=nb_month+1) + pd.DateOffset(days=14)
    data_df_time = np.zeros((df_time_complete.shape[0], df.shape[1]))+np.nan
    df_time = pd.DataFrame(columns=df.columns, data=data_df_time, index=df_time_complete)
    df_time = df_time.rename_axis('time',axis=0)
    df_time['time'] = df_time.index.values
    df = df.set_index('time')
    df['time'] = df.index.values
    df_red = pd.concat([df, df_time])
    df_red = df_red[~df_red.index.duplicated(keep='first')].sort_index()
    return df_red

mooring_name_list = ['206', '207', '227', '231', '229', '232', '233', '230'][::-1]
lat_list = [-63.48, -63.72, -59.5, -66.5, -63.97, -69, -69.4, -66][::-1]
lon_list = [-52.1, -50.83, 0, 0, 0,0, 0, 0.16][::-1]
lon_BGEP = [-150,-150,-140,-140]
lat_BGEP = [75,78,77,74]
mooring_number = ['(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', ][::-1]
path = ''
sns.set_style('whitegrid')

fig = plt.figure(figsize=(12, 12))
sns.set_style('whitegrid')
ax1 = fig.add_subplot(8, 2, 7+8)
ax0 = fig.add_subplot(8, 2, 8+8, sharey=ax1)
ax3 = fig.add_subplot(8, 2, 5+8, sharex=ax1)
ax2 = fig.add_subplot(8, 2, 6+8, sharex=ax0, sharey=ax3)
ax5 = fig.add_subplot(8, 2, 3+8, sharex=ax1)
ax4 = fig.add_subplot(8, 2, 4+8, sharex=ax0, sharey=ax5)
ax7 = fig.add_subplot(8, 2, 1+8, sharex=ax1)
ax6 = fig.add_subplot(8, 2, 2+8, sharex=ax0, sharey=ax7)

for n,mooring_name in enumerate(mooring_name_list):
    print(n)
    if n==1:
        ax = ax1
    elif n==0:
        ax = ax0
        plt.setp(ax0.get_yticklabels(), visible=False)

    elif n==3:
        plt.setp(ax3.get_xticklabels(), visible=False)
        ax = ax3
    elif n==2:
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax = ax2
    elif n==5:
        plt.setp(ax5.get_xticklabels(), visible=False)

        ax = ax5
    elif n==4:
        plt.setp(ax4.get_xticklabels(), visible=False)
        plt.setp(ax4.get_yticklabels(), visible=False)

        ax = ax4
    elif n==7:
        plt.setp(ax7.get_xticklabels(), visible=False)

        ax = ax7
    elif n==6:
        plt.setp(ax6.get_xticklabels(), visible=False)
        plt.setp(ax6.get_yticklabels(), visible=False)

        ax = ax6

    list_mooring_path = glob.glob(path + 'AWI/TS__Time_Serie*%s*.csv' %mooring_name)

    sns.set_style('whitegrid')

    list_mooring = []
    list_mooring_path = np.sort(list_mooring_path)
    sns.set_style('whitegrid')

    for i, f in enumerate(list_mooring_path):
        mooring = pd.read_csv(f)
        sns.set_style('whitegrid')

        mooring['time'] = pd.to_datetime(mooring.time, format ='%Y-%m-%d')
        size = mooring.shape[0]
        mooring = fill_df_nan(mooring, '199606', '201301')
        mooring_time = pd.to_datetime(mooring.time, format ='%Y-%m-%d')
        sns.set_style('whitegrid')

        ax.plot(mooring_time, mooring.mooring_draft, alpha=1, color='lightgray', linewidth=1, label='Daily draft')
        ax.plot(mooring_time, mooring.mooring_draft_month, color='k', label='31-days rolling average draft')
        ax.scatter(mooring_time, mooring['draft_from_edraft_ers1rSH_corr_interp'], color='teal', marker='o', s=10, label='ERS-1', zorder=10000)
        ax.scatter(mooring_time, mooring['draft_from_edraft_ers2rSH_corr_interp'], color='orange', marker='o', s=10, label='ERS-2', zorder=10000)
        ax.scatter(mooring_time, mooring['draft_from_edraft_env3SH_corr_interp'], color='darkred', marker='o', s=10, label='Envisat', zorder=10000)

        list_mooring.append(mooring)
    ins = ax.inset_axes([0.53, 0.1, 0.8, 0.8])
    m = plot_map_trajectories_mission(lon_list, lat_list, ins, n, -1)

    df = pd.concat(list_mooring)
    diff = (pd.concat([(df['draft_from_edraft_ers1rSH_corr_interp'] - df.mooring_draft_month).dropna(), (df['draft_from_edraft_env3SH_corr_interp'] - df.mooring_draft_month).dropna(), (df['draft_from_edraft_ers2rSH_corr_interp'] - df.mooring_draft_month).dropna()]))
    df_sat = pd.concat([(df['draft_from_edraft_ers1rSH_corr_interp']), (df['draft_from_edraft_env3SH_corr_interp']), (df['draft_from_edraft_ers2rSH_corr_interp'])])
    mean = diff.mean()
    median = diff.median()
    std = diff.std()
    rmse = (diff**2).mean()**0.5
    corr_coef = df_sat.corr(df.mooring_draft_month)
    mooring = mooring.sort_index()

    sns.set_style('whitegrid')

    mooring['time'] = pd.to_datetime(mooring.time, format='%Y-%m-%d')
    xticks = mooring.loc[((mooring.index.month == 1) | (mooring.index).month == 7) & (mooring.index.day == 15)].index.values

    ax.set_title('%s AWI %s'%(mooring_number[n], mooring_name), loc='left')

    ax.set_ylim((-0.5, 5))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.85)
    px = 0.2
    py = 1.15
    ax.text(px, py, ' N = %s, Bias = %.3f, Med = %.3f, SD = %.3f, RMSE = %.3f, $r$ = %.3f' % (
        diff.shape[0], mean, median, std, rmse, corr_coef), horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, bbox=props, fontsize="9")

axes = fig.axes
axes[1].set_xticks(xticks)
axes[1].set_xticklabels(xticks, rotation=20, horizontalalignment='right')
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[0].set_xticks(xticks)
axes[0].set_xticklabels(xticks, rotation=20, horizontalalignment='right')
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[0].set_xlabel('Year')
axes[1].set_xlabel('Year')
axes[6].set_ylabel('Sea Ice Draft (m)')
axes[4].set_ylabel('Sea Ice Draft (m)')
axes[2].set_ylabel('Sea Ice Draft (m)')
axes[0].set_ylabel('Sea Ice Draft (m)')
plt.tight_layout()

plt.subplots_adjust(top=0.94, bottom=0.05, wspace=0.04, right=0.98, hspace=0.72)

#  ------------- add BGEP ------------------ #

def stat(datax, datay, ax, px=0.1, py=1.15):
    """

    :param datax:
    :param datay:
    :param ax: ax on wich the stats have to be printed
    :param px: position x along ax axis
    :param py: position y alonf ax axis
    :return: stat box on the axis
    """
    rmse = ((datay - datax).dropna() ** 2).mean() ** .5
    corr_coef = datax.corr(datay, method='pearson')
    dif = (datay - datax).dropna()
    props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.85)
    ax.text(px, py, ' N = %s, Bias = %.3f, Med = %.3f, SD = %.3f, RMSE = %.3f, $r$ = %.3f'%(dif.shape[0], dif.mean(), dif.median(), dif.std(), rmse, corr_coef), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, bbox=props, fontsize=9)
def fill_df_nan_BGEP(df, start_date, end_date):
    """
    Complete dataframe with nans for each month there are no values
    :param df: dataframe to fill
    :param start_date: first date to start filling
    :param end_date: last date to fill
    :return: pandas dataframe filled
    """
    # Complete df with nan
    nb_month = pts.diff_month(datetime.datetime(int(end_date[0:4]),12,31), datetime.datetime(int(start_date[0:4]),1,1))+1
    df_time_complete = pd.date_range(np.datetime64('%s-%s-01' %(start_date[0:4], start_date[4::])), freq='MS', periods=nb_month+1) + pd.DateOffset(days=14)
    data_df_time = np.zeros((df_time_complete.shape[0], df.shape[1]))+np.nan
    df_time = pd.DataFrame(columns=df.columns, data=data_df_time, index=df_time_complete)
    df_time = df_time.rename_axis('time',axis=0)
    df_red = pd.concat([df, df_time])
    df_red = df_red[~df_red.index.duplicated(keep='first')].sort_index()
    return df_red


ds_BGEP_A_env = pd.read_csv('BGEP/combine__Time_Serie_BGEP_A_daily_2003-10_2012-03.csv').rename(columns = {'draft_from_edraft_env3_corr_interp':'sat'})
ds_BGEP_A_env['time'] = pd.to_datetime(ds_BGEP_A_env['time'])
ds_BGEP_A_env = ds_BGEP_A_env.set_index('time')
ds_BGEP_A_env['mooring_monthly'] = ds_BGEP_A_env['mooring_draft_raw'].rolling(31, min_periods=1, center=True).mean()
ds_BGEP_B_env = pd.read_csv('BGEP/combine__Time_Serie_BGEP_B_daily_2003-10_2012-03.csv').rename(columns = {'draft_from_edraft_env3_corr_interp':'sat'})
ds_BGEP_B_env['time'] = pd.to_datetime(ds_BGEP_B_env['time'])
ds_BGEP_B_env = ds_BGEP_B_env.set_index('time')
ds_BGEP_B_env['mooring_monthly'] = ds_BGEP_B_env['mooring_draft_raw'].rolling(31, min_periods=1, center=True).mean()
ds_BGEP_C_env = pd.read_csv('BGEP/combine__Time_Serie_BGEP_C_daily_2003-10_2008-04.csv').rename(columns = {'draft_from_edraft_env3_corr_interp':'sat'})
ds_BGEP_C_env['time'] = pd.to_datetime(ds_BGEP_C_env['time'])
ds_BGEP_C_env = ds_BGEP_C_env.set_index('time')
ds_BGEP_C_env['mooring_monthly'] = ds_BGEP_C_env['mooring_draft_raw'].rolling(31, min_periods=1, center=True).mean()
ds_BGEP_D_env = pd.read_csv('BGEP/combine__Time_Serie_BGEP_D_daily_2006-10_2012-03.csv').rename(columns = {'draft_from_edraft_env3_corr_interp':'sat'})
ds_BGEP_D_env['time'] = pd.to_datetime(ds_BGEP_D_env['time'])
ds_BGEP_D_env = ds_BGEP_D_env.set_index('time')
ds_BGEP_D_env['mooring_monthly'] = ds_BGEP_D_env['mooring_draft_raw'].rolling(31, min_periods=1, center=True).mean()

ds_BGEP_A_c2 = pd.read_csv('BGEP/combine__Time_Serie_BGEP_A_daily_2010-11_2022-10.csv').rename(columns = {'draft_from_edraft_c2esaD1_SARpIN_interp':'sat'})
ds_BGEP_A_c2['mooring_monthly'] = ds_BGEP_A_c2['mooring_draft_raw'].rolling(31, min_periods=1, center=True).mean()
ds_BGEP_A_c2['time'] = pd.to_datetime(ds_BGEP_A_c2['time'])
ds_BGEP_A_c2 = ds_BGEP_A_c2.set_index('time')
ds_BGEP_B_c2 = pd.read_csv('BGEP/combine__Time_Serie_BGEP_B_daily_2010-11_2022-04.csv').rename(columns = {'draft_from_edraft_c2esaD1_SARpIN_interp':'sat'})
ds_BGEP_B_c2['time'] = pd.to_datetime(ds_BGEP_B_c2['time'])
ds_BGEP_B_c2 = ds_BGEP_B_c2.set_index('time')
ds_BGEP_B_c2['mooring_monthly'] = ds_BGEP_B_c2['mooring_draft_raw'].rolling(31, min_periods=1, center=True).mean()
ds_BGEP_D_c2 = pd.read_csv('BGEP/combine__Time_Serie_BGEP_D_daily_2010-11_2022-10.csv').rename(columns = {'draft_from_edraft_c2esaD1_SARpIN_interp':'sat'})
ds_BGEP_D_c2['time'] = pd.to_datetime(ds_BGEP_D_c2['time'])
ds_BGEP_D_c2 = ds_BGEP_D_c2.set_index('time')
ds_BGEP_D_c2['mooring_monthly'] = ds_BGEP_D_c2['mooring_draft_raw'].rolling(31, min_periods=1, center=True).mean()

BGEP_A = pd.concat([ds_BGEP_A_env, ds_BGEP_A_c2])
BGEP_A = fill_df_nan_BGEP(BGEP_A, '200201', '202301')
BGEP_A['date'] = BGEP_A.index.astype('str')
BGEP_B = pd.concat([ds_BGEP_B_env, ds_BGEP_B_c2])
BGEP_B = fill_df_nan_BGEP(BGEP_B, '200201', '202301')
BGEP_B['date'] = BGEP_B.index.astype('str')

BGEP_C = pd.concat([pd.DataFrame(), ds_BGEP_C_env])
BGEP_C = fill_df_nan_BGEP(BGEP_C, '200201', '202301')
BGEP_C['date'] = BGEP_C.index.astype('str')

BGEP_D = pd.concat([ds_BGEP_D_env, ds_BGEP_D_c2])
BGEP_D = fill_df_nan_BGEP(BGEP_D, '200201', '202301')
BGEP_D['date'] = BGEP_D.index.astype('str')

sns.set_style('whitegrid')
sns.set_style('whitegrid')

sns.set_style('whitegrid')
ax4 = fig.add_subplot(8,1,4)

ax4.plot(BGEP_D.index, BGEP_D['mooring_draft_raw'], alpha=1, color='lightgray', linewidth=1, label='Daily draft')
ax4.plot(BGEP_D['mooring_monthly'], color='k', label='31-days rolling average draft')
ax4.scatter(ds_BGEP_D_c2.index, ds_BGEP_D_c2['sat'], marker='.', color='royalblue', zorder=10, label='CS-2')
ax4.scatter(ds_BGEP_D_env.index, ds_BGEP_D_env['sat'], marker='.', color='darkred', zorder=12, label='Envisat')
ax4.set_title('(d) BGEP D', loc='left')
ax4.set_ylabel('Sea Ice Draft (m)')
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

stat(BGEP_D['mooring_monthly'], BGEP_D['sat'], ax4)
ax4.set_ylim(ymax=4)

ins = ax4.inset_axes([0.562, 0.1, 0.8, 0.8])
m = plot_map_trajectories_mission_NH(lon_BGEP, lat_BGEP, ins, 3, 1)

xticks=np.arange(np.datetime64("2003"), np.datetime64("2023"), np.timedelta64(1, 'Y'))
xticks = BGEP_A.loc[(BGEP_A.date.str.endswith('01-15', na='False'))].index.strftime('%Y-%m')
ax4.set_xticks(xticks)
ax4.set_xticklabels(xticks, rotation=25, horizontalalignment='right')


sns.set_style('whitegrid')
ax1 = fig.add_subplot(8,1,1, sharex=ax4)
ax1.plot(BGEP_A.index, BGEP_A['mooring_draft_raw'], alpha=1, color='lightgray', linewidth=1, label='Daily draft')
ax1.plot(BGEP_A['mooring_monthly'], color='k', label='31-days rolling average draft')
ax1.scatter(ds_BGEP_A_c2.index, ds_BGEP_A_c2['sat'], marker='.', color='royalblue', zorder=1000, label='CS-2')
ax1.scatter(ds_BGEP_A_env.index, ds_BGEP_A_env['sat'], marker='.', color='darkred', zorder=1000, label='Envisat')
ax1.set_title('(a) BGEP A', loc='left')
stat(BGEP_A['mooring_monthly'], BGEP_A['sat'], ax1)
xticks = BGEP_A.loc[(BGEP_A.date.str.endswith('01-15', na='False'))].index.values
ax1.set_ylim(ymax=4)

ins = ax1.inset_axes([0.562, 0.1, 0.8, 0.8])
m = plot_map_trajectories_mission_NH(lon_BGEP, lat_BGEP, ins, 0, 1)
ax1.set_ylabel('Sea Ice Draft (m)')
plt.setp(ax1.get_xticklabels(), visible=False)

sns.set_style('whitegrid')
ax3 = fig.add_subplot(8,1,3, sharex=ax4)

ax3.plot(BGEP_C.index, BGEP_C['mooring_draft_raw'], alpha=1, color='lightgray', linewidth=1, label='Daily draft')
ax3.plot(BGEP_C['mooring_monthly'], color='k', label='31-days rolling average draft')
ax3.scatter(ds_BGEP_C_env.index, ds_BGEP_C_env['sat'], marker='.', color='darkred', zorder=12, label='Envisat')
ax3.set_title('(c) BGEP C', loc='left')

stat(BGEP_C['mooring_monthly'], BGEP_C['sat'], ax3)
ax3.set_ylabel('Sea Ice Draft (m)')
xticks = BGEP_C.loc[(BGEP_C.date.str.endswith('01-15', na='False'))].index.values
ax3.set_xticks(xticks)
ax3.set_xticklabels(xticks, rotation=45, horizontalalignment='right')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.set_ylim(ymax=4)

ins = ax3.inset_axes([0.562, 0.1, 0.8, 0.8])
m = plot_map_trajectories_mission_NH(lon_BGEP, lat_BGEP, ins, 2, 1)
plt.setp(ax3.get_xticklabels(), visible=False)

sns.set_style('whitegrid')
ax2 = fig.add_subplot(8,1,2, sharex=ax4)

ax2.plot(BGEP_B.index, BGEP_B['mooring_draft_raw'], alpha=1, color='lightgray', linewidth=1, label='Daily draft')
ax2.plot(BGEP_B['mooring_monthly'], color='k', label='31-days rolling average draft')
ax2.scatter(ds_BGEP_B_c2.index, ds_BGEP_B_c2['sat'], marker='.', color='royalblue', zorder=10, label='CS-2')
ax2.scatter(ds_BGEP_B_env.index, ds_BGEP_B_env['sat'], marker='.', color='darkred', zorder=12, label='Envisat')
ax2.set_title('(b) BGEP B', loc='left')

stat(BGEP_B['mooring_monthly'], BGEP_B['sat'], ax2)
ax2.set_ylabel('Sea Ice Draft (m)')
xticks = BGEP_B.loc[(BGEP_B.date.str.endswith('01-15', na='False'))].index.values
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticks, rotation=45, horizontalalignment='right')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.set_ylim(ymax=4)
ins = ax2.inset_axes([0.562, 0.1, 0.8, 0.8])
m = plot_map_trajectories_mission_NH(lon_BGEP, lat_BGEP, ins, 1, 1)
plt.setp(ax2.get_xticklabels(), visible=False)


handlesb, labelsb = plt.gca().get_legend_handles_labels()
by_labelb = dict(zip(labelsb, handlesb))
labels.append(labelsb[2])
handles.append(handlesb[2])
by_labelb = dict(zip(labels, handles))
fig.legend(by_labelb.values(), by_labelb.keys(), loc='upper left', ncol=6, frameon=False)
plt.savefig('BGEP_AWI.pdf')

plt.show()
