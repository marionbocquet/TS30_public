"""
Created by Marion Bocquet
Date : 15/04/2024
Credits : LEGOS/CNES/CLS
The script aims to plot the time series and create pdfs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plots_TS_function as pts



start_date = '199401'
end_date = '202306'
PATH = './DATA/'

##### LOAD DATA #####

NH_vol_ers1r = pd.read_csv(PATH + 'W99m_ers1r_corr.csv')
NH_vol_ers1r_rft = pd.read_csv(PATH + 'rft_W99m_ers1r_corr.csv')
NH_vol_ers1r_rft = pts.fill_df_nan(NH_vol_ers1r_rft, start_date, end_date)
NH_vol_ers2r_rft = pd.read_csv(PATH + 'rft_W99m_ers2r_corr.csv')
NH_vol_ers2r_rft = pts.fill_df_nan(NH_vol_ers2r_rft, start_date, end_date)
NH_vol_env3_rft = pd.read_csv(PATH + 'rft_W99m_env3_corr.csv')
NH_vol_env3_rft = pts.fill_df_nan(NH_vol_env3_rft, start_date, end_date)
NH_vol_c2_rft = pd.read_csv(PATH + 'rft_W99m_c2esaD1.csv')
NH_vol_c2_rft = pts.fill_df_nan(NH_vol_c2_rft, start_date, end_date)

NH_vol_ers1r = pts.fill_df_nan(NH_vol_ers1r, start_date, end_date)
NH_mc_vol_ers1r = pd.read_csv(PATH + 'W99_STAT_ers1r.csv')
NH_mc_vol_ers1r = NH_mc_vol_ers1r.set_index('Unnamed: 0')
NH_vol_ers1r_q025 = NH_mc_vol_ers1r.loc['5%']
NH_vol_ers1r_q025 = pts.fill_df_nan(NH_vol_ers1r_q025, start_date, end_date)
NH_vol_ers1r_q975 = NH_mc_vol_ers1r.loc['95%']
NH_vol_ers1r_q975 = pts.fill_df_nan(NH_vol_ers1r_q975, start_date, end_date)

NH_vol_ers2r = pd.read_csv(PATH + 'W99m_ers2r_corr.csv')
NH_vol_ers2r = pts.fill_df_nan(NH_vol_ers2r, start_date, end_date)
NH_mc_vol_ers2r = pd.read_csv(PATH + 'W99_STAT_ers2r.csv')
NH_mc_vol_ers2r = NH_mc_vol_ers2r.set_index('Unnamed: 0')
NH_vol_ers2r_q025 = NH_mc_vol_ers2r.loc['5%']
NH_vol_ers2r_q025 = pts.fill_df_nan(NH_vol_ers2r_q025, start_date, end_date)
NH_vol_ers2r_q975 = NH_mc_vol_ers2r.loc['95%']
NH_vol_ers2r_q975 = pts.fill_df_nan(NH_vol_ers2r_q975, start_date, end_date)

NH_vol_env3 = pd.read_csv(PATH + 'W99m_env3_corr.csv')
NH_vol_env3 = pts.fill_df_nan(NH_vol_env3, start_date, end_date)
NH_mc_vol_env3 = pd.read_csv(PATH + 'W99_STAT_env3.csv')
NH_mc_vol_env3 = NH_mc_vol_env3.set_index('Unnamed: 0')
NH_vol_env3_q025 = NH_mc_vol_env3.loc['5%']
NH_vol_env3_q025 = pts.fill_df_nan(NH_vol_env3_q025, start_date, end_date)
NH_vol_env3_q975 = NH_mc_vol_env3.loc['95%']
NH_vol_env3_q975 = pts.fill_df_nan(NH_vol_env3_q975, start_date, end_date)

NH_vol_c2 = pd.read_csv(PATH + 'W99m_c2esaD1.csv')
NH_vol_c2 = pts.fill_df_nan(NH_vol_c2, start_date, end_date)
NH_mc_vol_c2 = pd.read_csv(PATH + 'W99_STAT_c2esaD1.csv')
NH_mc_vol_c2 = NH_mc_vol_c2.set_index('Unnamed: 0')
NH_vol_c2_q025 = NH_mc_vol_c2.loc['5%']
NH_vol_c2_q025 = pts.fill_df_nan(NH_vol_c2_q025, start_date, end_date)
NH_vol_c2_q975 = NH_mc_vol_c2.loc['95%']
NH_vol_c2_q975 = pts.fill_df_nan(NH_vol_c2_q975, start_date, end_date)

NH_vol_piomas = pd.read_csv(PATH + 'NH_volume_piomas.csv')
NH_vol_nextsim = pd.read_csv(PATH + 'NH_volume_nextsim.csv')
NH_vol_nextsim['time'] = pd.to_datetime(NH_vol_nextsim.time)

NH_vol_cs2smos = pd.read_csv(PATH + 'NH_volume_cs2smos.csv')
NH_vol_cs2_cci = pd.read_csv(PATH + 'NH_volume_cs2_cci.csv')
NH_vol_cs2_cci = pts.fill_df_nan(NH_vol_cs2_cci, '199401', '202306')

NH_vol_env_cci = pd.read_csv(PATH + 'NH_volume_env_cci.csv')
NH_vol_env_cci = pts.fill_df_nan(NH_vol_env_cci, '199401', '202306')

NH_vol_global_models = pd.read_csv(PATH + 'NH_volume_global_models.csv')

SH_vol_ers1r = pd.read_csv(PATH + 'ers1rSH_corr.csv')
SH_vol_ers1r = pts.fill_df_nan(SH_vol_ers1r, start_date, end_date)
SH_mc_vol_ers1r = pd.read_csv(PATH + 'STAT_ers1rSH_corr.csv')
SH_mc_vol_ers1r = SH_mc_vol_ers1r.set_index('Unnamed: 0')
SH_vol_ers1r_q025 = SH_mc_vol_ers1r.loc['5%']
SH_vol_ers1r_q025 = pts.fill_df_nan(SH_vol_ers1r_q025, start_date, end_date)
SH_vol_ers1r_q975 = SH_mc_vol_ers1r.loc['95%']
SH_vol_ers1r_q975 = pts.fill_df_nan(SH_vol_ers1r_q975, start_date, end_date)

SH_vol_ers2r = pd.read_csv(PATH + 'ers2rSH_corr.csv')
SH_vol_ers2r = pts.fill_df_nan(SH_vol_ers2r, start_date, end_date)
SH_mc_vol_ers2r = pd.read_csv(PATH + 'STAT_ers2rSH_corr.csv')
SH_mc_vol_ers2r = SH_mc_vol_ers2r.set_index('Unnamed: 0')
SH_vol_ers2r_q025 = SH_mc_vol_ers2r.loc['5%']
SH_vol_ers2r_q025 = pts.fill_df_nan(SH_vol_ers2r_q025, start_date, end_date)
SH_vol_ers2r_q975 = SH_mc_vol_ers2r.loc['95%']
SH_vol_ers2r_q975 = pts.fill_df_nan(SH_vol_ers2r_q975, start_date, end_date)

SH_vol_env3 = pd.read_csv(PATH + 'env3SH_corr.csv')
SH_vol_env3 = pts.fill_df_nan(SH_vol_env3, start_date, end_date)
SH_mc_vol_env3 = pd.read_csv(PATH + 'STAT_env3SH_corr.csv')
SH_mc_vol_env3 = SH_mc_vol_env3.set_index('Unnamed: 0')
SH_vol_env3_q025 = SH_mc_vol_env3.loc['5%']
SH_vol_env3_q025 = pts.fill_df_nan(SH_vol_env3_q025, start_date, end_date)
SH_vol_env3_q975 = SH_mc_vol_env3.loc['95%']
SH_vol_env3_q975 = pts.fill_df_nan(SH_vol_env3_q975, start_date, end_date)

SH_vol_c2 = pd.read_csv(PATH + 'c2esaDSH1_SARpIN.csv')
SH_vol_c2 = pts.fill_df_nan(SH_vol_c2, start_date, end_date)
SH_mc_vol_c2 = pd.read_csv(PATH + 'STAT_c2esaDSH1_SARpIN.csv')
SH_mc_vol_c2 = SH_mc_vol_c2.set_index('Unnamed: 0')
SH_vol_c2_q025 = SH_mc_vol_c2.loc['5%']
SH_vol_c2_q025 = pts.fill_df_nan(SH_vol_c2_q025, start_date, end_date)
SH_vol_c2_q975 = SH_mc_vol_c2.loc['95%']
SH_vol_c2_q975 = pts.fill_df_nan(SH_vol_c2_q975, start_date, end_date)

SH_vol_cs2_cci = pd.read_csv(PATH + 'SH_volume_cs2_cci.csv')
SH_vol_env_cci = pd.read_csv(PATH + 'SH_volume_env_cci.csv')

SH_vol_global_models = pd.read_csv(PATH + 'SH_volume_global_models.csv')
SH_vol_giomas = pd.read_csv(PATH +  'SH_volume_giomas.csv')


print('All .csv are loaded')
#####--- END LOAD DATA #####


####### Deals with overlaps #####

dates_overlap_NH = [np.datetime64('1995-10-15'), np.datetime64('1995-11-15'),
                    np.datetime64('1995-12-15'), np.datetime64('1996-01-15'),
                    np.datetime64('1996-02-15'), np.datetime64('1996-03-15'),
                    np.datetime64('1996-04-15'), np.datetime64('2002-10-15'),
                    np.datetime64('2002-11-15'), np.datetime64('2002-12-15'),
                    np.datetime64('2003-01-15'), np.datetime64('2003-02-15'),
                    np.datetime64('2003-03-15'), np.datetime64('2003-04-15'),
                    np.datetime64('2010-11-15'), np.datetime64('2010-12-15'),
                    np.datetime64('2011-01-15'), np.datetime64('2011-02-15'),
                    np.datetime64('2011-03-15'), np.datetime64('2011-04-15'),
                    np.datetime64('2011-10-15'), np.datetime64('2011-11-15'),
                    np.datetime64('2011-12-15'), np.datetime64('2012-01-15'),
                    np.datetime64('2012-02-15'), np.datetime64('2012-03-15')]
dates_drop_overlap_ers1_NH = [np.datetime64('1995-10-15'), np.datetime64('1995-11-15'),
                              np.datetime64('1995-12-15'), np.datetime64('1996-01-15'),
                              np.datetime64('1996-02-15'), np.datetime64('1996-03-15'),
                              np.datetime64('1996-04-15')]
dates_drop_overlap_ers2_NH = [np.datetime64('2002-10-15'), np.datetime64('2002-11-15'),
                              np.datetime64('2002-12-15'), np.datetime64('2003-01-15'),
                              np.datetime64('2003-02-15'), np.datetime64('2003-03-15'),
                              np.datetime64('2003-04-15')]
dates_drop_overlap_env3_NH = [np.datetime64('2011-10-15'), np.datetime64('2011-11-15'),
                              np.datetime64('2011-12-15'), np.datetime64('2012-01-15'),
                              np.datetime64('2012-02-15'), np.datetime64('2012-03-15')]
dates_drop_overlap_cs2_NH = [np.datetime64('2010-11-15'), np.datetime64('2010-12-15'),
                             np.datetime64('2011-01-15'), np.datetime64('2011-02-15'),
                             np.datetime64('2011-03-15'), np.datetime64('2011-04-15')]

dates_overlap_SH = [np.datetime64('1995-05-15'), np.datetime64('1995-06-15'),
                    np.datetime64('1995-07-15'), np.datetime64('1995-08-15'),
                    np.datetime64('1995-09-15'), np.datetime64('1995-10-15'),
                    np.datetime64('1995-11-15'), np.datetime64('1995-12-15'),
                    np.datetime64('1996-01-15'), np.datetime64('1996-02-15'),
                    np.datetime64('1996-03-15'), np.datetime64('1996-04-15'),
                    np.datetime64('1996-05-15'), np.datetime64('2002-06-15'),
                    np.datetime64('2002-07-15'), np.datetime64('2002-08-15'),
                    np.datetime64('2002-09-15'), np.datetime64('2002-10-15'),
                    np.datetime64('2002-11-15'), np.datetime64('2002-12-15'),
                    np.datetime64('2003-01-15'), np.datetime64('2003-02-15'),
                    np.datetime64('2003-03-15'), np.datetime64('2003-04-15'),
                    np.datetime64('2003-05-15'), np.datetime64('2010-12-15'),
                    np.datetime64('2011-01-15'), np.datetime64('2011-02-15'),
                    np.datetime64('2011-03-15'), np.datetime64('2011-04-15'),
                    np.datetime64('2011-05-15'), np.datetime64('2011-06-15'),
                    np.datetime64('2011-07-15'), np.datetime64('2011-08-15'),
                    np.datetime64('2011-09-15'), np.datetime64('2011-10-15'),
                    np.datetime64('2011-11-15'), np.datetime64('2011-12-15'),
                    np.datetime64('2012-01-15'), np.datetime64('2012-02-15'),
                    np.datetime64('2012-03-15')]
dates_drop_overlap_ers1_SH = [np.datetime64('1995-05-15'), np.datetime64('1995-06-15'),
                              np.datetime64('1995-07-15'), np.datetime64('1995-08-15'),
                              np.datetime64('1995-09-15'), np.datetime64('1995-10-15'),
                              np.datetime64('1995-11-15'), np.datetime64('1995-12-15'),
                              np.datetime64('1996-01-15'), np.datetime64('1996-02-15'),
                              np.datetime64('1996-03-15'), np.datetime64('1996-04-15'),
                              np.datetime64('1996-05-15')]
dates_drop_overlap_ers2_SH = [np.datetime64('2002-06-15'), np.datetime64('2002-07-15'),
                              np.datetime64('2002-08-15'), np.datetime64('2002-09-15'),
                              np.datetime64('2002-10-15'), np.datetime64('2002-11-15'),
                              np.datetime64('2002-12-15'), np.datetime64('2003-01-15'),
                              np.datetime64('2003-02-15'), np.datetime64('2003-03-15'),
                              np.datetime64('2003-04-15'), np.datetime64('2003-05-15')]
dates_drop_overlap_env3_SH = [np.datetime64('2011-01-15'), np.datetime64('2011-02-15'),
                              np.datetime64('2011-03-15'), np.datetime64('2011-04-15'),
                              np.datetime64('2011-05-15'), np.datetime64('2011-06-15'),
                              np.datetime64('2011-07-15'), np.datetime64('2011-08-15'),
                              np.datetime64('2011-09-15'), np.datetime64('2011-10-15'),
                              np.datetime64('2011-11-15'), np.datetime64('2011-12-15'),
                              np.datetime64('2012-01-15'), np.datetime64('2012-02-15'),
                              np.datetime64('2012-03-15')]
dates_drop_overlap_cs2_SH = [np.datetime64('2010-12-15')]


df_volumes_NH = pts.concat_df_dealing_overlaps(dates_drop_overlap_ers1_NH,
                                               dates_drop_overlap_ers2_NH,
                                               dates_drop_overlap_env3_NH,
                                               dates_drop_overlap_cs2_NH,
                                               [NH_vol_ers1r, NH_vol_ers2r, NH_vol_env3, NH_vol_c2])

df_volumes_NH_rft = pts.concat_df_dealing_overlaps(dates_drop_overlap_ers1_NH,
                                               dates_drop_overlap_ers2_NH,
                                               dates_drop_overlap_env3_NH,
                                               dates_drop_overlap_cs2_NH,
                                               [NH_vol_ers1r_rft, NH_vol_ers2r_rft, NH_vol_env3_rft, NH_vol_c2_rft])
df_volumes_SH = pts.concat_df_dealing_overlaps(dates_drop_overlap_ers1_SH,
                                               dates_drop_overlap_ers2_SH,
                                               dates_drop_overlap_env3_SH,
                                               dates_drop_overlap_cs2_SH,
                                               [SH_vol_ers1r, SH_vol_ers2r, SH_vol_env3, SH_vol_c2])

df_volumes_NH = pts.fill_df_nan(df_volumes_NH, '199401', '202304')
df_volumes_NH_rft = pts.fill_df_nan(df_volumes_NH_rft, '199401', '202304')

df_volumes_SH = pts.fill_df_nan(df_volumes_SH, '199401', '202306')

df_volumes_NH = df_volumes_NH[df_volumes_NH.time.notna()]
df_volumes_NH_rft = df_volumes_NH_rft[df_volumes_NH_rft.time.notna()]


df_volumes_SH = df_volumes_SH[df_volumes_SH.time.notna()]
df_volumes_SH['evolume_region_1'] = df_volumes_SH['evolume'] - df_volumes_SH['evolume_region_2'] - df_volumes_SH[
    'evolume_region_3'] - df_volumes_SH['evolume_region_4'] - df_volumes_SH['evolume_region_5'] - df_volumes_SH[
                                        'evolume_region_6'] - df_volumes_SH['evolume_region_7']
df_volumes_SH['evolume_region_56'] = df_volumes_SH['evolume_region_5'] + df_volumes_SH['evolume_region_6']

df_volumes_NH_q025 = pts.concat_df_dealing_overlaps(dates_drop_overlap_ers1_NH,
                                                    dates_drop_overlap_ers2_NH,
                                                    dates_drop_overlap_env3_NH,
                                                    dates_drop_overlap_cs2_NH,
                                                    [NH_vol_ers1r_q025, NH_vol_ers2r_q025,
                                                     NH_vol_env3_q025, NH_vol_c2_q025])
df_volumes_NH_q975 = pts.concat_df_dealing_overlaps(dates_drop_overlap_ers1_NH,
                                                    dates_drop_overlap_ers2_NH,
                                                    dates_drop_overlap_env3_NH,
                                                    dates_drop_overlap_cs2_NH,
                                                    [NH_vol_ers1r_q975, NH_vol_ers2r_q975,
                                                     NH_vol_env3_q975, NH_vol_c2_q975])

df_volumes_SH_q025 = pts.concat_df_dealing_overlaps(dates_drop_overlap_ers1_SH,
                                                    dates_drop_overlap_ers2_SH,
                                                    dates_drop_overlap_env3_SH,
                                                    dates_drop_overlap_cs2_SH,
                                                    [SH_vol_ers1r_q025, SH_vol_ers2r_q025,
                                                     SH_vol_env3_q025, SH_vol_c2_q025])

df_volumes_NH_q975 = pts.concat_df_dealing_overlaps(dates_drop_overlap_ers1_NH,
                                                    dates_drop_overlap_ers2_NH,
                                                    dates_drop_overlap_env3_NH,
                                                    dates_drop_overlap_cs2_NH,
                                                    [NH_vol_ers1r_q975, NH_vol_ers2r_q975,
                                                     NH_vol_env3_q975, NH_vol_c2_q975])

df_volumes_SH_q025 = pts.concat_df_dealing_overlaps(dates_drop_overlap_ers1_SH,
                                                    dates_drop_overlap_ers2_SH,
                                                    dates_drop_overlap_env3_SH,
                                                    dates_drop_overlap_cs2_SH,
                                                    [SH_vol_ers1r_q025, SH_vol_ers2r_q025,
                                                     SH_vol_env3_q025, SH_vol_c2_q025])

df_volumes_SH_q975 = pts.concat_df_dealing_overlaps(dates_drop_overlap_ers1_SH,
                                                    dates_drop_overlap_ers2_SH,
                                                    dates_drop_overlap_env3_SH,
                                                    dates_drop_overlap_cs2_SH,
                                                    [SH_vol_ers1r_q975, SH_vol_ers2r_q975,
                                                     SH_vol_env3_q975, SH_vol_c2_q975])
NH_vol_piomas['time'] = pd.to_datetime(NH_vol_piomas.time)
NH_vol_nextsim['time'] = pd.to_datetime(NH_vol_nextsim.time)
NH_vol_cs2smos['time'] = pd.to_datetime(NH_vol_cs2smos.time)


print('Overlaps taken into account')

print('Computation of OMIP-2 means')
list_var_volume = NH_vol_global_models.columns
list_omip2 = []
list_omip1 = []

for var in list_var_volume:
        if 'omip1' in var:
             list_omip1.append(var)
        if 'omip2' in var:
            list_omip2.append(var)
volume_omip2 = NH_vol_global_models[list_omip2]
mean_omip2 = volume_omip2.mean(axis=1)
std_omip2 = volume_omip2.std(axis=1)

list_var_volume = SH_vol_global_models.columns
list_omip2 = []
list_omip1 = []

for var in list_var_volume:
        if 'omip1' in var:
             list_omip1.append(var)
        if 'omip2' in var:
            list_omip2.append(var)
volume_omip2_SH = SH_vol_global_models[list_omip2]
mean_omip2_SH = volume_omip2_SH.mean(axis=1)
std_omip2_SH = volume_omip2_SH.std(axis=1)


print('OMIP-2 has been processed')
####### End Deals with overlaps #####


NH_vol_piomas['time'] = pd.to_datetime(NH_vol_piomas.time)
NH_vol_nextsim['time'] = pd.to_datetime(NH_vol_nextsim.time)
NH_vol_nextsim = NH_vol_nextsim.set_index('time')
NH_vol_piomas = NH_vol_piomas.set_index('time')
SH_vol_global_models['time'] = pd.to_datetime(SH_vol_global_models.time)
SH_vol_global_models = SH_vol_global_models.set_index('time')
SH_vol_global_models['month'] = SH_vol_global_models.index.month
SH_vol_global_models['year'] = SH_vol_global_models.index.year
mean_omip2 = pd.DataFrame(mean_omip2, columns=['volume'])
mean_omip2['time'] = pd.to_datetime(SH_vol_global_models.index)
mean_omip2 = mean_omip2.set_index('time')
mean_omip2_SH = pd.DataFrame(mean_omip2_SH, columns=['volume'])
mean_omip2_SH['time'] = pd.to_datetime(SH_vol_global_models.index)
mean_omip2_SH = mean_omip2_SH.set_index('time')


df_volumes_NH['time'] = pd.to_datetime(df_volumes_NH.time)
df_volumes_NH = df_volumes_NH.set_index('time', drop=False)
df_volumes_NH['second'] = np.array(list(map(pts.app_total_seconds, df_volumes_NH['time'])))


df_volumes_NH_rft['time'] = pd.to_datetime(df_volumes_NH_rft.time)
df_volumes_NH_rft = df_volumes_NH_rft.set_index('time', drop=False)
df_volumes_NH_rft['second'] = np.array(list(map(pts.app_total_seconds, df_volumes_NH_rft['time'])))



# Appendice A5 et A6
pts.figure_S1_SH(df_volumes_SH, SH_vol_cs2_cci, SH_vol_env_cci, SH_vol_ers1r, SH_vol_ers2r, SH_vol_env3, SH_vol_c2)
pts.figure_S1_NH(df_volumes_NH, df_volumes_NH_rft, NH_vol_cs2smos, NH_vol_cs2_cci, NH_vol_env_cci,
                 NH_vol_ers1r, NH_vol_ers2r, NH_vol_env3, NH_vol_c2,
                 NH_vol_ers1r_rft, NH_vol_ers2r_rft, NH_vol_env3_rft, NH_vol_c2_rft)

# ---------------- Figure 8

pts.per_cat(df_volumes_NH, df_volumes_SH)


# ---------------- Figure 5

df_volumes_SH['evolume_region_56'] = df_volumes_SH.evolume_region_5 + df_volumes_SH.evolume_region_6
df_volumes_SH['time'] = pd.to_datetime(df_volumes_SH.time)
df_volumes_NH['time'] = pd.to_datetime(df_volumes_NH.time)
df_volumes_SH_anom = df_volumes_SH.groupby(df_volumes_SH.index.month).transform(lambda x: (x-x.mean()))
df_volumes_NH_anom = df_volumes_NH.groupby(df_volumes_NH.index.month).transform(lambda x: (x-x.mean()))

df_volumes_SH_anom_per = df_volumes_SH.groupby(df_volumes_SH.index.month).transform(lambda x: 100*(x-x.mean())/x.mean())
df_volumes_NH_anom_per = df_volumes_NH.groupby(df_volumes_NH.index.month).transform(lambda x: 100*(x-x.mean())/x.mean())
vol_clim_sh = df_volumes_SH.groupby(df_volumes_SH.index.month).mean()
vol_clim_nh = df_volumes_NH.groupby(df_volumes_NH.index.month).mean()

SH_vol_cs2_cci['time'] = pd.to_datetime(SH_vol_cs2_cci.time)
SH_vol_cs2_cci = SH_vol_cs2_cci.set_index('time')
SH_vol_giomas['time'] = pd.to_datetime(SH_vol_giomas.time)
SH_vol_giomas['month'] = pd.to_datetime(SH_vol_giomas.time).dt.month
SH_vol_giomas['year'] = pd.to_datetime(SH_vol_giomas.time).dt.year
SH_vol_giomas = SH_vol_giomas.set_index('time')
SH_vol_env_cci['time'] = pd.to_datetime(SH_vol_env_cci.time)
SH_vol_env_cci = SH_vol_env_cci.set_index('time')

##### seasonal cycle

df_volumes_NH['month'] = df_volumes_NH.index.month
df_volumes_SH['month'] = df_volumes_SH.index.month
df_volumes_NH['year'] = df_volumes_NH.index.year
df_volumes_SH['year'] = df_volumes_SH.index.year
mean_omip2_SH['month'] = mean_omip2_SH.index.month
mean_omip2_SH['year'] = mean_omip2_SH.index.year
NH_vol_piomas['year'] = NH_vol_piomas.index.year
NH_vol_piomas['month'] = NH_vol_piomas.index.month

NH_vol_env_cci['year'] = NH_vol_env_cci.index.year
NH_vol_env_cci['month'] = NH_vol_env_cci.index.month
NH_vol_env_cci['month_of_season'] = NH_vol_env_cci['month'].apply(pts.month_of_year_to_month_of_season_NH)
NH_vol_env_cci['year_of_winter']= NH_vol_env_cci['month'].apply(pts.month_of_year_to_month_of_season_NH_change_year) + NH_vol_env_cci['year']
NH_vol_cs2_cci['year'] = NH_vol_cs2_cci.index.year
NH_vol_cs2_cci['month'] = NH_vol_cs2_cci.index.month
NH_vol_cs2_cci['month_of_season'] = NH_vol_cs2_cci['month'].apply(pts.month_of_year_to_month_of_season_NH)
NH_vol_cs2_cci['year_of_winter']= NH_vol_cs2_cci['month'].apply(pts.month_of_year_to_month_of_season_NH_change_year) + NH_vol_cs2_cci['year']

SH_vol_env_cci['year'] = SH_vol_env_cci.index.year
SH_vol_env_cci['month'] = SH_vol_env_cci.index.month
SH_vol_env_cci['month_of_season'] = SH_vol_env_cci['month'].apply(pts.month_of_year_to_month_of_season_NH)
SH_vol_env_cci['year_of_winter']= SH_vol_env_cci['month'].apply(pts.month_of_year_to_month_of_season_NH_change_year) + SH_vol_env_cci['year']
SH_vol_cs2_cci['year'] = SH_vol_cs2_cci.index.year
SH_vol_cs2_cci['month'] = SH_vol_cs2_cci.index.month
SH_vol_cs2_cci['month_of_season'] = SH_vol_cs2_cci['month'].apply(pts.month_of_year_to_month_of_season_NH)
SH_vol_cs2_cci['year_of_winter']= SH_vol_cs2_cci['month'].apply(pts.month_of_year_to_month_of_season_NH_change_year) + SH_vol_cs2_cci['year']


NH_vol_nextsim['year'] = NH_vol_nextsim.index.year
NH_vol_nextsim['month'] = NH_vol_nextsim.index.month
NH_vol_nextsim['month_of_season'] = NH_vol_nextsim['month'].apply(pts.month_of_year_to_month_of_season_NH)
NH_vol_nextsim['year_of_winter']= NH_vol_nextsim['month'].apply(pts.month_of_year_to_month_of_season_NH_change_year) + NH_vol_nextsim['year']

NH_vol_piomas['month_of_season'] = NH_vol_piomas['month'].apply(pts.month_of_year_to_month_of_season_NH)
NH_vol_piomas['year_of_winter']= NH_vol_piomas['month'].apply(pts.month_of_year_to_month_of_season_NH_change_year) + NH_vol_piomas['year']


mean_omip2['year'] = mean_omip2.index.year
mean_omip2['month'] = mean_omip2.index.month
mean_omip2['month_of_season'] = mean_omip2['month'].apply(pts.month_of_year_to_month_of_season_NH)
mean_omip2['year_of_winter']= mean_omip2['month'].apply(pts.month_of_year_to_month_of_season_NH_change_year) + mean_omip2['year']

df_volumes_NH['month_of_season'] = df_volumes_NH['month'].apply(pts.month_of_year_to_month_of_season_NH)
df_volumes_NH['year_of_winter'] = df_volumes_NH['month'].apply(pts.month_of_year_to_month_of_season_NH_change_year) + df_volumes_NH['year']

month_NH = ['', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Aug', 'Sep']
month_SH = ['', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb']
"""pts.plot_TS_unc_clim(NH_vol_ers1r, NH_vol_ers1r_q025, NH_vol_ers1r_q975, NH_vol_ers2r, NH_vol_ers2r_q025, NH_vol_ers2r_q975,
                     NH_vol_env3, NH_vol_env3_q025, NH_vol_env3_q975, NH_vol_c2, NH_vol_c2_q025, NH_vol_c2_q975,
                     SH_vol_ers1r, SH_vol_ers1r_q025, SH_vol_ers1r_q975, SH_vol_ers2r, SH_vol_ers2r_q025,
                     SH_vol_ers2r_q975, SH_vol_env3, SH_vol_env3_q025, SH_vol_env3_q975, SH_vol_c2, SH_vol_c2_q025, SH_vol_c2_q975,
                     df_volumes_NH, month_NH, df_volumes_SH, mean_omip2_SH, SH_vol_env_cci, SH_vol_cs2_cci, month_SH)
"""
pts.plot_TS_unc_clim(NH_vol_ers1r, NH_vol_ers1r_q025, NH_vol_ers1r_q975, NH_vol_ers2r, NH_vol_ers2r_q025, NH_vol_ers2r_q975,
                     NH_vol_env3, NH_vol_env3_q025, NH_vol_env3_q975, NH_vol_c2, NH_vol_c2_q025, NH_vol_c2_q975,
                     SH_vol_ers1r, SH_vol_ers1r_q025, SH_vol_ers1r_q975, SH_vol_ers2r, SH_vol_ers2r_q025,
                     SH_vol_ers2r_q975, SH_vol_env3, SH_vol_env3_q025, SH_vol_env3_q975, SH_vol_c2, SH_vol_c2_q025, SH_vol_c2_q975,
                    df_volumes_NH, mean_omip2, NH_vol_piomas, NH_vol_env_cci, NH_vol_cs2_cci, NH_vol_nextsim, month_NH,
                    df_volumes_SH, mean_omip2_SH, SH_vol_giomas, SH_vol_env_cci, SH_vol_cs2_cci, month_SH)


# ---------------- Figure B3

fig, ax = pts.anom_NH(df_volumes_NH_anom)
pts.anom_SH(df_volumes_SH_anom, ax)
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
plt.subplots_adjust(bottom=0.05, top=0.95)

plt.savefig('Anomaly_regions_vol.pdf')


plt.show()
