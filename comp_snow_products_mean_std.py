"""
Created by Marion Bocquet
Date 15/05/2024
Credits : LEGOS/CNES/CLS

This script compares the snow depth products solutions with in situ sea ice thickness/draft -  Appendix number 1

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
    ax.text(px, py, 'N = %s\n Bias = %.3f\n Med = %.3f \n SD = %.3f\n RMSE = %.3f\n $r$ = %.3f'%(datax.shape[0], dif.mean(), dif.median(), dif.std(), rmse, corr_coef), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, bbox=props, fontsize=8)
    #plt.title('N = %s, Bias = %.3f, STD = %.3f\n, RMSE = %.3f, $R^2$ = %.3f'%(datax.shape[0], dif.mean(), dif.std(), rmse, corr_coef**2))


# There is 4 mooring locating in the fram strait
# F11 12 13 14

list_F11 = glob.glob('VALIDATION/NPI/COMBINE_*F11*.csv')
list_F12 = glob.glob('VALIDATION/NPI/COMBINE_*F12*.csv')
list_F13 = glob.glob('VALIDATION/NPI/COMBINE_*F13*.csv')
list_F14 = glob.glob('VALIDATION/NPI/COMBINE_*F14*.csv')

list_F12_AMSRW99 = glob.glob('VALIDATION/NPI/AMSRW99_*F12*.csv')
list_F11_AMSRW99 = glob.glob('VALIDATION/NPI/AMSRW99_*F11*.csv')
list_F13_AMSRW99 = glob.glob('VALIDATION/NPI/AMSRW99_*F13*.csv')
list_F14_AMSRW99 = glob.glob('VALIDATION/NPI/AMSRW99_*F14*.csv')

list_F12_LG = glob.glob('VALIDATION/NPI/LG_*F12*.csv')
list_F11_LG = glob.glob('VALIDATION/NPI/LG_*F11*.csv')
list_F13_LG = glob.glob('VALIDATION/NPI/LG_*F13*.csv')
list_F14_LG = glob.glob('VALIDATION/NPI/LG_*F14*.csv')

list_F12_ASD_clim = glob.glob('VALIDATION/NPI/ASD_clim_*F12*.csv')
list_F11_ASD_clim = glob.glob('VALIDATION/NPI/ASD_clim_*F11*.csv')
list_F13_ASD_clim = glob.glob('VALIDATION/NPI/ASD_clim_*F13*.csv')
list_F14_ASD_clim = glob.glob('VALIDATION/NPI/ASD_clim_*F14*.csv')

                             
df_F11_cs2_rft = pd.DataFrame()
df_F11_env3_rft = pd.DataFrame()
df_F12_cs2_rft = pd.DataFrame()
df_F12_env3_rft = pd.DataFrame()
df_F11_cs2 = pd.DataFrame()

df_F11_ers1 = pd.DataFrame()
df_F11_ers2 = pd.DataFrame()
df_F11_env3 = pd.DataFrame()
df_F11_cs2 = pd.DataFrame()

df_F12_ers1 = pd.DataFrame()
df_F12_ers2 = pd.DataFrame()
df_F12_env3 = pd.DataFrame()
df_F12_cs2 = pd.DataFrame()

df_F13_ers1 = pd.DataFrame()
df_F13_ers2 = pd.DataFrame()
df_F13_env3 = pd.DataFrame()
df_F13_cs2 = pd.DataFrame()

df_F14_ers1 = pd.DataFrame()
df_F14_ers2 = pd.DataFrame()
df_F14_env3 = pd.DataFrame()
df_F14_cs2 = pd.DataFrame()

df_F11_ers1_AMSRW99 = pd.DataFrame()
df_F11_ers2_AMSRW99 = pd.DataFrame()
df_F11_env3_AMSRW99 = pd.DataFrame()
df_F11_cs2_AMSRW99 = pd.DataFrame()

df_F12_ers1_AMSRW99 = pd.DataFrame()
df_F12_ers2_AMSRW99 = pd.DataFrame()
df_F12_env3_AMSRW99 = pd.DataFrame()
df_F12_cs2_AMSRW99 = pd.DataFrame()

df_F13_ers1_AMSRW99 = pd.DataFrame()
df_F13_ers2_AMSRW99 = pd.DataFrame()
df_F13_env3_AMSRW99 = pd.DataFrame()
df_F13_cs2_AMSRW99 = pd.DataFrame()

df_F14_ers1_AMSRW99 = pd.DataFrame()
df_F14_ers2_AMSRW99 = pd.DataFrame()
df_F14_env3_AMSRW99 = pd.DataFrame()
df_F14_cs2_AMSRW99 = pd.DataFrame()


df_F11_ers1_LG = pd.DataFrame()
df_F11_ers2_LG = pd.DataFrame()
df_F11_env3_LG = pd.DataFrame()
df_F11_cs2_LG = pd.DataFrame()

df_F12_ers1_LG = pd.DataFrame()
df_F12_ers2_LG = pd.DataFrame()
df_F12_env3_LG = pd.DataFrame()
df_F12_cs2_LG = pd.DataFrame()

df_F13_ers1_LG = pd.DataFrame()
df_F13_ers2_LG = pd.DataFrame()
df_F13_env3_LG = pd.DataFrame()
df_F13_cs2_LG = pd.DataFrame()

df_F14_ers1_LG = pd.DataFrame()
df_F14_ers2_LG = pd.DataFrame()
df_F14_env3_LG = pd.DataFrame()
df_F14_cs2_LG = pd.DataFrame()


df_F11_ers1_ASD_clim = pd.DataFrame()
df_F11_ers2_ASD_clim = pd.DataFrame()
df_F11_env3_ASD_clim = pd.DataFrame()
df_F11_cs2_ASD_clim = pd.DataFrame()

df_F12_ers1_ASD_clim = pd.DataFrame()
df_F12_ers2_ASD_clim = pd.DataFrame()
df_F12_env3_ASD_clim = pd.DataFrame()
df_F12_cs2_ASD_clim = pd.DataFrame()

df_F13_ers1_ASD_clim = pd.DataFrame()
df_F13_ers2_ASD_clim = pd.DataFrame()
df_F13_env3_ASD_clim = pd.DataFrame()
df_F13_cs2_ASD_clim = pd.DataFrame()

df_F14_ers1_ASD_clim = pd.DataFrame()
df_F14_ers2_ASD_clim = pd.DataFrame()
df_F14_env3_ASD_clim = pd.DataFrame()
df_F14_cs2_ASD_clim = pd.DataFrame()

BGEP_A_cs2_ASD_clim = pd.read_csv('VALIDATION/BGEP/ASD_clim_c2esaD1_SARpIN_interp_draft-ASD_clim__BGEP_A.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_B_cs2_ASD_clim = pd.read_csv('VALIDATION/BGEP/ASD_clim_c2esaD1_SARpIN_interp_draft-ASD_clim__BGEP_B.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_D_cs2_ASD_clim = pd.read_csv('VALIDATION/BGEP/ASD_clim_c2esaD1_SARpIN_interp_draft-ASD_clim__BGEP_D.csv', names=['BGEP', "sat", "type", "date"], header=0)

BGEP_A_env3_ASD_clim = pd.read_csv('VALIDATION/BGEP/ASD_clim_env3_corr_interp_draft-ASD_clim__BGEP_A.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_B_env3_ASD_clim = pd.read_csv('VALIDATION/BGEP/ASD_clim_env3_corr_interp_draft-ASD_clim__BGEP_B.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_C_env3_ASD_clim = pd.read_csv('VALIDATION/BGEP/ASD_clim_env3_corr_interp_draft-ASD_clim__BGEP_C.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_D_env3_ASD_clim = pd.read_csv('VALIDATION/BGEP/ASD_clim_env3_corr_interp_draft-ASD_clim__BGEP_D.csv', names=['BGEP', "sat", "type", "date"], header=0)

BGEP_A_cs2_AMSRW99 = pd.read_csv('VALIDATION/BGEP/AMSRW99_c2esaD1_SARpIN_interp_draft-AMSRW99__BGEP_A.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_B_cs2_AMSRW99 = pd.read_csv('VALIDATION/BGEP/AMSRW99_c2esaD1_SARpIN_interp_draft-AMSRW99__BGEP_B.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_D_cs2_AMSRW99 = pd.read_csv('VALIDATION/BGEP/AMSRW99_c2esaD1_SARpIN_interp_draft-AMSRW99__BGEP_D.csv', names=['BGEP', "sat", "type", "date"], header=0)

BGEP_A_env3_AMSRW99 = pd.read_csv('VALIDATION/BGEP/AMSRW99_env3_corr_interp_draft-AMSRW99__BGEP_A.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_B_env3_AMSRW99 = pd.read_csv('VALIDATION/BGEP/AMSRW99_env3_corr_interp_draft-AMSRW99__BGEP_B.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_C_env3_AMSRW99 = pd.read_csv('VALIDATION/BGEP/AMSRW99_env3_corr_interp_draft-AMSRW99__BGEP_C.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_D_env3_AMSRW99 = pd.read_csv('VALIDATION/BGEP/AMSRW99_env3_corr_interp_draft-AMSRW99__BGEP_D.csv', names=['BGEP', "sat", "type", "date"], header=0)

BGEP_A_cs2_LG = pd.read_csv('VALIDATION/BGEP/LG_c2esaD1_SARpIN_interp_draft-LG__BGEP_A.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_B_cs2_LG = pd.read_csv('VALIDATION/BGEP/LG_c2esaD1_SARpIN_interp_draft-LG__BGEP_B.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_D_cs2_LG = pd.read_csv('VALIDATION/BGEP/LG_c2esaD1_SARpIN_interp_draft-LG__BGEP_D.csv', names=['BGEP', "sat", "type", "date"], header=0)

BGEP_A_env3_LG = pd.read_csv('VALIDATION/BGEP/LG_env3_corr_interp_draft-LG__BGEP_A.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_B_env3_LG = pd.read_csv('VALIDATION/BGEP/LG_env3_corr_interp_draft-LG__BGEP_B.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_C_env3_LG = pd.read_csv('VALIDATION/BGEP/LG_env3_corr_interp_draft-LG__BGEP_C.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_D_env3_LG = pd.read_csv('VALIDATION/BGEP/LG_env3_corr_interp_draft-LG__BGEP_D.csv', names=['BGEP', "sat", "type", "date"], header=0)

BGEP_A_cs2 = pd.read_csv('VALIDATION/BGEP/COMBINE_c2esaD1_SARpIN_interp_draft__BGEP_A.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_B_cs2 = pd.read_csv('VALIDATION/BGEP/COMBINE_c2esaD1_SARpIN_interp_draft__BGEP_B.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_D_cs2 = pd.read_csv('VALIDATION/BGEP/COMBINE_c2esaD1_SARpIN_interp_draft__BGEP_D.csv', names=['BGEP', "sat", "type", "date"], header=0)

BGEP_A_env3 = pd.read_csv('VALIDATION/BGEP/COMBINE_env3_corr_interp_draft__BGEP_A.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_B_env3 = pd.read_csv('VALIDATION/BGEP/COMBINE_env3_corr_interp_draft__BGEP_B.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_C_env3 = pd.read_csv('VALIDATION/BGEP/COMBINE_env3_corr_interp_draft__BGEP_C.csv', names=['BGEP', "sat", "type", "date"], header=0)
BGEP_D_env3 = pd.read_csv('VALIDATION/BGEP/COMBINE_env3_corr_interp_draft__BGEP_D.csv', names=['BGEP', "sat", "type", "date"], header=0)

TS = xr.open_zarr('/Users/marionbocquet/PycharmProjects/TS30/data/NH_W99m/unc_all_sats_NH_199301_202304.zarr')


for f in range(len(list_F11)):
    if 'ers1r' in list_F11[f]:
        df_temp = pd.read_csv(list_F11[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F11_ers1 = pd.concat([df_F11_ers1, df_temp], ignore_index=False, axis=0)
        
    if 'ers2r' in list_F11[f]:
        df_temp = pd.read_csv(list_F11[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F11_ers2 = pd.concat([df_F11_ers2, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F11[f]:
        df_temp = pd.read_csv(list_F11[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F11_env3 = pd.concat([df_F11_env3, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F11[f]:
        df_temp = pd.read_csv(list_F11[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F11_cs2 = pd.concat([df_F11_cs2, df_temp], ignore_index=False, axis=0)


for f in range(len(list_F12)):
    if 'ers1r' in list_F12[f]:
        df_temp = pd.read_csv(list_F12[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F12_ers1 = pd.concat([df_F12_ers1, df_temp], ignore_index=False, axis=0)

    if 'ers2r' in list_F12[f]:
        df_temp = pd.read_csv(list_F12[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F12_ers2 = pd.concat([df_F12_ers2, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F12[f]:
        df_temp = pd.read_csv(list_F12[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F12_env3 = pd.concat([df_F12_env3, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F12[f]:
        df_temp = pd.read_csv(list_F12[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F12_cs2 = pd.concat([df_F12_cs2, df_temp], ignore_index=False, axis=0)

for f in range(len(list_F13)):
    if 'ers1r' in list_F13[f]:
        df_temp = pd.read_csv(list_F13[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F13_ers1 = pd.concat([df_F13_ers1, df_temp], ignore_index=False, axis=0)

    if 'ers2r' in list_F13[f]:
        df_temp = pd.read_csv(list_F13[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F13_ers2 = pd.concat([df_F13_ers2, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F13[f]:
        df_temp = pd.read_csv(list_F13[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F13_env3 = pd.concat([df_F13_env3, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F13[f]:
        df_temp = pd.read_csv(list_F13[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F13_cs2 = pd.concat([df_F13_cs2, df_temp], ignore_index=False, axis=0)

for f in range(len(list_F14)):
    if 'ers1r' in list_F14[f]:
        df_temp = pd.read_csv(list_F14[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F14_ers1 = pd.concat([df_F14_ers1, df_temp], ignore_index=False, axis=0)

    if 'ers2r' in list_F14[f]:
        df_temp = pd.read_csv(list_F14[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F14_ers2 = pd.concat([df_F14_ers2, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F14[f]:
        df_temp = pd.read_csv(list_F14[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F14_env3 = pd.concat([df_F14_env3, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F14[f]:
        df_temp = pd.read_csv(list_F14[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F14_cs2 = pd.concat([df_F14_cs2, df_temp], ignore_index=False, axis=0)

for f in range(len(list_F11_AMSRW99)):
    if 'ers1r' in list_F11_AMSRW99[f]:
        df_temp = pd.read_csv(list_F11_AMSRW99[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F11_ers1_AMSRW99 = pd.concat([df_F11_ers1_AMSRW99, df_temp], ignore_index=False, axis=0)
        
    if 'ers2r' in list_F11_AMSRW99[f]:
        df_temp = pd.read_csv(list_F11_AMSRW99[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F11_ers2_AMSRW99 = pd.concat([df_F11_ers2_AMSRW99, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F11_AMSRW99[f]:
        df_temp = pd.read_csv(list_F11_AMSRW99[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F11_env3_AMSRW99 = pd.concat([df_F11_env3_AMSRW99, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F11_AMSRW99[f]:
        df_temp = pd.read_csv(list_F11_AMSRW99[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F11_cs2_AMSRW99 = pd.concat([df_F11_cs2_AMSRW99, df_temp], ignore_index=False, axis=0)

for f in range(len(list_F12_AMSRW99)):
    if 'ers1r' in list_F12_AMSRW99[f]:
        df_temp = pd.read_csv(list_F12_AMSRW99[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F12_ers1_AMSRW99 = pd.concat([df_F12_ers1_AMSRW99, df_temp], ignore_index=False, axis=0)
        
    if 'ers2r' in list_F12_AMSRW99[f]:
        df_temp = pd.read_csv(list_F12_AMSRW99[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F12_ers2_AMSRW99 = pd.concat([df_F12_ers2_AMSRW99, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F12_AMSRW99[f]:
        df_temp = pd.read_csv(list_F12_AMSRW99[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F12_env3_AMSRW99 = pd.concat([df_F12_env3_AMSRW99, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F12_AMSRW99[f]:
        df_temp = pd.read_csv(list_F12_AMSRW99[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F12_cs2_AMSRW99 = pd.concat([df_F12_cs2_AMSRW99, df_temp], ignore_index=False, axis=0)

for f in range(len(list_F13_AMSRW99)):
    if 'ers1r' in list_F13_AMSRW99[f]:
        df_temp = pd.read_csv(list_F13_AMSRW99[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F13_ers1_AMSRW99 = pd.concat([df_F13_ers1_AMSRW99, df_temp], ignore_index=False, axis=0)
        
    if 'ers2r' in list_F13_AMSRW99[f]:
        df_temp = pd.read_csv(list_F13_AMSRW99[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F13_ers2_AMSRW99 = pd.concat([df_F13_ers2_AMSRW99, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F13_AMSRW99[f]:
        df_temp = pd.read_csv(list_F13_AMSRW99[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F13_env3_AMSRW99 = pd.concat([df_F13_env3_AMSRW99, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F13_AMSRW99[f]:
        df_temp = pd.read_csv(list_F13_AMSRW99[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F13_cs2_AMSRW99 = pd.concat([df_F13_cs2_AMSRW99, df_temp], ignore_index=False, axis=0)

for f in range(len(list_F14_AMSRW99)):
    if 'ers1r' in list_F14_AMSRW99[f]:
        df_temp = pd.read_csv(list_F14_AMSRW99[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F14_ers1_AMSRW99 = pd.concat([df_F14_ers1_AMSRW99, df_temp], ignore_index=False, axis=0)
        
    if 'ers2r' in list_F14_AMSRW99[f]:
        df_temp = pd.read_csv(list_F14_AMSRW99[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F14_ers2_AMSRW99 = pd.concat([df_F14_ers2_AMSRW99, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F14_AMSRW99[f]:
        df_temp = pd.read_csv(list_F14_AMSRW99[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F14_env3_AMSRW99 = pd.concat([df_F14_env3_AMSRW99, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F14_AMSRW99[f]:
        df_temp = pd.read_csv(list_F14_AMSRW99[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F14_cs2_AMSRW99 = pd.concat([df_F14_cs2_AMSRW99, df_temp], ignore_index=False, axis=0)

for f in range(len(list_F11_LG)):
    if 'ers1r' in list_F11_LG[f]:
        df_temp = pd.read_csv(list_F11_LG[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F11_ers1_LG = pd.concat([df_F11_ers1_LG, df_temp], ignore_index=False, axis=0)
        
    if 'ers2r' in list_F11_LG[f]:
        df_temp = pd.read_csv(list_F11_LG[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F11_ers2_LG = pd.concat([df_F11_ers2_LG, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F11_LG[f]:
        df_temp = pd.read_csv(list_F11_LG[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F11_env3_LG = pd.concat([df_F11_env3_LG, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F11_LG[f]:
        df_temp = pd.read_csv(list_F11_LG[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F11_cs2_LG = pd.concat([df_F11_cs2_LG, df_temp], ignore_index=False, axis=0)

for f in range(len(list_F12_LG)):
    if 'ers1r' in list_F12_LG[f]:
        df_temp = pd.read_csv(list_F12_LG[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F12_ers1_LG = pd.concat([df_F12_ers1_LG, df_temp], ignore_index=False, axis=0)
        
    if 'ers2r' in list_F12_LG[f]:
        df_temp = pd.read_csv(list_F12_LG[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F12_ers2_LG = pd.concat([df_F12_ers2_LG, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F12_LG[f]:
        df_temp = pd.read_csv(list_F12_LG[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F12_env3_LG = pd.concat([df_F12_env3_LG, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F12_LG[f]:
        df_temp = pd.read_csv(list_F12_LG[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F12_cs2_LG = pd.concat([df_F12_cs2_LG, df_temp], ignore_index=False, axis=0)

for f in range(len(list_F13_LG)):
    if 'ers1r' in list_F13_LG[f]:
        df_temp = pd.read_csv(list_F13_LG[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F13_ers1_LG = pd.concat([df_F13_ers1_LG, df_temp], ignore_index=False, axis=0)
        
    if 'ers2r' in list_F13_LG[f]:
        df_temp = pd.read_csv(list_F13_LG[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F13_ers2_LG = pd.concat([df_F13_ers2_LG, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F13_LG[f]:
        df_temp = pd.read_csv(list_F13_LG[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F13_env3_LG = pd.concat([df_F13_env3_LG, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F13_LG[f]:
        df_temp = pd.read_csv(list_F13_LG[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F13_cs2_LG = pd.concat([df_F13_cs2_LG, df_temp], ignore_index=False, axis=0)

for f in range(len(list_F14_LG)):
    if 'ers1r' in list_F14_LG[f]:
        df_temp = pd.read_csv(list_F14_LG[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F14_ers1_LG = pd.concat([df_F14_ers1_LG, df_temp], ignore_index=False, axis=0)
        
    if 'ers2r' in list_F14_LG[f]:
        df_temp = pd.read_csv(list_F14_LG[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F14_ers2_LG = pd.concat([df_F14_ers2_LG, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F14_LG[f]:
        df_temp = pd.read_csv(list_F14_LG[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F14_env3_LG = pd.concat([df_F14_env3_LG, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F14_LG[f]:
        df_temp = pd.read_csv(list_F14_LG[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F14_cs2_LG = pd.concat([df_F14_cs2_LG, df_temp], ignore_index=False, axis=0)

for f in range(len(list_F11_ASD_clim)):
    if 'ers1r' in list_F11_ASD_clim[f]:
        df_temp = pd.read_csv(list_F11_ASD_clim[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F11_ers1_ASD_clim = pd.concat([df_F11_ers1_ASD_clim, df_temp], ignore_index=False, axis=0)
        
    if 'ers2r' in list_F11_ASD_clim[f]:
        df_temp = pd.read_csv(list_F11_ASD_clim[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F11_ers2_ASD_clim = pd.concat([df_F11_ers2_ASD_clim, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F11_ASD_clim[f]:
        df_temp = pd.read_csv(list_F11_ASD_clim[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F11_env3_ASD_clim = pd.concat([df_F11_env3_ASD_clim, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F11_ASD_clim[f]:
        df_temp = pd.read_csv(list_F11_ASD_clim[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F11_cs2_ASD_clim = pd.concat([df_F11_cs2_ASD_clim, df_temp], ignore_index=False, axis=0)

for f in range(len(list_F12_ASD_clim)):
    if 'ers1r' in list_F12_ASD_clim[f]:
        df_temp = pd.read_csv(list_F12_ASD_clim[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F12_ers1_ASD_clim = pd.concat([df_F12_ers1_ASD_clim, df_temp], ignore_index=False, axis=0)
        
    if 'ers2r' in list_F12_ASD_clim[f]:
        df_temp = pd.read_csv(list_F12_ASD_clim[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F12_ers2_ASD_clim = pd.concat([df_F12_ers2_ASD_clim, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F12_ASD_clim[f]:
        df_temp = pd.read_csv(list_F12_ASD_clim[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F12_env3_ASD_clim = pd.concat([df_F12_env3_ASD_clim, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F12_ASD_clim[f]:
        df_temp = pd.read_csv(list_F12_ASD_clim[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F12_cs2_ASD_clim = pd.concat([df_F12_cs2_ASD_clim, df_temp], ignore_index=False, axis=0)

for f in range(len(list_F13_ASD_clim)):
    if 'ers1r' in list_F13_ASD_clim[f]:
        df_temp = pd.read_csv(list_F13_ASD_clim[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F13_ers1_ASD_clim = pd.concat([df_F13_ers1_ASD_clim, df_temp], ignore_index=False, axis=0)
        
    if 'ers2r' in list_F13_ASD_clim[f]:
        df_temp = pd.read_csv(list_F13_ASD_clim[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F13_ers2_ASD_clim = pd.concat([df_F13_ers2_ASD_clim, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F13_ASD_clim[f]:
        df_temp = pd.read_csv(list_F13_ASD_clim[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F13_env3_ASD_clim = pd.concat([df_F13_env3_ASD_clim, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F13_ASD_clim[f]:
        df_temp = pd.read_csv(list_F13_ASD_clim[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F13_cs2_ASD_clim = pd.concat([df_F13_cs2_ASD_clim, df_temp], ignore_index=False, axis=0)

for f in range(len(list_F14_ASD_clim)):
    if 'ers1r' in list_F14_ASD_clim[f]:
        df_temp = pd.read_csv(list_F14_ASD_clim[f], names=['NPO', "ers1", "type", "date"], header=0)
        df_F14_ers1_ASD_clim = pd.concat([df_F14_ers1_ASD_clim, df_temp], ignore_index=False, axis=0)
        
    if 'ers2r' in list_F14_ASD_clim[f]:
        df_temp = pd.read_csv(list_F14_ASD_clim[f], names=['NPO', "ers2", "type", "date"], header=0)
        df_F14_ers2_ASD_clim = pd.concat([df_F14_ers2_ASD_clim, df_temp], ignore_index=False, axis=0)

    if 'env3' in list_F14_ASD_clim[f]:
        df_temp = pd.read_csv(list_F14_ASD_clim[f], names=['NPO', "env3", "type", "date"], header=0)
        df_F14_env3_ASD_clim = pd.concat([df_F14_env3_ASD_clim, df_temp], ignore_index=False, axis=0)
        
    if 'c2esaD' in list_F14_ASD_clim[f]:
        df_temp = pd.read_csv(list_F14_ASD_clim[f], names=['NPO', "CS2", "type", "date"], header=0)
        df_F14_cs2_ASD_clim = pd.concat([df_F14_cs2_ASD_clim, df_temp], ignore_index=False, axis=0)

#df_F11_ers1['time'] = pd.to_datetime(df_F11_ers1['date'].astype('str'), format="%Y%m")
#df_F12_ers1['time'] = pd.to_datetime(df_F12_ers1['date'].astype('str'), format="%Y%m")
#df_F13_ers1['time'] = pd.to_datetime(df_F13_ers1['date'].astype('str'), format="%Y%m")



df_F14_ers1['time'] = pd.to_datetime(df_F14_ers1['date'].astype('str'), format="%Y%m")

df_F11_ers2['time'] = pd.to_datetime(df_F11_ers2['date'].astype('str'), format="%Y%m")
df_F12_ers2['time'] = pd.to_datetime(df_F12_ers2['date'].astype('str'), format="%Y%m")
df_F13_ers2['time'] = pd.to_datetime(df_F13_ers2['date'].astype('str'), format="%Y%m")
df_F14_ers2['time'] = pd.to_datetime(df_F14_ers2['date'].astype('str'), format="%Y%m")

df_F11_env3['time'] = pd.to_datetime(df_F11_env3['date'].astype('str'), format="%Y%m")
df_F12_env3['time'] = pd.to_datetime(df_F12_env3['date'].astype('str'), format="%Y%m")
df_F13_env3['time'] = pd.to_datetime(df_F13_env3['date'].astype('str'), format="%Y%m")
df_F14_env3['time'] = pd.to_datetime(df_F14_env3['date'].astype('str'), format="%Y%m")

df_F11_cs2['time'] = pd.to_datetime(df_F11_cs2['date'].astype('str'), format="%Y%m")
df_F12_cs2['time'] = pd.to_datetime(df_F12_cs2['date'].astype('str'), format="%Y%m")
df_F13_cs2['time'] = pd.to_datetime(df_F13_cs2['date'].astype('str'), format="%Y%m")
df_F14_cs2['time'] = pd.to_datetime(df_F14_cs2['date'].astype('str'), format="%Y%m")

df_F11_env3=df_F11_env3.rename(columns = {'env3':'sat'})
df_F11_ers2=df_F11_ers2.rename(columns = {'ers2':'sat'})
df_F11_cs2=df_F11_cs2.rename(columns = {'CS2':'sat'})
df_F11 = pd.concat([df_F11_env3, df_F11_ers2, df_F11_cs2])

df_F12_env3=df_F12_env3.rename(columns = {'env3':'sat'})
df_F12_ers2=df_F12_ers2.rename(columns = {'ers2':'sat'})
df_F12_cs2=df_F12_cs2.rename(columns = {'CS2':'sat'})
df_F12 = pd.concat([df_F12_env3, df_F12_ers2, df_F12_cs2])


df_F13_env3=df_F13_env3.rename(columns = {'env3':'sat'})
df_F13_ers2=df_F13_ers2.rename(columns = {'ers2':'sat'})
df_F13_cs2=df_F13_cs2.rename(columns = {'CS2':'sat'})
df_F13 = pd.concat([df_F13_env3, df_F13_ers2, df_F13_cs2])


df_F14_env3=df_F14_env3.rename(columns = {'env3':'sat'})
df_F14_ers1=df_F14_ers1.rename(columns = {'ers1':'sat'})
df_F14_ers2=df_F14_ers2.rename(columns = {'ers2':'sat'})
df_F14_cs2=df_F14_cs2.rename(columns = {'CS2':'sat'})
df_F14 = pd.concat([df_F14_env3, df_F14_ers1, df_F14_ers2, df_F14_cs2])

# AMSRW99

df_F14_ers1_AMSRW99['time'] = pd.to_datetime(df_F14_ers1_AMSRW99['date'].astype('str'), format="%Y%m")
df_F11_ers2_AMSRW99['time'] = pd.to_datetime(df_F11_ers2_AMSRW99['date'].astype('str'), format="%Y%m")
df_F12_ers2_AMSRW99['time'] = pd.to_datetime(df_F12_ers2_AMSRW99['date'].astype('str'), format="%Y%m")
df_F13_ers2_AMSRW99['time'] = pd.to_datetime(df_F13_ers2_AMSRW99['date'].astype('str'), format="%Y%m")
df_F14_ers2_AMSRW99['time'] = pd.to_datetime(df_F14_ers2_AMSRW99['date'].astype('str'), format="%Y%m")

df_F11_env3_AMSRW99['time'] = pd.to_datetime(df_F11_env3_AMSRW99['date'].astype('str'), format="%Y%m")
df_F12_env3_AMSRW99['time'] = pd.to_datetime(df_F12_env3_AMSRW99['date'].astype('str'), format="%Y%m")
df_F13_env3_AMSRW99['time'] = pd.to_datetime(df_F13_env3_AMSRW99['date'].astype('str'), format="%Y%m")
df_F14_env3_AMSRW99['time'] = pd.to_datetime(df_F14_env3_AMSRW99['date'].astype('str'), format="%Y%m")

df_F11_cs2_AMSRW99['time'] = pd.to_datetime(df_F11_cs2_AMSRW99['date'].astype('str'), format="%Y%m")
df_F12_cs2_AMSRW99['time'] = pd.to_datetime(df_F12_cs2_AMSRW99['date'].astype('str'), format="%Y%m")
df_F13_cs2_AMSRW99['time'] = pd.to_datetime(df_F13_cs2_AMSRW99['date'].astype('str'), format="%Y%m")
df_F14_cs2_AMSRW99['time'] = pd.to_datetime(df_F14_cs2_AMSRW99['date'].astype('str'), format="%Y%m")

df_F11_env3_AMSRW99 =df_F11_env3_AMSRW99.rename(columns = {'env3':'sat'})
df_F11_ers2_AMSRW99 =df_F11_ers2_AMSRW99.rename(columns = {'ers2':'sat'})
df_F11_cs2_AMSRW99 =df_F11_cs2_AMSRW99.rename(columns = {'CS2':'sat'})
df_F11_AMSRW99 = pd.concat([df_F11_env3_AMSRW99, df_F11_ers2_AMSRW99, df_F11_cs2_AMSRW99])

df_F12_env3_AMSRW99 =df_F12_env3_AMSRW99.rename(columns = {'env3':'sat'})
df_F12_ers2_AMSRW99 =df_F12_ers2_AMSRW99.rename(columns = {'ers2':'sat'})
df_F12_cs2_AMSRW99 =df_F12_cs2_AMSRW99.rename(columns = {'CS2':'sat'})
df_F12_AMSRW99 = pd.concat([df_F12_env3_AMSRW99, df_F12_ers2_AMSRW99, df_F12_cs2_AMSRW99])


df_F13_env3_AMSRW99 = df_F13_env3_AMSRW99.rename(columns = {'env3':'sat'})
df_F13_ers2_AMSRW99 = df_F13_ers2_AMSRW99.rename(columns = {'ers2':'sat'})
df_F13_cs2_AMSRW99 = df_F13_cs2_AMSRW99.rename(columns = {'CS2':'sat'})
df_F13_AMSRW99 = pd.concat([df_F13_env3_AMSRW99, df_F13_ers2_AMSRW99, df_F13_cs2_AMSRW99])


df_F14_env3_AMSRW99 =df_F14_env3_AMSRW99.rename(columns = {'env3':'sat'})
df_F14_ers1_AMSRW99 =df_F14_ers1_AMSRW99.rename(columns = {'ers1':'sat'})
df_F14_ers2_AMSRW99 =df_F14_ers2_AMSRW99.rename(columns = {'ers2':'sat'})
df_F14_cs2_AMSRW99 =df_F14_cs2_AMSRW99.rename(columns = {'CS2':'sat'})
df_F14_AMSRW99 = pd.concat([df_F14_env3_AMSRW99, df_F14_ers1_AMSRW99, df_F14_ers2_AMSRW99, df_F14_cs2_AMSRW99])


# LG

df_F14_ers1_LG['time'] = pd.to_datetime(df_F14_ers1_LG['date'].astype('str'), format="%Y%m")
df_F11_ers2_LG['time'] = pd.to_datetime(df_F11_ers2_LG['date'].astype('str'), format="%Y%m")
df_F12_ers2_LG['time'] = pd.to_datetime(df_F12_ers2_LG['date'].astype('str'), format="%Y%m")
df_F13_ers2_LG['time'] = pd.to_datetime(df_F13_ers2_LG['date'].astype('str'), format="%Y%m")
df_F14_ers2_LG['time'] = pd.to_datetime(df_F14_ers2_LG['date'].astype('str'), format="%Y%m")

df_F11_env3_LG['time'] = pd.to_datetime(df_F11_env3_LG['date'].astype('str'), format="%Y%m")
df_F12_env3_LG['time'] = pd.to_datetime(df_F12_env3_LG['date'].astype('str'), format="%Y%m")
df_F13_env3_LG['time'] = pd.to_datetime(df_F13_env3_LG['date'].astype('str'), format="%Y%m")
df_F14_env3_LG['time'] = pd.to_datetime(df_F14_env3_LG['date'].astype('str'), format="%Y%m")

df_F11_cs2_LG['time'] = pd.to_datetime(df_F11_cs2_LG['date'].astype('str'), format="%Y%m")
df_F12_cs2_LG['time'] = pd.to_datetime(df_F12_cs2_LG['date'].astype('str'), format="%Y%m")
df_F13_cs2_LG['time'] = pd.to_datetime(df_F13_cs2_LG['date'].astype('str'), format="%Y%m")
df_F14_cs2_LG['time'] = pd.to_datetime(df_F14_cs2_LG['date'].astype('str'), format="%Y%m")

df_F11_env3_LG =df_F11_env3_LG.rename(columns = {'env3':'sat'})
df_F11_ers2_LG =df_F11_ers2_LG.rename(columns = {'ers2':'sat'})
df_F11_cs2_LG =df_F11_cs2_LG.rename(columns = {'CS2':'sat'})
df_F11_LG = pd.concat([df_F11_env3_LG, df_F11_ers2_LG, df_F11_cs2_LG])

df_F12_env3_LG =df_F12_env3_LG.rename(columns = {'env3':'sat'})
df_F12_ers2_LG =df_F12_ers2_LG.rename(columns = {'ers2':'sat'})
df_F12_cs2_LG =df_F12_cs2_LG.rename(columns = {'CS2':'sat'})
df_F12_LG = pd.concat([df_F12_env3_LG, df_F12_ers2_LG, df_F12_cs2_LG])


df_F13_env3_LG = df_F13_env3_LG.rename(columns = {'env3':'sat'})
df_F13_ers2_LG = df_F13_ers2_LG.rename(columns = {'ers2':'sat'})
df_F13_cs2_LG = df_F13_cs2_LG.rename(columns = {'CS2':'sat'})
df_F13_LG = pd.concat([df_F13_env3_LG, df_F13_ers2_LG, df_F13_cs2_LG])


df_F14_env3_LG =df_F14_env3_LG.rename(columns = {'env3':'sat'})
df_F14_ers1_LG =df_F14_ers1_LG.rename(columns = {'ers1':'sat'})
df_F14_ers2_LG =df_F14_ers2_LG.rename(columns = {'ers2':'sat'})
df_F14_cs2_LG =df_F14_cs2_LG.rename(columns = {'CS2':'sat'})
df_F14_LG = pd.concat([df_F14_env3_LG, df_F14_ers1_LG, df_F14_ers2_LG, df_F14_cs2_LG])

# ASD_clim

df_F14_ers1_ASD_clim['time'] = pd.to_datetime(df_F14_ers1_ASD_clim['date'].astype('str'), format="%Y%m")
df_F11_ers2_ASD_clim['time'] = pd.to_datetime(df_F11_ers2_ASD_clim['date'].astype('str'), format="%Y%m")
df_F12_ers2_ASD_clim['time'] = pd.to_datetime(df_F12_ers2_ASD_clim['date'].astype('str'), format="%Y%m")
df_F13_ers2_ASD_clim['time'] = pd.to_datetime(df_F13_ers2_ASD_clim['date'].astype('str'), format="%Y%m")
df_F14_ers2_ASD_clim['time'] = pd.to_datetime(df_F14_ers2_ASD_clim['date'].astype('str'), format="%Y%m")

df_F11_env3_ASD_clim['time'] = pd.to_datetime(df_F11_env3_ASD_clim['date'].astype('str'), format="%Y%m")
df_F12_env3_ASD_clim['time'] = pd.to_datetime(df_F12_env3_ASD_clim['date'].astype('str'), format="%Y%m")
df_F13_env3_ASD_clim['time'] = pd.to_datetime(df_F13_env3_ASD_clim['date'].astype('str'), format="%Y%m")
df_F14_env3_ASD_clim['time'] = pd.to_datetime(df_F14_env3_ASD_clim['date'].astype('str'), format="%Y%m")

df_F11_cs2_ASD_clim['time'] = pd.to_datetime(df_F11_cs2_ASD_clim['date'].astype('str'), format="%Y%m")
df_F12_cs2_ASD_clim['time'] = pd.to_datetime(df_F12_cs2_ASD_clim['date'].astype('str'), format="%Y%m")
df_F13_cs2_ASD_clim['time'] = pd.to_datetime(df_F13_cs2_ASD_clim['date'].astype('str'), format="%Y%m")
df_F14_cs2_ASD_clim['time'] = pd.to_datetime(df_F14_cs2_ASD_clim['date'].astype('str'), format="%Y%m")

df_F11_env3_ASD_clim =df_F11_env3_ASD_clim.rename(columns = {'env3':'sat'})
df_F11_ers2_ASD_clim =df_F11_ers2_ASD_clim.rename(columns = {'ers2':'sat'})
df_F11_cs2_ASD_clim =df_F11_cs2_ASD_clim.rename(columns = {'CS2':'sat'})
df_F11_ASD_clim = pd.concat([df_F11_env3_ASD_clim, df_F11_ers2_ASD_clim, df_F11_cs2_ASD_clim])

df_F12_env3_ASD_clim =df_F12_env3_ASD_clim.rename(columns = {'env3':'sat'})
df_F12_ers2_ASD_clim =df_F12_ers2_ASD_clim.rename(columns = {'ers2':'sat'})
df_F12_cs2_ASD_clim =df_F12_cs2_ASD_clim.rename(columns = {'CS2':'sat'})
df_F12_ASD_clim = pd.concat([df_F12_env3_ASD_clim, df_F12_ers2_ASD_clim, df_F12_cs2_ASD_clim])


df_F13_env3_ASD_clim = df_F13_env3_ASD_clim.rename(columns = {'env3':'sat'})
df_F13_ers2_ASD_clim = df_F13_ers2_ASD_clim.rename(columns = {'ers2':'sat'})
df_F13_cs2_ASD_clim = df_F13_cs2_ASD_clim.rename(columns = {'CS2':'sat'})
df_F13_ASD_clim = pd.concat([df_F13_env3_ASD_clim, df_F13_ers2_ASD_clim, df_F13_cs2_ASD_clim])


df_F14_env3_ASD_clim =df_F14_env3_ASD_clim.rename(columns = {'env3':'sat'})
df_F14_ers1_ASD_clim =df_F14_ers1_ASD_clim.rename(columns = {'ers1':'sat'})
df_F14_ers2_ASD_clim =df_F14_ers2_ASD_clim.rename(columns = {'ers2':'sat'})
df_F14_cs2_ASD_clim =df_F14_cs2_ASD_clim.rename(columns = {'CS2':'sat'})
df_F14_ASD_clim = pd.concat([df_F14_env3_ASD_clim, df_F14_ers1_ASD_clim, df_F14_ers2_ASD_clim, df_F14_cs2_ASD_clim])


# BGEP COMBINE
print(BGEP_A_cs2)
BGEP_A_cs2['time'] = pd.to_datetime(BGEP_A_cs2['date'].astype('str'), format="%Y%m")
BGEP_B_cs2['time'] = pd.to_datetime(BGEP_A_cs2['date'].astype('str'), format="%Y%m")
BGEP_D_cs2['time'] = pd.to_datetime(BGEP_A_cs2['date'].astype('str'), format="%Y%m")

BGEP_A_env3['time'] = pd.to_datetime(BGEP_A_env3['date'].astype('str'), format="%Y%m")
BGEP_B_env3['time'] = pd.to_datetime(BGEP_A_env3['date'].astype('str'), format="%Y%m")
BGEP_C_env3['time'] = pd.to_datetime(BGEP_A_env3['date'].astype('str'), format="%Y%m")
BGEP_D_env3['time'] = pd.to_datetime(BGEP_A_env3['date'].astype('str'), format="%Y%m")

# BGEP AMSRW99


BGEP_A_cs2_AMSRW99['time'] = pd.to_datetime(BGEP_A_cs2_AMSRW99['date'].astype('str'), format="%Y%m")
BGEP_B_cs2_AMSRW99['time'] = pd.to_datetime(BGEP_A_cs2_AMSRW99['date'].astype('str'), format="%Y%m")
BGEP_D_cs2_AMSRW99['time'] = pd.to_datetime(BGEP_A_cs2_AMSRW99['date'].astype('str'), format="%Y%m")

BGEP_A_env3_AMSRW99['time'] = pd.to_datetime(BGEP_A_env3_AMSRW99['date'].astype('str'), format="%Y%m")
BGEP_B_env3_AMSRW99['time'] = pd.to_datetime(BGEP_A_env3_AMSRW99['date'].astype('str'), format="%Y%m")
BGEP_C_env3_AMSRW99['time'] = pd.to_datetime(BGEP_A_env3_AMSRW99['date'].astype('str'), format="%Y%m")
BGEP_D_env3_AMSRW99['time'] = pd.to_datetime(BGEP_A_env3_AMSRW99['date'].astype('str'), format="%Y%m")

# BGEP snow-LG

BGEP_A_cs2_LG['time'] = pd.to_datetime(BGEP_A_cs2_LG['date'].astype('str'), format="%Y%m")
BGEP_B_cs2_LG['time'] = pd.to_datetime(BGEP_A_cs2_LG['date'].astype('str'), format="%Y%m")
BGEP_D_cs2_LG['time'] = pd.to_datetime(BGEP_A_cs2_LG['date'].astype('str'), format="%Y%m")

BGEP_A_env3_LG['time'] = pd.to_datetime(BGEP_A_env3_LG['date'].astype('str'), format="%Y%m")
BGEP_B_env3_LG['time'] = pd.to_datetime(BGEP_A_env3_LG['date'].astype('str'), format="%Y%m")
BGEP_C_env3_LG['time'] = pd.to_datetime(BGEP_A_env3_LG['date'].astype('str'), format="%Y%m")
BGEP_D_env3_LG['time'] = pd.to_datetime(BGEP_A_env3_LG['date'].astype('str'), format="%Y%m")

# BGEP ASD clim


BGEP_A_cs2_ASD_clim['time'] = pd.to_datetime(BGEP_A_cs2_ASD_clim['date'].astype('str'), format="%Y%m")
BGEP_B_cs2_ASD_clim['time'] = pd.to_datetime(BGEP_A_cs2_ASD_clim['date'].astype('str'), format="%Y%m")
BGEP_D_cs2_ASD_clim['time'] = pd.to_datetime(BGEP_A_cs2_ASD_clim['date'].astype('str'), format="%Y%m")

BGEP_A_env3_ASD_clim['time'] = pd.to_datetime(BGEP_A_env3_ASD_clim['date'].astype('str'), format="%Y%m")
BGEP_B_env3_ASD_clim['time'] = pd.to_datetime(BGEP_A_env3_ASD_clim['date'].astype('str'), format="%Y%m")
BGEP_C_env3_ASD_clim['time'] = pd.to_datetime(BGEP_A_env3_ASD_clim['date'].astype('str'), format="%Y%m")
BGEP_D_env3_ASD_clim['time'] = pd.to_datetime(BGEP_A_env3_ASD_clim['date'].astype('str'), format="%Y%m")


cmap = plt.cm.get_cmap('inferno')
cmap_cut = cmap(np.arange(cmap.N))[:-40]
cmap = LinearSegmentedColormap.from_list('cut', cmap_cut, cmap.N)


fig = plt.figure('COMBINE scatter', figsize=(15, 15))
sns.set_style('whitegrid')

ax1 = fig.add_subplot(4,4,1)
sns.set_style('whitegrid')

plt.scatter(df_F14_ers1['NPO'], df_F14_ers1['sat'], c = 'teal', vmin=0, vmax=1, label='ERS-1', marker='v')
plt.scatter(df_F14_ers2['NPO'], df_F14_ers2['sat'], c = 'darkorange', vmin=0, vmax=1, label='ERS-2', marker='v')
plt.scatter(df_F14_env3['NPO'], df_F14_env3['sat'], c = 'darkred', vmin=0, vmax=1, marker='v', label='Envisat')
plt.scatter(df_F14_cs2['NPO'], df_F14_cs2['sat'], c = 'royalblue', vmin=0, vmax=1, marker='v', label='CryoSat-2')
plt.legend()
plt.scatter(df_F13_ers2['NPO'], df_F13_ers2['sat'], c = 'darkorange', vmin=0, vmax=1, label='ERS-2', marker='s')
plt.scatter(df_F13_env3['NPO'], df_F13_env3['sat'], c = 'darkred', vmin=0, vmax=1, marker='s', label='Envisat')
plt.scatter(df_F13_cs2['NPO'], df_F13_cs2['sat'], c = 'royalblue', vmin=0, vmax=1, marker='s', label='CryoSat-2')

plt.scatter(df_F12_ers2['NPO'], df_F12_ers2['sat'], c = 'darkorange', vmin=0, vmax=1, label='ERS-2', marker='o')
plt.scatter(df_F12_env3['NPO'], df_F12_env3['sat'], c = 'darkred', vmin=0, vmax=1, marker='o', label='Envisat')
plt.scatter(df_F12_cs2['NPO'], df_F12_cs2['sat'], c = 'royalblue', vmin=0, vmax=1, marker='o', label='CryoSat-2')

plt.scatter(df_F11_ers2['NPO'], df_F11_ers2['sat'], c = 'darkorange', vmin=0, vmax=1, label='ERS-2', marker='*')
plt.scatter(df_F11_env3['NPO'], df_F11_env3['sat'], c = 'darkred', vmin=0, vmax=1, marker='*', label='Envisat')
plt.scatter(df_F11_cs2['NPO'], df_F11_cs2['sat'], c = 'royalblue', vmin=0, vmax=1, marker='*', label='CryoSat-2')

x = np.linspace(-0.2, 4, 10)
ax1.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
ax1.set_ylabel('TS draft (m)')
ax1.set_xlabel('NPI draft (m)')
#stat(df_F14['NPO'], df_F14['sat'], ax1, px=0.05, py=0.85)
axes = plt.gca()
axes.set_aspect('equal', adjustable='box')
plt.plot(x, x, 'r')
plt.title('Weighted mean')
df_COMBINE = pd.concat([df_F11, df_F12, df_F13, df_F14], ignore_index=False, axis=0)
stat(df_COMBINE['NPO'], df_COMBINE['sat'], ax1, px=0.05, py=0.85)

ax2 = fig.add_subplot(4,4,3)
sns.set_style('whitegrid')

plt.scatter(df_F14_ers1_AMSRW99['NPO'], df_F14_ers1_AMSRW99['sat'], c = 'teal', vmin=0, vmax=1, marker='v', label='ERS-1')#, marker='v')
plt.scatter(df_F14_ers2_AMSRW99['NPO'], df_F14_ers2_AMSRW99['sat'], c = 'darkorange', vmin=0, vmax=1, marker='v')#, label='ERS-2')
plt.scatter(df_F14_env3_AMSRW99['NPO'], df_F14_env3_AMSRW99['sat'], c = 'darkred', vmin=0, vmax=1, marker='v')#, label='Envisat')
plt.scatter(df_F14_cs2_AMSRW99['NPO'], df_F14_cs2_AMSRW99['sat'], c = 'royalblue', vmin=0, vmax=1, marker='v')#, label='CryoSat-2')

plt.scatter(df_F13_ers2_AMSRW99['NPO'], df_F13_ers2_AMSRW99['sat'], c = 'darkorange', vmin=0, vmax=1, marker='s')#, label='ERS-2')
plt.scatter(df_F13_env3_AMSRW99['NPO'], df_F13_env3_AMSRW99['sat'], c = 'darkred', vmin=0, vmax=1, marker='s')#, label='Envisat')
plt.scatter(df_F13_cs2_AMSRW99['NPO'], df_F13_cs2_AMSRW99['sat'], c = 'royalblue', vmin=0, vmax=1, marker='s')#, label='CryoSat-2')

plt.scatter(df_F12_ers2_AMSRW99['NPO'], df_F12_ers2_AMSRW99['sat'], c = 'darkorange', vmin=0, vmax=1, marker='o')#, label='ERS-2'
plt.scatter(df_F12_env3_AMSRW99['NPO'], df_F12_env3_AMSRW99['sat'], c = 'darkred', vmin=0, vmax=1, marker='o') #label='Envisat')
plt.scatter(df_F12_cs2_AMSRW99['NPO'], df_F12_cs2_AMSRW99['sat'], c = 'royalblue', vmin=0, vmax=1, marker='o') #label='CryoSat-2')

plt.scatter(df_F11_ers2_AMSRW99['NPO'], df_F11_ers2_AMSRW99['sat'], c = 'darkorange', vmin=0, vmax=1, marker='*')# label='ERS-2
plt.scatter(df_F11_env3_AMSRW99['NPO'], df_F11_env3_AMSRW99['sat'], c = 'darkred', vmin=0, vmax=1, marker='*')#, label='Envisat')
plt.scatter(df_F11_cs2_AMSRW99['NPO'], df_F11_cs2_AMSRW99['sat'], c = 'royalblue', vmin=0, vmax=1, marker='*')#, label='CryoSat-2')

x = np.linspace(-0.2, 4, 10)
ax2.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
ax2.set_ylabel('TS draft (m)')
ax2.set_xlabel('NPI draft (m)')

axes = plt.gca()
axes.set_aspect('equal', adjustable='box')
plt.plot(x, x, 'r')
df_AMSRW99 = pd.concat([df_F11_AMSRW99, df_F12_AMSRW99, df_F13_AMSRW99, df_F14_AMSRW99], ignore_index=False, axis=0)
stat(df_AMSRW99['NPO'], df_AMSRW99['sat'], ax2, px=0.05, py=0.85)
plt.title('AMSRW99 Snow depth')

ax3 = fig.add_subplot(4,4,2)
sns.set_style('whitegrid')

plt.scatter(df_F14_ers1_LG['NPO'], df_F14_ers1_LG['sat'], c = 'teal', vmin=0, vmax=1, label='ERS-1', marker='v')
plt.scatter(df_F14_ers2_LG['NPO'], df_F14_ers2_LG['sat'], c = 'darkorange', vmin=0, vmax=1, label='ERS-2', marker='v')
plt.scatter(df_F14_env3_LG['NPO'], df_F14_env3_LG['sat'], c = 'darkred', vmin=0, vmax=1, marker='v', label='Envisat')
plt.scatter(df_F14_cs2_LG['NPO'], df_F14_cs2_LG['sat'], c = 'royalblue', vmin=0, vmax=1, marker='v', label='CryoSat-2')

plt.scatter(df_F13_ers2_LG['NPO'], df_F13_ers2_LG['sat'], c = 'darkorange', vmin=0, vmax=1, label='ERS-2', marker='s')
plt.scatter(df_F13_env3_LG['NPO'], df_F13_env3_LG['sat'], c = 'darkred', vmin=0, vmax=1, marker='s', label='Envisat')
plt.scatter(df_F13_cs2_LG['NPO'], df_F13_cs2_LG['sat'], c = 'royalblue', vmin=0, vmax=1, marker='s', label='CryoSat-2')

plt.scatter(df_F12_ers2_LG['NPO'], df_F12_ers2_LG['sat'], c = 'darkorange', vmin=0, vmax=1, label='ERS-2', marker='o')
plt.scatter(df_F12_env3_LG['NPO'], df_F12_env3_LG['sat'], c = 'darkred', vmin=0, vmax=1, marker='o', label='Envisat')
plt.scatter(df_F12_cs2_LG['NPO'], df_F12_cs2_LG['sat'], c = 'royalblue', vmin=0, vmax=1, marker='o', label='CryoSat-2')

plt.scatter(df_F11_ers2_LG['NPO'], df_F11_ers2_LG['sat'], c = 'darkorange', vmin=0, vmax=1, label='ERS-2', marker='*')
plt.scatter(df_F11_env3_LG['NPO'], df_F11_env3_LG['sat'], c = 'darkred', vmin=0, vmax=1, marker='*', label='Envisat')
plt.scatter(df_F11_cs2_LG['NPO'], df_F11_cs2_LG['sat'], c = 'royalblue', vmin=0, vmax=1, marker='*', label='CryoSat-2')

x = np.linspace(-0.2, 4, 10)
ax3.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
ax3.set_ylabel('TS draft (m)')
ax3.set_xlabel('NPI draft (m)')
axes = plt.gca()
axes.set_aspect('equal', adjustable='box')
plt.plot(x, x, 'r')
df_LG = pd.concat([df_F11_LG, df_F12_LG, df_F13_LG, df_F14_LG], ignore_index=False, axis=0)
stat(df_LG['NPO'], df_LG['sat'], ax3, px=0.05, py=0.85)
plt.title('SnowModel-LG snow load')

ax4 = fig.add_subplot(4,4,4)
sns.set_style('whitegrid')

plt.scatter(df_F14_ers1_ASD_clim['NPO'], df_F14_ers1_ASD_clim['sat'], c = 'teal', vmin=0, vmax=1, label='ERS-1', marker='v')
plt.scatter(df_F14_ers2_ASD_clim['NPO'], df_F14_ers2_ASD_clim['sat'], c = 'darkorange', vmin=0, vmax=1, label='ERS-2', marker='v')
plt.scatter(df_F14_env3_ASD_clim['NPO'], df_F14_env3_ASD_clim['sat'], c = 'darkred', vmin=0, vmax=1, marker='v', label='Envisat')
plt.scatter(df_F14_cs2_ASD_clim['NPO'], df_F14_cs2_ASD_clim['sat'], c = 'royalblue', vmin=0, vmax=1, marker='v', label='CryoSat-2')

plt.scatter(df_F13_ers2_ASD_clim['NPO'], df_F13_ers2_ASD_clim['sat'], c = 'darkorange', vmin=0, vmax=1, label='ERS-2', marker='s')
plt.scatter(df_F13_env3_ASD_clim['NPO'], df_F13_env3_ASD_clim['sat'], c = 'darkred', vmin=0, vmax=1, marker='s', label='Envisat')
plt.scatter(df_F13_cs2_ASD_clim['NPO'], df_F13_cs2_ASD_clim['sat'], c = 'royalblue', vmin=0, vmax=1, marker='s', label='CryoSat-2')

plt.scatter(df_F12_ers2_ASD_clim['NPO'], df_F12_ers2_ASD_clim['sat'], c = 'darkorange', vmin=0, vmax=1, label='ERS-2', marker='o')
plt.scatter(df_F12_env3_ASD_clim['NPO'], df_F12_env3_ASD_clim['sat'], c = 'darkred', vmin=0, vmax=1, marker='o', label='Envisat')
plt.scatter(df_F12_cs2_ASD_clim['NPO'], df_F12_cs2_ASD_clim['sat'], c = 'royalblue', vmin=0, vmax=1, marker='o', label='CryoSat-2')

plt.scatter(df_F11_ers2_ASD_clim['NPO'], df_F11_ers2_ASD_clim['sat'], c = 'darkorange', vmin=0, vmax=1, label='ERS-2', marker='*')
plt.scatter(df_F11_env3_ASD_clim['NPO'], df_F11_env3_ASD_clim['sat'], c = 'darkred', vmin=0, vmax=1, marker='*', label='Envisat')
plt.scatter(df_F11_cs2_ASD_clim['NPO'], df_F11_cs2_ASD_clim['sat'], c = 'royalblue', vmin=0, vmax=1, marker='*', label='CryoSat-2')

x = np.linspace(-0.2, 4, 10)
ax4.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
ax4.set_ylabel('TS draft (m)')
ax4.set_xlabel('NPI draft (m)')

axes = plt.gca()
axes.set_aspect('equal', adjustable='box')
plt.plot(x, x, 'r')
plt.title('ASD climatology snow depth')
df_ASD_clim = pd.concat([df_F11_ASD_clim, df_F12_ASD_clim, df_F13_ASD_clim, df_F14_ASD_clim], ignore_index=False, axis=0)
stat(df_ASD_clim['NPO'], df_ASD_clim['sat'], ax4, px=0.05, py=0.85)


ax5 = fig.add_subplot(4,4,5)
sns.set_style('whitegrid')

plt.scatter(BGEP_A_env3['BGEP'], BGEP_A_env3['sat'], c = 'darkred', vmin=0, vmax=1, marker='v', label='Envisat')
plt.scatter(BGEP_A_cs2['BGEP'], BGEP_A_cs2['sat'], c = 'royalblue', vmin=0, vmax=1, marker='v', label='CryoSat-2')
plt.legend()
plt.scatter(BGEP_B_env3['BGEP'], BGEP_B_env3['sat'], c = 'darkred', vmin=0, vmax=1, marker='s', label='Envisat')
plt.scatter(BGEP_B_cs2['BGEP'], BGEP_B_cs2['sat'], c = 'royalblue', vmin=0, vmax=1, marker='s', label='CryoSat-2')

plt.scatter(BGEP_C_env3['BGEP'], BGEP_C_env3['sat'], c = 'darkred', vmin=0, vmax=1, marker='o', label='Envisat')
#plt.scatter(BGEP_C_cs2['NPO'], BGEP_C_cs2['sat'], c = 'royalblue', vmin=0, vmax=1, marker='o', label='CryoSat-2')

plt.scatter(BGEP_D_env3['BGEP'], BGEP_D_env3['sat'], c = 'darkred', vmin=0, vmax=1, marker='*', label='Envisat')
plt.scatter(BGEP_D_cs2['BGEP'], BGEP_D_cs2['sat'], c = 'royalblue', vmin=0, vmax=1, marker='*', label='CryoSat-2')

x = np.linspace(-0.2, 4, 10)
ax5.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
ax5.set_ylabel('TS draft (m)')
ax5.set_xlabel('BGEP draft (m)')

#stat(df_F14['NPO'], df_F14['sat'], ax1, px=0.05, py=0.85)
axes = plt.gca()
axes.set_aspect('equal', adjustable='box')
plt.plot(x, x, 'r')
#plt.title('Weighted mean')
BGEP = pd.concat([BGEP_A_env3, BGEP_A_cs2, BGEP_B_env3, BGEP_B_cs2, BGEP_D_env3, BGEP_D_cs2, BGEP_C_env3], ignore_index=False, axis=0)
stat(BGEP['BGEP'], BGEP['sat'], ax5, px=0.05, py=0.85)



ax6 = fig.add_subplot(4,4,7)
sns.set_style('whitegrid')

plt.scatter(BGEP_A_env3_AMSRW99['BGEP'], BGEP_A_env3_AMSRW99['sat'], c = 'darkred', vmin=0, vmax=1, marker='v', label='Envisat')
plt.scatter(BGEP_A_cs2_AMSRW99['BGEP'], BGEP_A_cs2_AMSRW99['sat'], c = 'royalblue', vmin=0, vmax=1, marker='v', label='CryoSat-2')
plt.scatter(BGEP_B_env3_AMSRW99['BGEP'], BGEP_B_env3_AMSRW99['sat'], c = 'darkred', vmin=0, vmax=1, marker='s', label='Envisat')
plt.scatter(BGEP_B_cs2_AMSRW99['BGEP'], BGEP_B_cs2_AMSRW99['sat'], c = 'royalblue', vmin=0, vmax=1, marker='s', label='CryoSat-2')

plt.scatter(BGEP_C_env3_AMSRW99['BGEP'], BGEP_C_env3_AMSRW99['sat'], c = 'darkred', vmin=0, vmax=1, marker='o', label='Envisat')
#plt.scatter(BGEP_C_cs2['NPO'], BGEP_C_cs2['sat'], c = 'royalblue', vmin=0, vmax=1, marker='o', label='CryoSat-2')

plt.scatter(BGEP_D_env3_AMSRW99['BGEP'], BGEP_D_env3_AMSRW99['sat'], c = 'darkred', vmin=0, vmax=1, marker='*', label='Envisat')
plt.scatter(BGEP_D_cs2_AMSRW99['BGEP'], BGEP_D_cs2_AMSRW99['sat'], c = 'royalblue', vmin=0, vmax=1, marker='*', label='CryoSat-2')

x = np.linspace(-0.2, 4, 10)
ax6.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
ax6.set_xlabel('BGEP draft (m)')

ax6.set_ylabel('TS draft (m)')
#stat(df_F14['NPO'], df_F14['sat'], ax1, px=0.05, py=0.85)
axes = plt.gca()
axes.set_aspect('equal', adjustable='box')
plt.plot(x, x, 'r')
#plt.title('SIT from AMSRW99 snow depth')
BGEP_AMSRW99 = pd.concat([BGEP_A_env3_AMSRW99, BGEP_A_cs2_AMSRW99, BGEP_B_env3_AMSRW99, BGEP_B_cs2_AMSRW99, 
                BGEP_D_env3_AMSRW99, BGEP_D_cs2_AMSRW99, BGEP_C_env3_AMSRW99], ignore_index=False, axis=0)
stat(BGEP_AMSRW99['BGEP'], BGEP_AMSRW99['sat'], ax6, px=0.05, py=0.85)


ax7 = fig.add_subplot(4,4,6)
sns.set_style('whitegrid')

plt.scatter(BGEP_A_env3_LG['BGEP'], BGEP_A_env3_LG['sat'], c = 'darkred', vmin=0, vmax=1, marker='v', label='Envisat')
plt.scatter(BGEP_A_cs2_LG['BGEP'], BGEP_A_cs2_LG['sat'], c = 'royalblue', vmin=0, vmax=1, marker='v', label='CryoSat-2')
plt.scatter(BGEP_B_env3_LG['BGEP'], BGEP_B_env3_LG['sat'], c = 'darkred', vmin=0, vmax=1, marker='s', label='Envisat')
plt.scatter(BGEP_B_cs2_LG['BGEP'], BGEP_B_cs2_LG['sat'], c = 'royalblue', vmin=0, vmax=1, marker='s', label='CryoSat-2')

plt.scatter(BGEP_C_env3_LG['BGEP'], BGEP_C_env3_LG['sat'], c = 'darkred', vmin=0, vmax=1, marker='o', label='Envisat')
#plt.scatter(BGEP_C_cs2['NPO'], BGEP_C_cs2['sat'], c = 'royalblue', vmin=0, vmax=1, marker='o', label='CryoSat-2')

plt.scatter(BGEP_D_env3_LG['BGEP'], BGEP_D_env3_LG['sat'], c = 'darkred', vmin=0, vmax=1, marker='*', label='Envisat')
plt.scatter(BGEP_D_cs2_LG['BGEP'], BGEP_D_cs2_LG['sat'], c = 'royalblue', vmin=0, vmax=1, marker='*', label='CryoSat-2')

x = np.linspace(-0.2, 4, 10)
ax7.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
ax7.set_ylabel('TS draft (m)')
ax7.set_xlabel('BGEP draft (m)')
#stat(df_F14['NPO'], df_F14['sat'], ax1, px=0.05, py=0.85)
axes = plt.gca()
axes.set_aspect('equal', adjustable='box')
plt.plot(x, x, 'r')
#plt.title('SIT from SnowModel-LG snow load')
BGEP_LG = pd.concat([BGEP_A_env3_LG, BGEP_A_cs2_LG, BGEP_B_env3_LG, BGEP_B_cs2_LG, 
                BGEP_D_env3_LG, BGEP_D_cs2_LG, BGEP_C_env3_LG], ignore_index=False, axis=0)
stat(BGEP_LG['BGEP'], BGEP_LG['sat'], ax7, px=0.05, py=0.85)

ax8 = fig.add_subplot(4,4,8)
sns.set_style('whitegrid')

plt.scatter(BGEP_A_env3_ASD_clim['BGEP'], BGEP_A_env3_ASD_clim['sat'], c = 'darkred', vmin=0, vmax=1, marker='v', label='Envisat')
plt.scatter(BGEP_A_cs2_ASD_clim['BGEP'], BGEP_A_cs2_ASD_clim['sat'], c = 'royalblue', vmin=0, vmax=1, marker='v', label='CryoSat-2')
plt.scatter(BGEP_B_env3_ASD_clim['BGEP'], BGEP_B_env3_ASD_clim['sat'], c = 'darkred', vmin=0, vmax=1, marker='s', label='Envisat')
plt.scatter(BGEP_B_cs2_ASD_clim['BGEP'], BGEP_B_cs2_ASD_clim['sat'], c = 'royalblue', vmin=0, vmax=1, marker='s', label='CryoSat-2')

plt.scatter(BGEP_C_env3_ASD_clim['BGEP'], BGEP_C_env3_ASD_clim['sat'], c = 'darkred', vmin=0, vmax=1, marker='o', label='Envisat')
#plt.scatter(BGEP_C_cs2['NPO'], BGEP_C_cs2['sat'], c = 'royalblue', vmin=0, vmax=1, marker='o', label='CryoSat-2')

plt.scatter(BGEP_D_env3_ASD_clim['BGEP'], BGEP_D_env3_ASD_clim['sat'], c = 'darkred', vmin=0, vmax=1, marker='*', label='Envisat')
plt.scatter(BGEP_D_cs2_ASD_clim['BGEP'], BGEP_D_cs2_ASD_clim['sat'], c = 'royalblue', vmin=0, vmax=1, marker='*', label='CryoSat-2')

x = np.linspace(-0.2, 4, 10)
ax8.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
ax8.set_ylabel('TS draft (m)')
ax8.set_xlabel('BGEP draft (m)')

#stat(df_F14['NPO'], df_F14['sat'], ax1, px=0.05, py=0.85)
axes = plt.gca()
axes.set_aspect('equal', adjustable='box')
plt.plot(x, x, 'r')
#plt.title('ASD climatology snow depth')
BGEP_ASD_clim = pd.concat([BGEP_A_env3_ASD_clim, BGEP_A_cs2_ASD_clim, BGEP_B_env3_ASD_clim, BGEP_B_cs2_ASD_clim, 
                BGEP_D_env3_ASD_clim, BGEP_D_cs2_ASD_clim, BGEP_C_env3_ASD_clim], ignore_index=False, axis=0)
stat(BGEP_ASD_clim['BGEP'], BGEP_ASD_clim['sat'], ax8, px=0.05, py=0.85)

ax9 = fig.add_subplot(4, 4, 9)
sns.set_style('whitegrid')
im = pmf.plot_map(TS.sit_interp.mean(dim='time'), 1, 0, 5, cmc.roma_r, 'Sea ice thickness (m)', orientation='horizontal')

ax10 = fig.add_subplot(4, 4, 10)
sns.set_style('whitegrid')
im_diff = pmf.plot_map(TS.sit_interp.mean(dim='time')-TS.SIT_LG_ERA.mean(dim='time'), 1, -1, 1, cmc.vik, 'Sea ice thickness (m)', orientation='horizontal')

ax11 = fig.add_subplot(4, 4, 11)
sns.set_style('whitegrid')
pmf.plot_map(TS.sit_interp.mean(dim='time')-TS.SIT_AMSRW99.mean(dim='time'), 1, -1, 1, cmc.vik, 'Sea ice thickness (m)', orientation='horizontal')

ax12 = fig.add_subplot(4, 4, 12)
sns.set_style('whitegrid')
pmf.plot_map(TS.sit_interp.mean(dim='time')-TS.SIT_ASD_clim.mean(dim='time'), 1, -1, 1, cmc.vik, 'Sea ice thickness (m)', orientation='horizontal')



ax13 = fig.add_subplot(4, 4, 13)
sns.set_style('whitegrid')
im_std = pmf.plot_map(TS.sit_interp.std(dim='time'), 1, 0, 1, cmc.davos, 'Sea ice thickness (m)', orientation='horizontal')

ax14 = fig.add_subplot(4, 4, 14)
sns.set_style('whitegrid')
im_nstd = pmf.plot_map(TS.SIT_LG_ERA.std(dim='time')/TS.sit_interp.std(dim='time'), 1, 0.5, 1.5, cmc.cork, 'Sea ice thickness (m)', orientation='horizontal')

ax15 = fig.add_subplot(4, 4, 15)
sns.set_style('whitegrid')
pmf.plot_map(TS.SIT_AMSRW99.std(dim='time')/TS.sit_interp.std(dim='time'), 1, 0.5, 1.5, cmc.cork, 'Sea ice thickness (m)', orientation='horizontal')

ax16 = fig.add_subplot(4, 4, 16)
sns.set_style('whitegrid')
pmf.plot_map(TS.SIT_ASD_clim.std(dim='time')/TS.sit_interp.std(dim='time'), 1, 0.5, 1.5, cmc.cork, 'Sea ice thickness (m)', orientation='horizontal')
#plt.subplots_adjust(bottom=0.01, top=0.99)
#plt.tight_layout()

from matplotlib import rcParams
rcParams['savefig.bbox'] = 'tight'
plt.savefig('insitu_improvement_cmap.pdf')
plt.show()

