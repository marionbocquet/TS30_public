"""
Created by Marion Bocquet
Date 15/05/2024
Credits : LEGOS/CNES/CLS

This script make the figure 3, NPI/Submarine Validation

"""
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from cmcrameri import cm

def stat(datax, datay, ax, px=0.2, py=1.15):
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
    ax.text(px, py, ' N = %s\n Bias = %.3f\n Med = %.3f \n SD = %.3f\n RMSE = %.3f\n r = %.3f'%(datax.shape[0], dif.mean(), dif.median(), dif.std(), rmse, corr_coef), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, bbox=props, fontsize=8)


def stat(datax, datay, ax, px=0.05, py=0.85):
    rmse = ((datay - datax) ** 2).mean() ** .5
    corr_coef = datax.corr(datay, method='pearson')
    dif = datay-datax
    props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.85)
    ax.text(px, py, ' N = %s\n Bias = %.3f\n Med = %.3f \n SD = %.3f\n RMSE = %.3f\n r = %.3f'%(datax.shape[0], dif.mean(), dif.median(), dif.std(), rmse, corr_coef), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, bbox=props, fontsize=8)
    #plt.title('N = %s, Bias = %.3f, STD = %.3f\n, RMSE = %.3f, $R^2$ = %.3f'%(datax.shape[0], dif.mean(), dif.std(), rmse, corr_coef**2))


def scatter_gaussian_gathered(datax, datay, ax, min_lim, max_lim):
    """

    :param datax:
    :param datay:
    :param ax:
    :param min_lim:
    :param max_lim:
    :return:
    """
    xy = np.vstack([datax, datay])
    z = gaussian_kde(xy)(xy)
    pos_max_density = np.argmax(z)
    # sns.set_style("whitegrid")
    im = plt.scatter(datax, datay, c=z / (max(z)), s=5, cmap=cmap_d, vmin=0, vmax=1)

    min_x = min(datax)
    max_x = max(datax)
    min_y = min(datay)
    max_y = max(datay)

    x = np.linspace(min_lim, max_lim, 10)
    ax.set(xlim=[min_lim, max_lim], ylim=[min_lim, max_lim])
    axes = plt.gca()
    axes.set_aspect('equal', adjustable='box')
    plt.plot(x, x, 'r')
    # plt.colorbar(label = 'Normalized Density')
    return im

# There is 4 mooring locating in the fram strait
# F11 12 13 14


list_F11 = glob.glob('NPI/COMBINE_*F11*.csv')
list_F12 = glob.glob('NPI/COMBINE_*F12*.csv')
list_F13 = glob.glob('NPI/COMBINE_*F13*.csv')
list_F14 = glob.glob('NPI/COMBINE_*F14*.csv')

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

cmap = plt.cm.get_cmap('inferno')
cmap_cut = cmap(np.arange(cmap.N))[:-40]
cmap = LinearSegmentedColormap.from_list('cut', cmap_cut, cmap.N)

fig = plt.figure('F11 scatter', figsize=(12,7))
sns.set_style('whitegrid')
ax1 = fig.add_subplot(2,4,6)
sns.set_style('whitegrid')
plt.scatter(df_F14_ers1['NPO'], df_F14_ers1['sat'], c = 1-df_F14_ers1['type'], cmap=cmap, vmin=0, vmax=1, label='ERS-1', marker='v')
plt.scatter(df_F14_ers2['NPO'], df_F14_ers2['sat'], c = 1-df_F14_ers2['type'], cmap=cmap, vmin=0, vmax=1, label='ERS-2')
plt.scatter(df_F14_env3['NPO'], df_F14_env3['sat'], c = 1-df_F14_env3['type'], cmap=cmap, vmin=0, vmax=1, marker='^', label='Envisat')
plt.scatter(df_F14_cs2['NPO'], df_F14_cs2['sat'], c = 1-df_F14_cs2['type'], cmap=cmap, vmin=0, vmax=1, marker='*', label='CryoSat-2')
x = np.linspace(-0.2, 4, 10)
ax1.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
ax1.set_title('NPI F14')
ax1.set_title('(e)', loc='left')
stat(df_F14['NPO'], df_F14['sat'], ax1, px=0.05, py=0.85)
axes = plt.gca()

handlesb, labelsb = plt.gca().get_legend_handles_labels()
by_labelb = dict(zip(labelsb, handlesb))
axes.set_aspect('equal', adjustable='box')
plt.plot(x, x, 'r')

ax2 = fig.add_subplot(2,4,5)
sns.set_style('whitegrid')
plt.scatter(df_F13_ers2['NPO'], df_F13_ers2['sat'], c = 1-df_F13_ers2['type'], cmap=cmap, vmin=0, vmax=1, label='ERS-2')
plt.scatter(df_F13_env3['NPO'], df_F13_env3['sat'], c = 1-df_F13_env3['type'], cmap=cmap, vmin=0, vmax=1, marker='^', label='Envisat')
plt.scatter(df_F13_cs2['NPO'], df_F13_cs2['sat'], c = 1-df_F13_cs2['type'], cmap=cmap, vmin=0, vmax=1, marker='*', label='CryoSat-2')
x = np.linspace(-0.2, 4, 10)
ax2.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
axes = plt.gca()
axes.set_aspect('equal', adjustable='box')
plt.plot(x, x, 'r')

ax2.set_title('NPI F13')
ax2.set_title('(d)', loc='left')
ax2.set_ylabel('Sea Ice Draft from satellites (m)')

stat(df_F13['NPO'], df_F13['sat'], ax2, px=0.05, py=0.85)

ax3 = fig.add_subplot(2,4,2)
sns.set_style('whitegrid')
plt.scatter(df_F12_ers2['NPO'], df_F12_ers2['sat'], c = 1-df_F12_ers2['type'], cmap=cmap, vmin=0, vmax=1, label='ERS-2')
plt.scatter(df_F12_env3['NPO'], df_F12_env3['sat'], c = 1-df_F12_env3['type'], cmap=cmap, vmin=0, vmax=1, marker='^', label='Envisat')
plt.scatter(df_F12_cs2['NPO'], df_F12_cs2['sat'], c = 1-df_F12_cs2['type'], cmap=cmap, vmin=0, vmax=1, marker='*', label='CryoSat-2')
x = np.linspace(-0.2, 4, 10)
ax3.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
ax2.set_xlabel('NPI Sea Ice Draft (m)')
axes = plt.gca()
axes.set_aspect('equal', adjustable='box')
ax3.set_title('NPI F12')
ax3.set_title('(b)', loc='left')
plt.plot(x, x, 'r')
stat(df_F12['NPO'], df_F12['sat'], ax3, px=0.05, py=0.85)


ax4 = fig.add_subplot(2,4,1)
sns.set_style('whitegrid')
plt.scatter(df_F11_ers2['NPO'], df_F11_ers2['sat'], c = 1-df_F11_ers2['type'], cmap=cmap, vmin=0, vmax=1, label='ERS-2')
plt.scatter(df_F11_env3['NPO'], df_F11_env3['sat'], c = 1-df_F11_env3['type'], cmap=cmap, vmin=0, vmax=1, marker='^', label='Envisat')
im = plt.scatter(df_F11_cs2['NPO'], df_F11_cs2['sat'], c = 1-df_F11_cs2['type'], cmap=cmap, vmin=0, vmax=1, marker='*', label='CryoSat-2')
x = np.linspace(-0.2, 4, 10)
ax4.set(xlim = [-0.2, 4], ylim = [-0.2, 4])
ax1.set_xlabel('NPI Sea Ice Draft (m)')
ax4.set_title('NPI F11')
ax4.set_title('(a)', loc='left')
ax4.set_ylabel('Sea Ice Draft from satellites (m)')
axes = plt.gca()
axes.set_aspect('equal', adjustable='box')
plt.plot(x, x, 'r')
stat(df_F11['NPO'], df_F11['sat'], ax4, px=0.05, py=0.85)

fig.legend(by_labelb.values(), by_labelb.keys(), loc='upper left', ncol=6, frameon=False)

cbar_ax = fig.add_axes([0.52, 0.0785, 0.03, 0.844])
fig.colorbar(im, cax=cbar_ax, label = 'FYI fraction', orientation='vertical')
plt.subplots_adjust(left=0.05, bottom=0.01, right=0.95, top=0.99, wspace=0.13, hspace=0.001)

df_env3 = pd.read_csv('submarines/COMBINE_submarines_ARC_env3_corr_interp.csv', usecols=[1,2,3,4,5])
df_ers2 = pd.read_csv('submarines/COMBINE_submarines_ARC_ers2r_corr_interp.csv', usecols=[1,2,3,4,5])

sns.set_style('whitegrid')
ax5 = fig.add_subplot(2,4,4)
cmap_d = plt.cm.get_cmap(cm.vik)
scatter_gaussian_gathered(df_ers2.iloc[:, 0], df_ers2.iloc[:, 1], ax5, -0.2, 6)
stat(df_ers2.iloc[:, 0], df_ers2.iloc[:, 1], ax5)
ax5.set_xticks([0,1,2,3,4,5,6])
ax5.set_yticks([0,1,2,3,4,5,6])
ax5.set_ylabel('ERS-2 Sea Ice Draft  (m)')
ax5.set_title('Submarines vs ERS-2')
ax5.set_title('(c)', loc='left')

ax6 = fig.add_subplot(2,4,8)
cmap_d = plt.cm.get_cmap(cm.vik)
im = scatter_gaussian_gathered(df_env3.iloc[:, 0], df_env3.iloc[:, 1], ax6, -0.2, 6)
stat(df_env3.iloc[:, 0], df_env3.iloc[:, 1], ax6)
ax6.set_xticks([0,1,2,3,4,5,6])
ax6.set_yticks([0,1,2,3,4,5,6])
ax6.set_title('Submarines vs Envisat')
ax6.set_title('(f)', loc='left')

ax6.set_ylabel('Envisat Sea Ice Draft (m)')
ax6.set_xlabel('US/UK Submarines Sea ice draft (m)')

cbar_ax2 = fig.add_axes([0.64, 0.0785, 0.03, 0.844])
fig.colorbar(im, cax=cbar_ax2, label = 'Normalized Density', orientation='vertical')#, location='left')

plt.savefig('SCICEX_NPI_scatter.pdf')


plt.show()

