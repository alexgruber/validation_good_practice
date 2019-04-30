

import numpy as np
import pandas as pd

import platform
if platform.system() in ['Linux', 'Darwin']:
    import matplotlib
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from validation_good_practice.ancillary.grid import EASE2
from validation_good_practice.ancillary.paths import Paths


def plot_ease_img(data,tag,
                  llcrnrlat=24,
                  urcrnrlat=51,
                  llcrnrlon=-128,
                  urcrnrlon=-64,
                  figsize=(20,10),
                  cbrange=None,
                  cmap='jet',
                  title='',
                  fontsize=20):


    grid = EASE2()

    lons,lats = np.meshgrid(grid.ease_lons, grid.ease_lats)

    img = np.empty(lons.shape, dtype='float32')
    img.fill(None)

    ind_lat = data.loc[:,'row'].values.astype('int')
    ind_lon = data.loc[:,'col'].values.astype('int')

    img[ind_lat,ind_lon] = data.loc[:,tag]
    img_masked = np.ma.masked_invalid(img)

    plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')

    m = Basemap(projection='mill',
                llcrnrlat=llcrnrlat,
                urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,
                urcrnrlon=urcrnrlon,
                resolution='c')

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)

    if cbrange is not None:
        im.set_clim(vmin=cbrange[0], vmax=cbrange[1])

    cb = m.colorbar(im, "bottom", size="7%", pad="8%")

    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)

    plt.title(title,fontsize=fontsize)

    plt.tight_layout()
    plt.show()


def plot_test(res):

    cbrange = [0,1]

    plot_ease_img(res, 'r2_ismn_abs_m_ISMN_tc_ASCAT_SMAP_ISMN',
                  cbrange=cbrange)


if __name__=='__main__':

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2', 'ISMN']
    paths = Paths()

    res = pd.read_csv(paths.result_root / ('_'.join(sensors)) / 'result.csv', index_col=0)

    plot_test(res)



