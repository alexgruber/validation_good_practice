


import numpy as np
import pandas as pd

import platform
if platform.system() in ['Linux', 'Darwin']:
    import matplotlib
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from pathlib import Path

from itertools import combinations

from validation_good_practice.ancillary.grid import EASE2
from validation_good_practice.ancillary.paths import Paths


def boxplot_tca():
    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2', 'ISMN']
    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    metric = ['bias', 'r2', 'ubrmse']
    ylim = [[0, 1.5], [0, 1], [0, 0.1]]

    sensors = ['ASCAT', 'SMOS', 'SMAP']

    params = ['l_', 'm_', 'u_']
    titles = ['CI$_{2.5}$', 'CI$_{50}$', 'CI$_{97.5}$']

    figsize = (10, 8)
    fontsize = 14

    pos = [i + j for i in np.arange(1,4) for j in [-0.2,0,0.2]]
    colors = [s for n in np.arange(3) for s in ['lightblue', 'lightgreen', 'coral'] ]

    f = plt.figure(figsize=figsize)

    n = 0
    for m, yl in zip(metric, ylim):
        n += 1

        ax = plt.subplot(3, 1, n)
        plt.grid(color='k', linestyle='--', linewidth=0.25)

        data = list()

        for s in sensors:
            for p, t in zip(params, titles):

                if s == 'ASCAT':
                    tag = m + '_grid_abs_' + p + s
                    res[tag] = (res[tag + '_tc_ASCAT_SMOS_MERRA2'] + res[tag + '_tc_ASCAT_SMAP_MERRA2']) / 2
                else:
                    tag = m + '_grid_abs_' + p + s + '_tc_ASCAT_' + s + '_MERRA2'

                data.append(res[tag].values)

        box = ax.boxplot(data, whis=[10, 90], showfliers=False, positions=pos, widths=0.10, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set(color='black', linewidth=2)
            patch.set_facecolor(color)
        for patch in box['medians']:
            patch.set(color='black', linewidth=2)
        for patch in box['whiskers']:
            patch.set(color='black', linewidth=1)
        if n == 3:
            plt.figlegend((box['boxes'][0:3]), titles, 'upper left', bbox_to_anchor=(0.087,0.82), fontsize=fontsize - 2)

        plt.xlim(0.5, 3.5)
        # if n < 3:
        #     plt.xticks(np.arange(1,4),'')
        # else:
        plt.xticks(np.arange(1,4),sensors, fontsize=fontsize)

        plt.ylim(yl)
        plt.yticks(fontsize=fontsize)

        for i in np.arange(1,3):
            plt.axvline(i + 0.5, linewidth=1, color='k')

        plt.ylabel(m, fontsize=fontsize)


    f.subplots_adjust(hspace=0.2)

    # plt.show()

    fout = path / 'plots' / ('boxplot_tca.png')
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


def boxplot_relative_metrics():

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2', 'ISMN']
    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    metric = ['bias', 'r', 'ubrmsd']
    ylim = [[-0.5, 0.5], [0, 1], [0.02, 0.25]]

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2']
    tuples = ['_'.join(t) for t in combinations(sensors, 2)]
    names = [' - '.join(t) for t in combinations(sensors, 2)]

    params = ['l_', 'p_', 'u_']
    titles = ['CI$_{2.5}$', 'p.est.', 'CI$_{97.5}$']

    figsize = (15, 8)
    fontsize = 14

    pos = [i + j for i in np.arange(1,7) for j in [-0.2,0,0.2]]
    colors = [s for n in np.arange(6) for s in ['lightblue', 'lightgreen', 'coral'] ]

    f = plt.figure(figsize=figsize)

    n = 0
    for m, yl in zip(metric, ylim):
        n += 1


        ax = plt.subplot(3, 1, n)
        plt.grid(color='k', linestyle='--', linewidth=0.25)

        data = list()

        for t in tuples:
            for p in params:

                tag = m + '_grid_abs_' + p + t

                if m == 'r':
                    res[tag] *= res[tag]

                data.append(res[tag].values)

        box = ax.boxplot(data, whis=[10, 90], showfliers=False, positions=pos, widths=0.10, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set(color='black', linewidth=2)
            patch.set_facecolor(color)
        for patch in box['medians']:
            patch.set(color='black', linewidth=2)
        for patch in box['whiskers']:
            patch.set(color='black', linewidth=1)
        if n == 3:
            plt.figlegend((box['boxes'][0:3]), titles, bbox_to_anchor=(0.787,0.28), fontsize=fontsize - 2)

        plt.xlim(0.5, 6.5)
        plt.xticks(np.arange(1,7), names, fontsize=fontsize)

        plt.ylim(yl)
        plt.yticks(fontsize=fontsize)

        for i in np.arange(1,6):
            plt.axvline(i + 0.5, linewidth=1, color='k')

        plt.ylabel(m, fontsize=fontsize)


    f.subplots_adjust(hspace=0.2)

    # plt.show()

    fout = path / 'plots' / ('boxplot_relative_metrics.png')
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()



def plot_ease_img(data, tag,
                  llcrnrlat=24,
                  urcrnrlat=51,
                  llcrnrlon=-128,
                  urcrnrlon=-64,
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

    plt.title(title,fontsize=fontsize)

    return im


def spatial_plot_tca():

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2', 'ISMN']
    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    sensors = ['ASCAT','SMOS','SMAP']

    params = ['l_','m_','u_']
    titles = ['CI$_{2.5}$', 'CI$_{50}$', 'CI$_{97.5}$']

    figsize = (16,8)
    fontsize = 14

    metric = ['bias','r2', 'ubrmse']
    cbrange = [[0,1], [0,1], [0,0.06]]

    for m, cb in zip(metric,cbrange):

        f = plt.figure(figsize=figsize)

        n = 0
        for s in sensors:
            for p, t in zip(params,titles):
                n += 1
                plt.subplot(3,3,n)

                if s == 'ASCAT':
                    # tag = m + '_grid_abs_' + p + s + '_tc_ASCAT_SMAP_MERRA2'

                    tag = m + '_grid_abs_' + p + s
                    res[tag] = (res[tag + '_tc_ASCAT_SMOS_MERRA2'] + res[tag + '_tc_ASCAT_SMAP_MERRA2']) / 2
                else:
                    tag = m + '_grid_abs_' + p + s + '_tc_ASCAT_' + s + '_MERRA2'

                im = plot_ease_img(res, tag, fontsize=fontsize, cbrange=cb)

                if n < 4:
                    plt.title(t, fontsize=fontsize)

        f.subplots_adjust(wspace=0.05, hspace=0, bottom=0.1)

        xoffs = 0.11

        f.text(xoffs, 0.77, 'ASCAT', fontsize=fontsize, rotation=90)
        f.text(xoffs, 0.51, 'SMOS', fontsize=fontsize, rotation=90)
        f.text(xoffs, 0.24, 'SMAP', fontsize=fontsize, rotation=90)

        cbar_ax = f.add_axes([0.31, 0.07, 0.4, 0.027])
        cb = f.colorbar(im, orientation='horizontal', cax=cbar_ax)
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)

        # plt.show()

        fout = path / 'plots' / ('spatial_plot_tc_' + m + '.png')
        f.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()

def spatial_plot_relative_metrics():

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2', 'ISMN']
    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    sensors = ['ASCAT','SMOS','SMAP', 'MERRA2']
    tuples = ['_'.join(t) for t in combinations(sensors, 2)]
    names = [' - '.join(t) for t in combinations(sensors, 2)]

    yoffs = [0.16, 0.305, 0.43, 0.585, 0.71, 0.85][::-1]

    params = ['l_','p_','u_']

    figsize = (12.9,12)
    fontsize = 14

    metric = ['bias','r', 'ubrmsd']
    cbrange = [[0,0.4], [0,1], [0,0.1]]

    for m, cb in zip(metric, cbrange):

        titles = ['CI$_{2.5}$', m, 'CI$_{97.5}$']

        f = plt.figure(figsize=figsize)

        i = 0
        for t in tuples:
            for p, tit in zip(params, titles):
                i += 1
                plt.subplot(6,3,i)

                tag = m + '_grid_abs_' + p + t

                if m == 'r':
                    res[tag] *= res[tag]

                im = plot_ease_img(res, tag, fontsize=fontsize, cbrange=cb)

                if i < 4:
                    plt.title(tit, fontsize=fontsize)

        f.subplots_adjust(wspace=0.01, hspace=0, bottom=0.05)

        xoff = 0.11
        for yoff, name in zip(yoffs, names):
            f.text(xoff, yoff, name, fontsize=fontsize-2, rotation=90)

        cbar_ax = f.add_axes([0.3, 0.028, 0.4, 0.02])
        cb = f.colorbar(im, orientation='horizontal', cax=cbar_ax)
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)

        # plt.show()

        fout = path / 'plots' / ('spatial_plot_relative_metrics_' + m + '.png')
        f.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()


def spatial_plot_n():

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2', 'ISMN']
    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2']
    tuples = ['_'.join(t) for t in combinations(sensors, 2)]
    names = [' - '.join(t) for t in combinations(sensors, 2)]

    figsize = (17, 9)
    fontsize = 14

    cbrange = [0,400]

    f = plt.figure(figsize=figsize)

    i = 0
    for t, n in zip (tuples,names):
        i += 1

        plt.subplot(3, 3, i)

        tag = 'n_corr_grid_abs_' + t

        plot_ease_img(res, tag, fontsize=fontsize, cbrange=cbrange)

        if i < 8:
            plt.title(n, fontsize=fontsize)

    plt.subplot(3, 3, 8)
    im = plot_ease_img(res, 'n_grid', fontsize=fontsize, cbrange=cbrange)
    plt.title('Uncorrected', fontsize=fontsize)

    f.subplots_adjust(wspace=0.04, hspace=0.03, bottom=0.07)

    cbar_ax = f.add_axes([0.3, 0.03, 0.4, 0.025])
    cb = f.colorbar(im, orientation='horizontal', cax=cbar_ax)
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)

    # plt.show()

    fout = path / 'plots' / ('spatial_plot_sample_size.png')
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


if __name__=='__main__':

    # boxplot_tca()
    boxplot_relative_metrics()

    # spatial_plot_tca()
    # spatial_plot_relative_metrics()

    # spatial_plot_n()


