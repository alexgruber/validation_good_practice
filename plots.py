
import warnings
warnings.filterwarnings("ignore")

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

from validation_good_practice.ancillary.metrics import bias, ubRMSD, Pearson_R, TCA

from validation_good_practice.data_readers.interface import reader

def plot_ts():

    lon = -101.392047
    lat = 36.022264

    sensors = ['ASCAT','SMOS','MERRA2', 'ISMN']

    io = reader(sensors)

    grid = pd.read_csv('/data_sets/EASE2_grid/grid_lut.csv', index_col=0)

    gpi = grid.index[np.argmin((grid.ease2_lon.values-lon)**2 + (grid.ease2_lat.values-lat)**2)]

    data = io.read(gpi)[1][0][['SMOS','MERRA2']]

    print(data.mean())
    data.plot()

    plt.show()


def boxplot_tca_ismn(sensors):
    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    metric = ['bias', 'r2', 'ubrmse']
    ylim = [[0, 1.5], [0, 1], [0, 0.1]]

    sensors = sensors[:-2]

    params = ['l_', 'm_', 'u_']
    titles = ['CI$_{2.5}$', 'CI$_{50}$', 'CI$_{97.5}$']

    figsize = (10, 8)
    fontsize = 14

    pos = [i + j for i in np.arange(1,len(sensors)+1) for j in [-0.2,0,0.2]]
    colors = [s for n in np.arange(len(sensors)) for s in ['lightblue', 'lightgreen', 'coral'] ]

    f = plt.figure(figsize=figsize)

    n = 0
    for m, yl in zip(metric, ylim):
        n += 1

        ax = plt.subplot(3, 1, n)

        if n == 1:
            axpos = ax.get_position()

        plt.grid(color='k', linestyle='--', linewidth=0.25)

        data = list()

        for s in sensors:
            for p, t in zip(params, titles):

                if s == 'ASCAT':
                    if 'SMAP' in sensors:
                        tag = m + '_ismn_abs_' + p + s
                        res[tag] = (res[tag + '_tc_ASCAT_SMOS_ISMN'] + res[tag + '_tc_ASCAT_SMAP_ISMN']) / 2
                    else:
                        tag = m + '_ismn_abs_' + p + s + '_tc_ASCAT_SMOS_ISMN'
                else:
                    tag = m + '_ismn_abs_' + p + s + '_tc_ASCAT_' + s + '_ISMN'

                tmp = res[tag].values
                data.append(tmp[~np.isnan(tmp)])

        box = ax.boxplot(data, whis=[10, 90], showfliers=False, positions=pos, widths=0.10, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set(color='black', linewidth=2)
            patch.set_facecolor(color)
        for patch in box['medians']:
            patch.set(color='black', linewidth=2)
        for patch in box['whiskers']:
            patch.set(color='black', linewidth=1)

        plt.xlim(0.5, 3.5)
        plt.xticks(np.arange(1,4),sensors, fontsize=fontsize)

        plt.ylim(yl)
        plt.yticks(fontsize=fontsize)

        for i in np.arange(1,3):
            plt.axvline(i + 0.5, linewidth=1, color='k')

        plt.ylabel(m, fontsize=fontsize)

    f.subplots_adjust(hspace=0.2)

    plt.figlegend((box['boxes'][0:3]), titles, 'upper left', bbox_to_anchor=(axpos.x0-0.04,axpos.y1-0.055), fontsize=fontsize - 2)

    # plt.show()

    fout = path / 'plots' / ('boxplot_tca_ismn.png')
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


def boxplot_tca(sensors, mode='abs'):

    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    metric = ['bias', 'r2', 'ubrmse']
    ylim = [[0, 1.5], [0, 1], [0, 0.1]]

    sensors = sensors[:-2]

    params = ['l_', 'm_', 'u_']
    titles = ['CI$_{2.5}$', 'CI$_{50}$', 'CI$_{97.5}$']

    figsize = (10, 8)
    fontsize = 14

    pos = [i + j for i in np.arange(1,len(sensors)+1) for j in [-0.2,0,0.2]]
    colors = [s for n in np.arange(len(sensors)) for s in ['lightblue', 'lightgreen', 'coral'] ]

    f = plt.figure(figsize=figsize)

    n = 0
    for m, yl in zip(metric, ylim):
        n += 1

        ax = plt.subplot(3, 1, n)

        if n == 1:
            axpos = ax.get_position()

        plt.grid(color='k', linestyle='--', linewidth=0.25)

        data = list()

        for s in sensors:
            for p, t in zip(params, titles):

                if s == 'ASCAT':

                    if 'SMAP' in sensors:
                        tag = m + '_grid_'+mode+'_' + p + s
                        res[tag] = (res[tag + '_tc_ASCAT_SMOS_MERRA2'] + res[tag + '_tc_ASCAT_SMAP_MERRA2']) / 2
                    else:
                        tag = m + '_grid_'+mode+'_' + p + s + '_tc_ASCAT_SMOS_MERRA2'
                else:
                    tag = m + '_grid_'+mode+'_' + p + s + '_tc_ASCAT_' + s + '_MERRA2'

                data.append(res[tag].values)

        box = ax.boxplot(data, whis=[10, 90], showfliers=False, positions=pos, widths=0.10, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set(color='black', linewidth=2)
            patch.set_facecolor(color)
        for patch in box['medians']:
            patch.set(color='black', linewidth=2)
        for patch in box['whiskers']:
            patch.set(color='black', linewidth=1)

        plt.xlim(0.5, 3.5)

        plt.xticks(np.arange(1,4),sensors, fontsize=fontsize)

        plt.ylim(yl)
        plt.yticks(fontsize=fontsize)

        for i in np.arange(1,3):
            plt.axvline(i + 0.5, linewidth=1, color='k')

        plt.ylabel(m, fontsize=fontsize)

    f.subplots_adjust(hspace=0.2)

    plt.figlegend((box['boxes'][0:3]), titles, 'upper left', bbox_to_anchor=(axpos.x0-0.04,axpos.y1-0.055), fontsize=fontsize - 2)

    plt.show()

    # fout = path / 'plots' / (mode + '_boxplot_tca.png')
    # f.savefig(fout, dpi=300, bbox_inches='tight')
    # plt.close()


def boxplot_relative_metrics_ismn(sensors):

    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    metric = ['bias', 'r', 'ubrmsd']
    ylim = [[-0.12,0.12], [0, 1], [0.02, 0.25]]

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2', 'ISMN']
    tuples = ['_'.join(t) for t in combinations(sensors, 2)]
    names = [' - '.join(t) for t in combinations(sensors, 2)]

    params = ['l_', 'p_', 'u_']
    titles = ['CI$_{2.5}$', 'est.', 'CI$_{97.5}$']

    figsize = (24, 8)
    fontsize = 14

    pos = [i + j for i in np.arange(1,len(tuples)+1) for j in [-0.2,0,0.2]]
    colors = [s for n in np.arange(len(tuples)) for s in ['lightblue', 'lightgreen', 'coral'] ]

    f = plt.figure(figsize=figsize)

    n = 0
    for m, yl in zip(metric, ylim):
        n += 1


        ax = plt.subplot(3, 1, n)

        if n == 1:
            axpos = ax.get_position()

        plt.grid(color='k', linestyle='--', linewidth=0.25)

        data = list()

        for t in tuples:
            for p in params:

                if (m == 'bias') & ((t[0:5] == 'ASCAT')|(t[-4::] == 'ISMN')):
                    data.append([])
                    continue

                tag = m + '_ismn_abs_' + p + t

                if tag not in res:
                    data.append([])
                    continue

                if m == 'r':
                    res[tag] *= res[tag]

                tmp = res[tag].values
                data.append(tmp[~np.isnan(tmp)])


        box = ax.boxplot(data, whis=[10, 90], showfliers=False, positions=pos, widths=0.10, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set(color='black', linewidth=2)
            patch.set_facecolor(color)
        for patch in box['medians']:
            patch.set(color='black', linewidth=2)
        for patch in box['whiskers']:
            patch.set(color='black', linewidth=1)

        plt.xlim(0.5, len(tuples)+0.5)
        plt.xticks(np.arange(1,len(tuples)+1), names, fontsize=fontsize)

        plt.ylim(yl)
        plt.yticks(fontsize=fontsize)

        for i in np.arange(1,len(tuples)):
            plt.axvline(i + 0.5, linewidth=1, color='k')

        plt.ylabel(m, fontsize=fontsize)


    f.subplots_adjust(hspace=0.2)

    plt.figlegend((box['boxes'][0:3]), titles, 'upper left', bbox_to_anchor=(axpos.x0 - 0.06, axpos.y1 - 0.055),
                  fontsize=fontsize - 2)

    # plt.show()

    fout = path / 'plots' / ('boxplot_relative_metrics_ismn.png')
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


def boxplot_relative_metrics(sensors):

    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    metric = ['bias', 'r', 'ubrmsd']
    ylim = [[-0.12,0.12], [0, 1], [0.02, 0.25]]

    sensors = ['ASCAT', 'SMOS', 'MERRA2']
    tuples = ['_'.join(t) for t in combinations(sensors, 2)]
    names = [' - '.join(t) for t in combinations(sensors, 2)]

    params = ['l_', 'p_', 'u_']
    titles = ['CI$_{2.5}$', 'p.est.', 'CI$_{97.5}$']

    figsize = (15, 8)
    fontsize = 14


    pos = [i + j for i in np.arange(1,len(tuples)+1) for j in [-0.2,0,0.2]]
    colors = [s for n in np.arange(len(tuples)) for s in ['lightblue', 'lightgreen', 'coral'] ]

    f = plt.figure(figsize=figsize)

    n = 0
    for m, yl in zip(metric, ylim):
        n += 1


        ax = plt.subplot(3, 1, n)

        if n == 1:
            axpos = ax.get_position()

        plt.grid(color='k', linestyle='--', linewidth=0.25)

        data = list()

        for t in tuples:
            for p in params:

                if (m == 'bias') & (t[0:5] == 'ASCAT'):
                    data.append([])
                    continue

                tag = m + '_grid_abs_' + p + t

                if tag not in res:
                    data.append([])
                    continue

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

        plt.xlim(0.5, 6.5)
        plt.xticks(np.arange(1,4), names, fontsize=fontsize)

        plt.ylim(yl)
        plt.yticks(fontsize=fontsize)

        for i in np.arange(1,6):
            plt.axvline(i + 0.5, linewidth=1, color='k')

        plt.ylabel(m, fontsize=fontsize)


    f.subplots_adjust(hspace=0.2)

    plt.figlegend((box['boxes'][0:3]), titles, 'upper left', bbox_to_anchor=(axpos.x0 - 0.06, axpos.y1 - 0.055),
                  fontsize=fontsize - 2)

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
                  cmap='jet_r',
                  title='',
                  fontsize=20,
                  plot_cb=False,
                  print_median=False):

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

    if plot_cb is True:

        ticks = np.arange(cbrange[0],cbrange[1]+0.001, (cbrange[1]-cbrange[0])/4)
        cb = m.colorbar(im, "bottom", size="10%", pad="4%", ticks=ticks)
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)

    plt.title(title,fontsize=fontsize)

    if print_median is True:
        x, y = m(-79, 25)
        plt.text(x, y, 'med. = %.2f' % np.ma.median(img_masked), fontsize=fontsize-2)

    return im

def spatial_plot_tca_diff(mode='abs'):

    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    figsize = (8,8)
    fontsize = 14

    metrics = ['r2', 'ubrmse']
    cbrange = [[-0.2,0.2], [-0.02,0.02]]

    f = plt.figure(figsize=figsize)

    n = 0
    for m, cb in zip(metrics,cbrange):
        n += 1

        plt.subplot(2,1,n)

        tag1 = m + '_grid_'+mode+'_m_ASCAT_tc_ASCAT_SMOS_MERRA2'
        tag2 = m + '_grid_'+mode+'_m_ASCAT_tc_ASCAT_SMAP_MERRA2'

        tag = 'diff'

        res[tag] = (res[tag1] - res[tag2])

        im = plot_ease_img(res, tag, fontsize=fontsize, cbrange=cb, cmap=('jet' if m == 'r2' else 'jet_r'), plot_cb=True, print_median=True)

        plt.ylabel(m,fontsize=fontsize)

        if n ==1:
            plt.title('TCA difference', fontsize=fontsize)

    f.subplots_adjust(hspace=0.15, bottom=0.1)

    # plt.show()

    fout = path / 'plots' / (mode+'_spatial_plot_tca_diff.png')
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

def spatial_plot_tca(sensors,mode='abs'):

    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    sensors = sensors[:-2]

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

                    if 'SMAP' in sensors:
                        tag = m + '_grid_'+mode+'_' + p + s
                        res[tag] = (res[tag + '_tc_ASCAT_SMOS_MERRA2'] + res[tag + '_tc_ASCAT_SMAP_MERRA2']) / 2
                    else:
                        tag = m + '_grid_'+mode+'_' + p + s + '_tc_ASCAT_SMOS_MERRA2'

                else:
                    tag = m + '_grid_'+mode+'_' + p + s + '_tc_ASCAT_' + s + '_MERRA2'

                im = plot_ease_img(res, tag, fontsize=fontsize, cbrange=cb, cmap=('jet_r' if m=='r2' else 'jet'))

                if n < 4:
                    plt.title(t, fontsize=fontsize)
                if p == 'l_':
                    plt.ylabel(s,fontsize=fontsize)

        f.subplots_adjust(wspace=0.05, hspace=0, bottom=0.1)

        cbar_ax = f.add_axes([0.31, 0.07, 0.4, 0.027])
        cb = f.colorbar(im, orientation='horizontal', cax=cbar_ax)
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)

        plt.show()

        # fout = path / 'plots' / ('spatial_plot_tc_' + m + '.png')
        # f.savefig(fout, dpi=300, bbox_inches='tight')
        # plt.close()

def spatial_plot_tca_ci_diff(sensors, mode='abs'):

    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    sensors = sensors[:-2]

    params = ['m_', 'diff_']
    titles = ['CI$_{50}$', 'CI$_{97.5}$ - CI$_{2.5}$']

    figsize = (10,8)
    fontsize = 14

    metric = ['bias','r2', 'ubrmse'][1:3]
    if mode == 'abs':
        cbrange = [[0,1], [0,1], [0,0.06]][1:3]
        cbrange_diff = [[0,1], [0,1], [0,0.06]][1:3]
    else:
        cbrange = [[0,1], [0,1], [0,0.04]][1:3]
        cbrange_diff = [[0,1], [0,0.8], [0,0.04]][1:3]

    for m, cb, cb_diff in zip(metric,cbrange,cbrange_diff):

        f = plt.figure(figsize=figsize)

        n = 0
        for s in sensors:
            for p, t in zip(params,titles):
                n += 1
                plt.subplot(3,2,n)

                if p == 'm_':
                    if s == 'ASCAT':
                        if 'SMAP' in sensors:
                            tag = m + '_grid_'+mode+'_' + p + s
                            res[tag] = (res[tag + '_tc_ASCAT_SMOS_MERRA2'] + res[tag + '_tc_ASCAT_SMAP_MERRA2']) / 2
                        else:
                            tag = m + '_grid_'+mode+'_' + p + s + '_tc_ASCAT_SMOS_MERRA2'
                    else:
                        tag = m + '_grid_'+mode+'_' + p + s + '_tc_ASCAT_' + s + '_MERRA2'
                else:
                    if s == 'ASCAT':
                        if 'SMAP' in sensors:
                            tag_u = m + '_grid_'+mode+'_' + 'u_' + s
                            tag_l = m + '_grid_'+mode+'_' + 'l_' + s
                            res[tag_u] = (res[tag_u + '_tc_ASCAT_SMOS_MERRA2'] + res[tag_u + '_tc_ASCAT_SMAP_MERRA2']) / 2
                            res[tag_l] = (res[tag_l + '_tc_ASCAT_SMOS_MERRA2'] + res[tag_l + '_tc_ASCAT_SMAP_MERRA2']) / 2
                        else:
                            tag_u = m + '_grid_'+mode+'_' + 'u_' + s + '_tc_ASCAT_SMOS_MERRA2'
                            tag_l = m + '_grid_'+mode+'_' + 'l_' + s + '_tc_ASCAT_SMOS_MERRA2'
                    else:
                        tag_u = m + '_grid_'+mode+'_' + 'u_' + s + '_tc_ASCAT_' + s + '_MERRA2'
                        tag_l = m + '_grid_'+mode+'_' + 'l_' + s + '_tc_ASCAT_' + s + '_MERRA2'

                    res[tag] = res[tag_u] - res[tag_l]

                if p == 'm_':
                    im_r = plot_ease_img(res, tag, fontsize=fontsize, cbrange=cb, cmap=('jet_r' if m=='r2' else 'jet'))
                else:
                    im = plot_ease_img(res, tag, fontsize=fontsize, cbrange=cb_diff, cmap='jet')

                if n < 3:
                    plt.title(t, fontsize=fontsize)
                if p == 'm_':
                    plt.ylabel(s,fontsize=fontsize)

        f.subplots_adjust(wspace=0.05, hspace=0, bottom=0.1)

        pos1 = im_r.axes.get_position()
        pos2 = im.axes.get_position()

        ticks = np.arange(cb[0], cb[1], round((cb[1] - cb[0]) / 5 * 1000) / 1000)
        cbar_ax = f.add_axes([pos1.x0, 0.07, pos1.x1 - pos1.x0, 0.027])
        cb = f.colorbar(im_r, orientation='horizontal', cax=cbar_ax, ticks=ticks)
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)
        # cb.ax.set_xticklabels()

        ticks = np.arange(cb_diff[0], cb_diff[1], round((cb_diff[1] - cb_diff[0]) / 5 * 1000) / 1000)
        cbar_ax = f.add_axes([pos2.x0, 0.07, pos2.x1 - pos2.x0, 0.027])
        cb = f.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=ticks)
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)

        # plt.show()

        fout = path / 'plots' / (mode+'_spatial_plot_tc_' + m + '_ci_diff.png')
        f.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()

def spatial_plot_relative_metrics(sensors,mode='abs'):

    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2']


    tuples = ['_'.join(t) for t in combinations(sensors, 2)]
    names = [' - '.join(t) for t in combinations(sensors, 2)]

    params = ['l_','p_','u_']

    figsize = (12.9,12)
    fontsize = 14

    metric = ['bias','r', 'ubrmsd']

    cbrange = [[-0.12,0.12], [0,0.7], [0,0.1]]

    for m, cb in zip(metric, cbrange):

        titles = ['CI$_{2.5}$', 'R$^2$ ' if m=='r' else m, 'CI$_{97.5}$']

        f = plt.figure(figsize=figsize)

        i = 0
        for t, n in zip(tuples, names):
            for p, tit in zip(params, titles):
                i += 1

                plt.subplot(6,3,i)

                if i < 4:
                    plt.title(tit, fontsize=fontsize)

                if (m == 'bias') & (n[0:5] == 'ASCAT'):
                    plt.gca().set_frame_on(False)
                    plt.gca().set_xticks([])
                    plt.gca().set_yticks([])
                    continue

                tag = m + '_grid_'+mode+'_' + p + t

                if tag not in res:
                    plt.gca().set_frame_on(False)
                    plt.gca().set_xticks([])
                    plt.gca().set_yticks([])
                    continue


                if m == 'r':
                    res[tag] *= res[tag]

                im = plot_ease_img(res, tag, fontsize=fontsize, cbrange=cb, cmap=('jet_r' if m=='r' else 'jet'))

                if p == 'l_':
                    plt.ylabel(n,fontsize=fontsize-2)

        f.subplots_adjust(wspace=0.01, hspace=0, bottom=0.05)


        cbar_ax = f.add_axes([0.3, 0.028, 0.4, 0.02])
        cb = f.colorbar(im, orientation='horizontal', cax=cbar_ax)
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)

        # plt.show()

        fout = path / 'plots' / ('spatial_plot_relative_metrics_' + m + '.png')
        f.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()


def spatial_plot_relative_metrics_ci_diff(sensors, mode='abs'):

    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2']

    tuples = ['_'.join(t) for t in combinations(sensors, 2)]
    names = [' - '.join(t) for t in combinations(sensors, 2)]

    params = ['p_', 'diff_']

    figsize = (8.5,12)
    fontsize = 14

    metric = ['bias','r', 'ubrmsd'][0:1]
    if mode == 'abs':
        cbrange = [[-0.12,0.12], [0,0.8], [0,0.1]][0:1]
        cbrange_diff = [[0,0.07], [0,0.8], [0,0.04]][0:1]
    else:
        cbrange = [[-0.12, 0.12], [0, 0.8], [0, 0.06]][0:1]
        cbrange_diff = [[0, 0.03], [0, 0.5], [0, 0.02]][0:1]

    for m, cb, cb_diff in zip(metric, cbrange, cbrange_diff):

        titles = ['R$^2$' if m=='r' else m, 'CI$_{97.5}$ - CI$_{2.5}$']

        f = plt.figure(figsize=figsize)

        i = 0
        for t, n in zip(tuples, names):
            for p, tit in zip(params, titles):
                i += 1

                plt.subplot(6,2,i)

                if i < 3:
                    plt.title(tit, fontsize=fontsize)

                if (m == 'bias') & (n[0:5] == 'ASCAT'):
                    plt.gca().set_frame_on(False)
                    plt.gca().set_xticks([])
                    plt.gca().set_yticks([])
                    continue

                if p == 'p_':
                    tag = m + '_grid_'+mode+'_' + p + t

                    if tag not in res:
                        plt.gca().set_frame_on(False)
                        plt.gca().set_xticks([])
                        plt.gca().set_yticks([])
                        continue

                    if m == 'r':
                        res[tag] *= res[tag]
                else:

                    tag_u = m + '_grid_'+mode+'_u_' + t
                    tag_l = m + '_grid_'+mode+'_l_' + t

                    if tag_u not in res:
                        plt.gca().set_frame_on(False)
                        plt.gca().set_xticks([])
                        plt.gca().set_yticks([])
                        continue

                    if m == 'r':
                        res[tag_u] *= res[tag_u]
                        res[tag_l] *= res[tag_l]

                    res[tag] = res[tag_u] - res[tag_l]

                if p == 'p_':
                    im_r = plot_ease_img(res, tag, fontsize=fontsize, cbrange=cb, cmap=('jet_r' if m=='r' else 'jet'))
                else:
                    im = plot_ease_img(res, tag, fontsize=fontsize, cbrange=cb_diff, cmap='jet')

                if i < 3:
                    plt.title(tit, fontsize=fontsize)

                if p == 'p_':
                    plt.ylabel(n,fontsize=fontsize-2)

        f.subplots_adjust(wspace=0.02, hspace=0.0, bottom=0.05)

        pos1 = im_r.axes.get_position()
        pos2 = im.axes.get_position()

        cbar_ax = f.add_axes([pos1.x0, 0.028, pos1.x1 - pos1.x0, 0.02])

        ticks = np.arange(cb[0],cb[1],round((cb[1] - cb[0]) / 5 * 1000 ) / 1000)
        cbar = f.colorbar(im_r, orientation='horizontal', cax=cbar_ax, ticks=ticks)
        for t in cbar.ax.get_xticklabels():
            t.set_fontsize(fontsize)
        # cb.ax.set_xticklabels()

        ticks = np.arange(cb_diff[0],cb_diff[1],round((cb_diff[1] - cb_diff[0]) / 5 * 1000 ) / 1000)
        cbar_ax = f.add_axes([pos2.x0, 0.028, pos2.x1 - pos2.x0, 0.02])
        cbar = f.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=ticks)
        for t in cbar.ax.get_xticklabels():
            t.set_fontsize(fontsize)

        # plt.show()

        fout = path / 'plots' / (mode+'_spatial_plot_relative_metrics_' + m + '_ci_diff.png')
        f.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()


def spatial_plot_n(sensors, mode='abs'):

    path = Paths().result_root / ('_'.join(sensors))

    res = pd.read_csv(path / 'result.csv', index_col=0)

    sensors = ['ASCAT', 'SMOS', 'MERRA2']
    tuples = ['_'.join(t) for t in combinations(sensors, 2)]
    names = [' - '.join(t) for t in combinations(sensors, 2)]

    figsize = (17, 9)
    fontsize = 14

    cbrange1 = [0,(150 if mode == 'abs' else 350)]
    cbrange2 = [0,600]

    f = plt.figure(figsize=figsize)

    i = 0
    for t, n in zip (tuples,names):
        i += 1

        if (i % 3) == 0:
            i += 1

        plt.subplot(3, 3, i)

        tag = 'n_corr_grid_' + mode + '_' + t

        if tag not in res:
            plt.gca().set_frame_on(False)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            continue

        im1 = plot_ease_img(res, tag, fontsize=fontsize, cbrange=cbrange1)

        plt.title(n, fontsize=fontsize)

    plt.subplot(3, 3, 3)
    plot_ease_img(res, 'n_corr_grid_' + mode + '_tc', fontsize=fontsize, cbrange=cbrange1)
    plt.title('TCA', fontsize=fontsize)

    plt.subplot(3, 3, 6)
    im2 = plot_ease_img(res, 'n_grid', fontsize=fontsize, cbrange=cbrange2)
    plt.title('Uncorrected', fontsize=fontsize)

    f.subplots_adjust(wspace=0.04, hspace=0.03, bottom=0.05)

    x0 = (f.axes[-4].get_position().x1 + f.axes[-4].get_position().x0) / 2
    x1 = (f.axes[-3].get_position().x1 + f.axes[-3].get_position().x0) / 2
    cbar_ax = f.add_axes([x0, 0.03, x1-x0, 0.025])
    cb = f.colorbar(im1, orientation='horizontal', cax=cbar_ax)
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)

    x0 = f.axes[-2].get_position().x0
    x1 = f.axes[-2].get_position().x1
    y0 = f.axes[-2].get_position().y0
    cbar_ax = f.add_axes([x0, y0-0.04, x1-x0, 0.025])
    cb = f.colorbar(im2, orientation='horizontal', cax=cbar_ax)
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)

    # plt.show()

    fout = path / 'plots' / (mode + '_spatial_plot_sample_size.png')
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


if __name__=='__main__':

    mode = 'abs'
    # mode = 'anom_st'

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2', 'ISMN']
    # sensors = ['ASCAT', 'SMOS', 'MERRA2', 'ISMN']

    # spatial_plot_tca_diff(mode)

    # boxplot_tca_ismn(sensors, mode)
    # boxplot_relative_metrics_ismn(sensors, mode)


    boxplot_tca(sensors, mode)
    # boxplot_relative_metrics(sensors, mode)


    # spatial_plot_tca_ci_diff(sensors, mode)
    # spatial_plot_relative_metrics_ci_diff(sensors, mode)

    # spatial_plot_n(sensors, mode)

    # spatial_plot_tca(sensors, mode)
    # spatial_plot_relative_metrics(sensors, mode)
