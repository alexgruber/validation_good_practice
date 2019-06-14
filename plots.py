
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


def boxplot_tca_ismn(path, sensors):

    res = pd.read_csv(path / 'result.csv', index_col=0)

    metric = ['bias', 'r2', 'ubrmse']
    ylim = [[0, 1.5], [0, 1], [0, 0.1]]

    sensors = sensors[:-2]

    alpha = 100 - float(path.parent.name[2:])
    cistr_u = 'CI$_{%i}$' % (100 - alpha / 2)
    cistr_l = 'CI$_{%i}$' % (alpha / 2)

    params = ['l_', 'm_', 'u_']
    titles = [cistr_l, 'CI$_{50}$', cistr_u]

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

    fout = path / 'plots' / ('boxplot_tca_ismn.png')
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


def boxplot_tca(path, sensors):

    res = pd.read_csv(path / 'result.csv', index_col=0)

    metric = ['r2', 'ubrmse']
    ylim = [[0, 1], [0, 0.08]]

    sensors = sensors[:-2]

    alpha = 100 - float(path.parent.name[2:])
    cistr_u = 'CI$_{%i}$' % (100 - alpha / 2)
    cistr_l = 'CI$_{%i}$' % (alpha / 2)

    params = ['l_', 'm_', 'u_']
    titles = [cistr_l, 'CI$_{50}$', cistr_u]

    figsize = (13, 5)
    fontsize = 14

    pos = [i + j for i in np.arange(1,len(sensors)+1) for j in [-0.2,0,0.2]]
    colors = [s for n in np.arange(len(sensors)) for s in ['lightblue', 'lightgreen', 'coral'] ]

    f = plt.figure(figsize=figsize)

    n = 0
    for m, yl in zip(metric, ylim):
        for mode in ['abs', 'anom_st']:
            n += 1

            ax = plt.subplot(2, 2, n)

            if n == 4:
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

            if m == 'r2':
                plt.yticks(fontsize=fontsize)
            else:
                plt.yticks(np.arange(0,0.1,0.02), fontsize=fontsize)

            for i in np.arange(1,3):
                plt.axvline(i + 0.5, linewidth=1, color='k')

            if mode == 'abs':
                plt.ylabel(m, fontsize=fontsize)

            if m == 'r2':
                plt.title(('Absolute values' if mode=='abs' else 'Anomalies'),fontsize=fontsize)

    f.subplots_adjust(hspace=0.3)

    plt.figlegend((box['boxes'][0:3]), titles, 'upper left', bbox_to_anchor=(axpos.x1-0.15,axpos.y1-0.042), fontsize=fontsize - 2)

    fout = path / 'plots' / ('boxplot_tca.png')
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


def boxplot_relative_metrics_ismn(path):

    res = pd.read_csv(path / 'result.csv', index_col=0)

    metric = ['bias', 'r', 'ubrmsd']
    ylim = [[-0.12,0.12], [0, 1], [0.02, 0.25]]

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2', 'ISMN']
    tuples = ['_'.join(t) for t in combinations(sensors, 2)]
    names = [' - '.join(t) for t in combinations(sensors, 2)]

    alpha = 100 - float(path.parent.name[2:])
    cistr_u = 'CI$_{%i}$' % (100 - alpha / 2)
    cistr_l = 'CI$_{%i}$' % (alpha / 2)

    params = ['l_', 'p_', 'u_']
    titles = [cistr_l, 'p.est.', cistr_u]

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

    fout = path / 'plots' / ('boxplot_relative_metrics_ismn.png')
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


def boxplot_relative_metrics(path):

    res = pd.read_csv(path / 'result.csv', index_col=0)

    metric = ['bias', 'r', 'ubrmsd']
    ylim_abs = [[-0.12,0.07], [0, 1], [0.0, 0.1]]
    ylim_anom = [[-0.12,0.12], [0, 1], [0.0, 0.06]]

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2']
    tuples = ['_'.join(t) for t in combinations(sensors, 2)]
    names = [' - '.join(t) for t in combinations(sensors, 2)]

    alpha = 100 - float(path.parent.name[2:])
    cistr_u = 'CI$_{%i}$' % (100 - alpha / 2)
    cistr_l = 'CI$_{%i}$' % (alpha / 2)

    params = ['l_', 'p_', 'u_']
    titles = [cistr_l, 'p.est.', cistr_u]

    figsize = (15, 12)
    fontsize = 14

    f = plt.figure(figsize=figsize)

    r = -1
    for m, yl_abs, yl_anom in zip(metric, ylim_abs, ylim_anom):

        for mode in ['abs', 'anom_st']:
            if (m == 'bias') & (mode == 'anom_st'):
                continue
            r+= 1

            tup = (tuples[3::] if m == 'bias' else tuples)
            nam = (names[3::] if m == 'bias' else names)

            pos = [i + j for i in np.arange(1,len(tup)+1) for j in [-0.2,0,0.2]]
            colors = [s for n in np.arange(len(tup)) for s in ['lightblue', 'lightgreen', 'coral'] ]

            ax = plt.subplot2grid((5, 2),(r,0), colspan=(1 if m=='bias' else 2))

            if r == 0:
                axpos = ax.get_position()

            plt.grid(color='k', linestyle='--', linewidth=0.25)

            data = list()

            for t in tup:
                for p in params:

                    tag = m + '_grid_'+ mode + '_' + p + t

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

            plt.xlim(0.5, len(tup) + 0.5)
            plt.xticks(np.arange(1,len(tup)+1), nam, fontsize=fontsize)

            plt.ylim((yl_abs if mode == 'abs' else yl_anom))
            plt.yticks(fontsize=fontsize)

            for i in np.arange(1,len(tup)):
                plt.axvline(i + 0.5, linewidth=1, color='k')


            tmp_m = 'R$^2$' if m == 'r' else m
            label = (tmp_m + ' (abs.)' if mode == 'abs' else tmp_m + ' (anom.)')
            plt.ylabel(label, fontsize=fontsize)

    f.subplots_adjust(hspace=0.25)

    plt.figlegend((box['boxes'][0:3]), titles, 'upper left', bbox_to_anchor=(axpos.x1 - 0.04, axpos.y0 + 0.0),
                  fontsize=fontsize - 2)

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
        cb = m.colorbar(im, "bottom", size="8%", pad="4%", ticks=ticks)
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)

    plt.title(title,fontsize=fontsize)

    if print_median is True:
        x, y = m(-79, 25)
        plt.text(x, y, 'median = %.2f' % np.ma.median(img_masked), fontsize=fontsize-2)

    return im


def spatial_plot_tca_diff(path):

    res = pd.read_csv(path / 'result.csv', index_col=0)

    figsize = (16,8)
    fontsize = 14

    metrics = ['r2', 'ubrmse']
    cbrange = [[-0.2,0.2], [-0.02,0.02]]

    f = plt.figure(figsize=figsize)

    n = 0
    for mode in ['abs', 'anom_st']:
        for m, cb in zip(metrics,cbrange):

            n += 1

            plt.subplot(2,2,n)

            tag1 = m + '_grid_'+mode+'_m_ASCAT_tc_ASCAT_SMOS_MERRA2'
            tag2 = m + '_grid_'+mode+'_m_ASCAT_tc_ASCAT_SMAP_MERRA2'

            tag = 'diff'

            res[tag] = (res[tag1] - res[tag2])

            im = plot_ease_img(res, tag, fontsize=fontsize, cbrange=cb, cmap=('jet_r' if m == 'r2' else 'jet'), print_median=True)

            if m == 'r2':
                plt.ylabel(('Absolute' if mode == 'abs' else 'Anomalies'),fontsize=fontsize)

            if mode == 'abs':
                plt.title('$\Delta$' + m + ' ASCAT', fontsize=fontsize)

    f.subplots_adjust(hspace=0.0, wspace=0.02, bottom=0.09)

    pos1 = f.axes[-2].get_position()
    pos2 = f.axes[-1].get_position()

    im1 = f.axes[-2].collections[-1]
    im2 = f.axes[-1].collections[-1]

    ticks = np.arange(cbrange[0][0], cbrange[0][1], 0.1)
    cbar_ax = f.add_axes([pos1.x0, 0.05, pos1.x1 - pos1.x0, 0.035])
    cbar = f.colorbar(im1, orientation='horizontal', cax=cbar_ax, ticks=ticks)
    for t in cbar.ax.get_xticklabels():
        t.set_fontsize(fontsize)

    ticks = np.arange(cbrange[1][0], cbrange[1][1] + 0.01, 0.01)
    cbar_ax = f.add_axes([pos2.x0, 0.05, pos2.x1 - pos2.x0, 0.035])
    cbar = f.colorbar(im2, orientation='horizontal', cax=cbar_ax, ticks=ticks)
    for t in cbar.ax.get_xticklabels():
        t.set_fontsize(fontsize)

    fout = path / 'plots' / ('spatial_plot_tca_diff.png')
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


def spatial_plot_tca_ci_diff(path, sensors):

    res = pd.read_csv(path / 'result.csv', index_col=0)

    sensors = sensors[:-2]

    alpha = 100 - float(path.parent.name[2:])
    cistr_u = 'CI$_{%i}$' % (100 - alpha / 2)
    cistr_l = 'CI$_{%i}$' % (alpha / 2)

    params = ['m_', 'diff_']
    titles = ['CI$_{50}$', cistr_u + ' - ' + cistr_l]

    figsize = (10,8)
    fontsize = 14

    for mode in ['abs', 'anom_st']:

        metric = ['r2', 'ubrmse']
        if mode == 'abs':
            cbrange = [[0,1], [0,0.06]]
            cbrange_diff = [[0,0.6], [0,0.04]]
        else:
            cbrange = [[0,1], [0,0.04]]
            cbrange_diff = [[0,0.6], [0,0.03]]


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

            if m == 'r2':
                ticks = np.arange(cb[0], cb[1], 0.2)
                ticks_diff = np.arange(cb_diff[0], cb_diff[1]+0.1, 0.1)
            else:
                ticks = np.arange(cb[0], cb[1], 0.01)
                ticks_diff = np.arange(cb_diff[0], cb_diff[1]+0.01, 0.01)


            cbar_ax = f.add_axes([pos1.x0, 0.07, pos1.x1 - pos1.x0, 0.027])
            cb = f.colorbar(im_r, orientation='horizontal', cax=cbar_ax, ticks=ticks)
            for t in cb.ax.get_xticklabels():
                t.set_fontsize(fontsize)

            cbar_ax = f.add_axes([pos2.x0, 0.07, pos2.x1 - pos2.x0, 0.027])
            cb = f.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=ticks_diff)
            for t in cb.ax.get_xticklabels():
                t.set_fontsize(fontsize)

            fout = path / 'plots' / ('spatial_plot_tc_' + m + '_' + mode + '_ci_diff.png')
            f.savefig(fout, dpi=300, bbox_inches='tight')
            plt.close()


def spatial_plot_relative_metrics_ci_diff(path):

    res = pd.read_csv(path / 'result.csv', index_col=0)

    params = ['p_', 'diff_']

    fontsize = 14

    mode = 'abs'
    m = 'bias'

    sensors = ['SMOS', 'SMAP', 'MERRA2']

    tuples = ['_'.join(t) for t in combinations(sensors, 2)]
    names = [' - '.join(t) for t in combinations(sensors, 2)]

    figsize = (11, 8)

    if mode == 'abs':
        cb = [-0.12,0.12]
        cb_diff = [0,0.07]
    else:
        cb = [-0.12, 0.12]
        cb_diff = [0, 0.03]

    alpha = 100 - float(path.parent.name[2:])
    cistr_u = 'CI$_{%i}$' % (100 - alpha / 2)
    cistr_l = 'CI$_{%i}$' % (alpha / 2)
    titles = ['bias', cistr_u + ' - ' + cistr_l]

    f = plt.figure(figsize=figsize)

    i = 0
    for t, n in zip(tuples, names):
        for p, tit in zip(params, titles):
            i += 1

            plt.subplot(3, 2, i)

            if p == 'p_':
                tag = m + '_grid_' + mode + '_' + p + t

                if tag not in res:
                    plt.gca().set_frame_on(False)
                    plt.gca().set_xticks([])
                    plt.gca().set_yticks([])
                    continue

            else:

                tag_u = m + '_grid_' + mode + '_u_' + t
                tag_l = m + '_grid_' + mode + '_l_' + t

                if tag_u not in res:
                    plt.gca().set_frame_on(False)
                    plt.gca().set_xticks([])
                    plt.gca().set_yticks([])
                    continue

                res[tag] = res[tag_u] - res[tag_l]

            if p == 'p_':
                im_r = plot_ease_img(res, tag, fontsize=fontsize, cbrange=cb, cmap= 'jet')
            else:
                im = plot_ease_img(res, tag, fontsize=fontsize, cbrange=cb_diff, cmap='jet')

            if i < 3:
                plt.title(tit, fontsize=fontsize)

            if p == 'p_':
                plt.ylabel(n, fontsize=fontsize - 2)

    f.subplots_adjust(wspace=0.02, hspace=0.04, bottom=0.1)

    pos1 = im_r.axes.get_position()
    pos2 = im.axes.get_position()

    ticks = np.arange(cb[0], cb[1], 0.04)
    ticks_diff = np.arange(cb_diff[0], cb_diff[1] + 0.01, 0.02)

    cbar_ax = f.add_axes([pos1.x0, 0.065, pos1.x1 - pos1.x0, 0.025])
    cbar = f.colorbar(im_r, orientation='horizontal', cax=cbar_ax, ticks=ticks)
    for t in cbar.ax.get_xticklabels():
        t.set_fontsize(fontsize)

    cbar_ax = f.add_axes([pos2.x0, 0.065, pos2.x1 - pos2.x0, 0.025])
    cbar = f.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=ticks_diff)
    for t in cbar.ax.get_xticklabels():
        t.set_fontsize(fontsize)

    fout = path / 'plots' / ('spatial_plot_relative_metrics_' + m + '_' + mode + '_ci_diff.png')
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

    for mode in ['abs', 'anom_st']:

        sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2']

        tuples = ['_'.join(t) for t in combinations(sensors, 2)]
        names = [' - '.join(t) for t in combinations(sensors, 2)]

        figsize = (8.5,12)

        metric = ['r', 'ubrmsd']
        if mode == 'abs':
            cbrange = [[0,0.8], [0,0.1]]
            cbrange_diff = [[0,0.8], [0,0.04]]
            # cbrange = [[-0.12,0.12], [0,0.8], [0,0.1]]
            # cbrange_diff = [[0,0.07], [0,0.8], [0,0.04]]
        else:
            cbrange = [[0, 0.8], [0, 0.08]]
            cbrange_diff = [[0, 0.4], [0, 0.02]]
            # cbrange = [[-0.12, 0.12], [0, 0.8], [0, 0.06]]
            # cbrange_diff = [[0, 0.03], [0, 0.5], [0, 0.02]]

        for m, cb, cb_diff in zip(metric, cbrange, cbrange_diff):

            titles = ['R$^2$' if m=='r' else m, cistr_u + ' - ' + cistr_l]

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

            if m == 'r':
                ticks = np.arange(cb[0], cb[1], 0.2)
                ticks_diff = np.arange(cb_diff[0], cb_diff[1] + 0.1, (0.2 if mode == 'abs' else 0.1))
            else:
                ticks = np.arange(cb[0], cb[1], 0.02)
                ticks_diff = np.arange(cb_diff[0], cb_diff[1] + 0.01, 0.01)

            cbar = f.colorbar(im_r, orientation='horizontal', cax=cbar_ax, ticks=ticks)
            for t in cbar.ax.get_xticklabels():
                t.set_fontsize(fontsize)

            cbar_ax = f.add_axes([pos2.x0, 0.028, pos2.x1 - pos2.x0, 0.02])
            cbar = f.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=ticks_diff)
            for t in cbar.ax.get_xticklabels():
                t.set_fontsize(fontsize)

            fout = path / 'plots' / ('spatial_plot_relative_metrics_' + m + '_' + mode +'_ci_diff.png')
            f.savefig(fout, dpi=300, bbox_inches='tight')
            plt.close()


def spatial_plot_n(path):

    res = pd.read_csv(path / 'result.csv', index_col=0)

    fontsize = 16

    f = plt.figure(figsize=(10,6))

    plot_ease_img(res, 'n_grid', fontsize=fontsize, cbrange=[0,400], plot_cb=True)
    plt.title('# temporal matches', fontsize=fontsize)

    fout = path / 'plots' / 'spatial_plot_sample_size.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

    fontsize = 14

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2']
    tuples = ['_'.join(t) for t in combinations(sensors, 2)]
    names = [' - '.join(t) for t in combinations(sensors, 2)]

    for mode in ['abs', 'anom_st']:

        cbrange = [0,(200 if mode == 'abs' else 400)]

        f = plt.figure(figsize=(17, 9))

        i = 0
        for t, n in zip (tuples,names):
            i += 1

            plt.subplot(3, 3, i)

            tag = 'n_corr_grid_' + mode + '_' + t

            if tag not in res:
                plt.gca().set_frame_on(False)
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
                continue

            im = plot_ease_img(res, tag, fontsize=fontsize, cbrange=cbrange)

            plt.title(n, fontsize=fontsize)

        plt.subplot(3, 3, 8)
        plot_ease_img(res, 'n_corr_grid_' + mode + '_tc', fontsize=fontsize, cbrange=cbrange, plot_cb=True)
        plt.title('All', fontsize=fontsize)

        f.subplots_adjust(wspace=0.04, hspace=0.03, bottom=0.05)

        fout = path / 'plots' / ('spatial_plot_sample_size_corrected_' + mode + '.png')
        f.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()


def generate_plots():

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2', 'ISMN']

    path = Paths().result_root / 'CI80' / ('_'.join(sensors))

    if not (path / 'plots').exists():
        Path.mkdir(path / 'plots')

    spatial_plot_n(path)

    spatial_plot_relative_metrics_ci_diff(path)
    boxplot_relative_metrics(path)

    spatial_plot_tca_diff(path)
    spatial_plot_tca_ci_diff(path, sensors)
    boxplot_tca(path, sensors)




