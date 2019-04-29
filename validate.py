
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from multiprocessing import Pool

from itertools import combinations

from scipy.stats import pearsonr

from validation_good_practice.data_readers.interface import reader
from validation_good_practice.ancillary.metrics import bias, ubRMSD, Pearson_R, TCA
from validation_good_practice.ancillary.paths import Paths

def run():

    # part = 10
    # main(part)

    parts = np.arange(30) + 1
    p = Pool(30)
    p.map(main, parts)

def result_template(sensors, gpi):

    tuples = list(combinations(sensors, 2))
    triplets = list(combinations(sensors, 3))

    res = {'col': 0,
           'row': 0,
           'n_grid': np.nan,
           'n_ismn': np.nan}
           # 'dt_opt_grid': np.nan,
           # 'dt_opt_ismn': np.nan,
           # 'bl_abs_grid': np.nan,
           # 'bl_anom_st_grid:': np.nan,
           # 'bl_anom_lt_grid:': np.nan,
           # 'bl_abs_ismn': np.nan,
           # 'bl_anom_st_ismn:': np.nan,
           # 'bl_anom_lt_ismn:': np.nan}

    for m in ['grid_abs', 'grid_anom_st', 'grid_anom_lt',
              'ismn_abs', 'ismn_anom_st', 'ismn_anom_lt']:

        # point estimate, upper and lower confidence limit, bootstrap-median
        for est in ['_p_', '_u_', '_l_', '_m_']:

            # all possible TC combinations, i.e., not using SMOS and SMAP together
            for tc in [t for t in triplets if not (('SMOS' in t)&('SMAP' in t))]:

                tcstr = '_tc_' + '_'.join(tc)

                if (m[0:4] != 'grid')|(tc[2] != 'ISMN'):
                    res.update(dict(zip(['bias_' + m + est + s + tcstr for s in tc],np.full(len(tc),np.nan))))
                    res.update(dict(zip(['ubrmse_' + m + est + s + tcstr for s in tc],np.full(len(tc),np.nan))))
                    res.update(dict(zip(['r2_' + m + est + s + tcstr for s in tc],np.full(len(tc),np.nan))))

            # No median available for analytical CIs
            if est != '_m_':
                res.update(dict(zip(['bias_' + m + est + '_'.join(t) for t in tuples],np.full(len(tuples),np.nan))))
                res.update(dict(zip(['ubrmsd_' + m + est + '_'.join(t) for t in tuples],np.full(len(tuples),np.nan))))
                res.update(dict(zip(['r_' + m + est + '_'.join(t) for t in tuples],np.full(len(tuples),np.nan))))

        res.update(dict(zip(['p_' + m + '_p_' + '_'.join(t) for t in tuples], np.full(len(tuples),np.nan))))

        res.update(dict(zip(['n_corr_' + m + '_'.join(t) for t in tuples],np.full(len(tuples),np.nan))))

    return pd.DataFrame(res, index=(gpi,))


def main(part):

    parts = 30

    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2', 'ISMN']

    paths = Paths()

    result_file = paths.result_root / ('_'.join(sensors)) / ('result_%i.csv' % part)
    if not result_file.parent.exists():
        result_file.parent.mkdir(parents=True)

    lut = pd.read_csv(paths.lut, index_col=0)

    # Split grid cell list for parallelization
    subs = (np.arange(parts + 1) * len(lut) / parts).astype('int')
    subs[-1] = len(lut)
    start = subs[part - 1]
    end = subs[part]

    lut = lut.iloc[start:end, :]

    io = reader(sensors)

    for cnt, (gpi, data) in enumerate(lut.iterrows()):
        print('%i / %i' % (cnt, len(lut)))

        res = result_template(sensors, gpi)

        res.loc[gpi, 'col'] = data.ease2_col
        res.loc[gpi, 'row'] = data.ease2_row

        try:
        # if True:
            mode, dfs = io.read(gpi)

            for m, df in zip(mode,dfs):

                if df is not None:

                    scl = m[0:4]

                    if scl == 'grid':
                        res.loc[gpi, 'n_grid'] = len(df)
                    else:
                        res.loc[gpi, 'n_ismn'] = len(df)


                    b = bias(df)
                    ubrmsd = ubRMSD(df, n_corr=b.loc[:,:,'n_corr'])
                    R = Pearson_R(df, n_corr=b.loc[:,:,'n_corr'])

                    res.loc[gpi, 'n_'+ scl] = len(df)

                    for t in list(combinations(df.columns.values, 2)):

                        res.loc[gpi, 'n_corr_' + m + '_' + '_'.join(t)] = R.loc[t[0],t[1],'n_corr']

                        res.loc[gpi, 'bias_' + m + '_l_' + '_'.join(t)] = b.loc[t[0],t[1],'CI_l']
                        res.loc[gpi, 'bias_' + m + '_p_' + '_'.join(t)] = b.loc[t[0],t[1],'bias']
                        res.loc[gpi, 'bias_' + m + '_u_' + '_'.join(t)] = b.loc[t[0],t[1],'CI_u']

                        res.loc[gpi, 'ubrmsd_' + m + '_l_' + '_'.join(t)] = ubrmsd.loc[t[0],t[1],'CI_l']
                        res.loc[gpi, 'ubrmsd_' + m + '_p_' + '_'.join(t)] = ubrmsd.loc[t[0],t[1],'ubRMSD']
                        res.loc[gpi, 'ubrmsd_' + m + '_u_' + '_'.join(t)] = ubrmsd.loc[t[0],t[1],'CI_u']

                        res.loc[gpi, 'r_' + m + '_l_' + '_'.join(t)] = R.loc[t[0],t[1],'CI_l']
                        res.loc[gpi, 'r_' + m + '_p_' + '_'.join(t)] = R.loc[t[0],t[1],'R']
                        res.loc[gpi, 'r_' + m + '_u_' + '_'.join(t)] = R.loc[t[0],t[1],'CI_u']
                        res.loc[gpi, 'p_' + m + '_p_' + '_'.join(t)] = R.loc[t[0],t[1],'p']


                    for t in list(combinations(df.columns.values, 3)):

                        if (('SMOS' in t) & ('SMAP' in t)):
                            continue

                        tcstr = '_tc_' + '_'.join(t)

                        tca = TCA(df[list(t)])

                        if (scl != 'grid') | (t[2] != 'ISMN'):

                            for s in t:
                                res.loc[gpi, 'bias_' + m + '_p_' + s + tcstr] = tca.loc['beta_p',s]
                                res.loc[gpi, 'bias_' + m + '_l_' + s + tcstr] = tca.loc['beta_l',s]
                                res.loc[gpi, 'bias_' + m + '_m_' + s + tcstr] = tca.loc['beta_m',s]
                                res.loc[gpi, 'bias_' + m + '_u_' + s + tcstr] = tca.loc['beta_u',s]

                                res.loc[gpi, 'ubrmse_' + m + '_p_' + s + tcstr] = tca.loc['ubRMSE_p',s]
                                res.loc[gpi, 'ubrmse_' + m + '_l_' + s + tcstr] = tca.loc['ubRMSE_l',s]
                                res.loc[gpi, 'ubrmse_' + m + '_m_' + s + tcstr] = tca.loc['ubRMSE_m',s]
                                res.loc[gpi, 'ubrmse_' + m + '_u_' + s + tcstr] = tca.loc['ubRMSE_u',s]

                                res.loc[gpi, 'r2_' + m + '_p_' + s + tcstr] = tca.loc['r2_p',s]
                                res.loc[gpi, 'r2_' + m + '_l_' + s + tcstr] = tca.loc['r2_l',s]
                                res.loc[gpi, 'r2_' + m + '_m_' + s + tcstr] = tca.loc['r2_m',s]
                                res.loc[gpi, 'r2_' + m + '_u_' + s + tcstr] = tca.loc['r2_u',s]

                    pass

        except:
            print('gpi %i failed.' % gpi)
            continue

        if not result_file.exists():
            res.to_csv(result_file, float_format='%0.3f')
        else:
            res.to_csv(result_file, float_format='%0.3f', mode='a', header=False)


if __name__=='__main__':

    run()