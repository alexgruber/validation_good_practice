
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from multiprocessing import Pool

from itertools import combinations, repeat

from validation_good_practice.data_readers.interface import reader
from validation_good_practice.ancillary.metrics import bias, ubRMSD, Pearson_R, TCA, correct_n
from validation_good_practice.ancillary.paths import Paths


def run():
    """ This in the main routine that parallelizes the validation """

    # The processing will be parallelized on 30 kernels
    parts = 30

    # Confidence intervals will be calculated at a 80% confidence level
    alpha = 0.80

    # The validation will be done using all available sensors.
    sensors = ['ASCAT', 'SMOS', 'SMAP', 'MERRA2', 'ISMN']

    res_path = Paths().result_root / ('CI%i' % (alpha * 100)) / ('_'.join(sensors))
    if not res_path.exists():
        res_path.mkdir(parents=True)

    # Parallelized processing
    p = Pool(parts)
    arg1 = np.arange(parts) + 1
    arg2 = repeat(parts, parts)
    arg3 = repeat(sensors, parts)
    arg4 = repeat(alpha, parts)
    arg5 = repeat(res_path, parts)
    p.starmap(main, zip(arg1,arg2,arg3,arg4,arg5))

    # merge in parallel generated results into one single result file.
    merge_result_files(res_path)


    # Optional: Non-parallelized run for debugging purposes
    # part = 10
    # main(part, parts, sensors, alpha, res_path)



def merge_result_files(path):
    """ This routine merges .csv files in a particular folder into one single file. """

    files = list(path.glob('**/*.csv'))

    result = pd.DataFrame()
    for f in files:
        tmp = pd.read_csv(f, index_col=0)
        result = result.append(tmp)
        f.unlink()

    result.sort_index().to_csv(path / 'result.csv', float_format='%0.3f')


def main(part, parts, sensors, alpha, res_path):
    """
    This calculates validation statistics for a subset of the study domain.

    Attributes
    ----------
    part : int
        part of the subset to process
    parts : int
        number of subsets to divide the study domain into
    sensors : list of str
        sensors to be considered in the validation
    alpha : float [0,1]
        confidence level of the confidence intervals
    res_path : pathlib.Path
        Path where to store the result file

    """

    result_file = res_path / ('result_%i.csv' % part)

    lut = pd.read_csv(Paths().lut, index_col=0)

    # Split grid cell list for parallelization
    subs = (np.arange(parts + 1) * len(lut) / parts).astype('int')
    subs[-1] = len(lut)
    start = subs[part - 1]
    end = subs[part]

    # Look-up table that contains the grid cells to iterate over
    lut = lut.iloc[start:end, :]

    io = reader(sensors)

    for cnt, (gpi, data) in enumerate(lut.iterrows()):
        print('%i / %i' % (cnt, len(lut)))

        # Get the template of data fields to store results into
        res = result_template(sensors, gpi)

        res.loc[gpi, 'col'] = data.ease2_col
        res.loc[gpi, 'row'] = data.ease2_row

        try:
            mode, dfs = io.read(gpi, calc_anom_lt=False)

            # Iterate over all data sets (absolute values and anomalies collocated with and without ISMN)
            for m, df in zip(mode,dfs):

                if df is not None:

                    # Only calculate corrected sample size once to speed up processing
                    res.loc[gpi, 'n_corr_' + m + '_tc'] = correct_n(df)

                    # check if current data set contains ISMN data or not.
                    scl = m[0:4]
                    if scl == 'grid':
                        res.loc[gpi, 'n_grid'] = len(df)
                    else:
                        res.loc[gpi, 'n_ismn'] = len(df)

                    b = bias(df, alpha=alpha)
                    R = Pearson_R(df, alpha=alpha, n_corr=b.loc[:,:,'n_corr'])

                    # rescale all columns to MERRA2 before calculating ubRMSD
                    tmp_df = df.copy()
                    for col in sensors:
                        if (col == 'MERRA2')|(not col in tmp_df):
                            continue

                        tmp_df.loc[:, col] = ((tmp_df[col] - tmp_df[col].mean()) / tmp_df[col].std()) * tmp_df[
                            'MERRA2'].std() + tmp_df['MERRA2'].mean()
                    ubrmsd = ubRMSD(tmp_df, alpha=alpha, n_corr=b.loc[:,:,'n_corr'])

                    res.loc[gpi, 'n_'+ scl] = len(df)

                    # calculate relative metrics for all pair-wise combinations
                    for t in list(combinations(df.columns.values, 2)):

                        res.loc[gpi, 'n_corr_' + m + '_' + '_'.join(t)] = R.loc[t[0],t[1],'n_corr']

                        res.loc[gpi, 'bias_' + m + '_l_' + '_'.join(t)] = b.loc[t[0],t[1],'CI_l_corr']
                        res.loc[gpi, 'bias_' + m + '_p_' + '_'.join(t)] = b.loc[t[0],t[1],'bias']
                        res.loc[gpi, 'bias_' + m + '_u_' + '_'.join(t)] = b.loc[t[0],t[1],'CI_u_corr']

                        res.loc[gpi, 'ubrmsd_' + m + '_l_' + '_'.join(t)] = ubrmsd.loc[t[0],t[1],'CI_l_corr']
                        res.loc[gpi, 'ubrmsd_' + m + '_p_' + '_'.join(t)] = ubrmsd.loc[t[0],t[1],'ubRMSD']
                        res.loc[gpi, 'ubrmsd_' + m + '_u_' + '_'.join(t)] = ubrmsd.loc[t[0],t[1],'CI_u_corr']

                        res.loc[gpi, 'r_' + m + '_l_' + '_'.join(t)] = R.loc[t[0],t[1],'CI_l_corr']
                        res.loc[gpi, 'r_' + m + '_p_' + '_'.join(t)] = R.loc[t[0],t[1],'R']
                        res.loc[gpi, 'r_' + m + '_u_' + '_'.join(t)] = R.loc[t[0],t[1],'CI_u_corr']
                        res.loc[gpi, 'p_' + m + '_p_' + '_'.join(t)] = R.loc[t[0],t[1],'p']


                    # calculate TCA-based metrics for all triplets except those including both SMOS and SMAP
                    for t in list(combinations(df.columns.values, 3)):
                        if (('SMOS' in t) & ('SMAP' in t)):
                            continue

                        tcstr = '_tc_' + '_'.join(t)

                        tca = TCA(df[list(t)], alpha=alpha)

                        # calculate TCA only for coarse-resolution data sets triplets that have been collocated
                        # without ISMN data
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

        except:
            continue

        if not result_file.exists():
            res.to_csv(result_file, float_format='%0.3f')
        else:
            res.to_csv(result_file, float_format='%0.3f', mode='a', header=False)


def result_template(sensors, gpi):
    """ This creates a result templates with all possible data fields """

    tuples = list(combinations(sensors, 2))
    triplets = list(combinations(sensors, 3))

    res = {'col': 0,
           'row': 0,
           'n_grid': np.nan,
           'n_ismn': np.nan}

    # TODO: Currently, that's hardcoded for calculating short-term anomalies only!!
    for m in ['grid_abs', 'grid_anom_st', 'ismn_abs', 'ismn_anom_st']:

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

        res.update(dict(zip(['n_corr_' + m + '_' + '_'.join(t) for t in tuples],np.full(len(tuples),np.nan))))

        res.update({'n_corr_' + m + '_tc': np.nan})

    return pd.DataFrame(res, index=(gpi,))
