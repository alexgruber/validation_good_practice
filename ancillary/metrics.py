
import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import pearsonr, norm, t, chi
import scipy.optimize as optimization

def estimate_tau(in_df, n_lags=90):
    """
    Estimate characteristic time lengths for pd.DataFrame columns by fitting an exponential auto-correlation function.

    Parameters
    ----------
    in_df : pd.DataFrame
        Input data frame
    n_lags : maximum allowed lag size to be considered

    """

    df = in_df.copy().resample('1D').last()
    n_cols = len(df.columns)

    # calculate auto-correlation coefficients for different lags
    rho = np.full((n_cols,n_lags), np.nan)
    for lag in np.arange(n_lags):
        for i,col in enumerate(df):
            Ser_l = df[col].copy()
            Ser_l.index += pd.Timedelta(lag, 'D')
            rho[i,lag] = df[col].corr(Ser_l)

    # Fit exponential function to auto-correlations and estimate tau
    tau = np.full(n_cols, np.nan)
    for i in np.arange(n_cols):
        try:
            ind = np.where(~np.isnan(rho[i,:]))[0]
            if len(ind) > 10:
                popt = optimization.curve_fit(lambda x, a: np.exp(a * x), np.arange(n_lags)[ind], rho[i,ind],
                                              bounds = [-1., -1. / n_lags])[0]
                tau[i] = np.log(np.exp(-1.)) / popt
        except:
            # If fit doesn't converge, fall back to the lag where calculated auto-correlation actually drops below 1/e
            ind = np.where(rho[i,:] < np.exp(-1))[0]
            tau[i] = ind[0] if (len(ind) > 0) else n_lags # maximum = # calculated lags

    return tau


def estimate_lag1_autocorr(df, tau=None):
    """
    Estimate geometric average median lag-1 auto-correlation

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame
    tau : list of floats
        Pass auto-correlation of individual df columns if already calculated to speed up processing

    """

    # Get auto-correlation length for all time series
    if tau is None:
        tau = estimate_tau(df)

    # Calculate geometric average lag-1 auto-corr
    avg_spc_t = np.median((df.index[1::] - df.index[0:-1]).days)
    rho_i = np.exp(-avg_spc_t/tau)
    rho = rho_i.prod()**(1./len(rho_i))

    return rho


def correct_n(df, rho=None):
    """
    Calculate corrected sample size based on avergae lag-1 auto-correlation.

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame
    rho : float
        Pass average lag-1 auto-correlation if already calculated to speed up processing.

    """

    # get geometric average median lag-1 auto-correlation
    if rho is None:
        rho = estimate_lag1_autocorr(df)

    return round(len(df) * (1 - rho) / (1 + rho))


def calc_bootstrap_blocklength(df, rho=None):
    """
    Calculate optimal block length [days] for block-bootstrapping of a data frame

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame
    rho : float
        Pass average lag-1 auto-correlation if already calculated to speed up processing

    """

    # Get average lag-1 auto-correlation
    if rho is None:
        rho = estimate_lag1_autocorr(df)

    # Estimate block length (maximum 0.8 * data length)
    bl = min([round((np.sqrt(6) * rho / (1 - rho**2))**(2/3.)*len(df)**(1/3.)), round(0.8*len(df))])

    return bl


def bootstrap(df, bl=None):
    """
    Bootstrap sample generator for a data frame with given block length [days]

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame
    bl : float
        Pass optimal block-length if already calculated to speed up processing

    """

    # Get optimal bootstrap block length
    if bl is None:
        bl = calc_bootstrap_blocklength(df)

    N_df = len(df)
    t = df.index

    # build up list of all blocks (only consider block if number of data is at least half the block length)
    if bl > 1:
        blocks = list()
        for i in np.arange(N_df - int(bl / 2.)):
            ind = np.where((t >= t[i]) & (t < (t[i] + pd.Timedelta(bl, 'D'))))[0]
            if len(ind) > (bl / 2.):
                blocks.append(ind)
        N_blocks = len(blocks)

    # randomly draw a sample of blocks and trim it to df length
    while True:
        if bl == 1:
            ind = np.round(np.random.uniform(0, N_df-1, N_df)).astype('int')
        else:
            tmp_ind = np.round(np.random.uniform(0, N_blocks - 1, int(np.ceil(2. * N_df / bl)))).astype('int')
            ind = [i for block in np.array(blocks)[tmp_ind] for i in block]
        yield df.iloc[ind[0:N_df],:]



def TCA_calc(df, ref_ind=2):

    """
    Calculates triple collocation based squared correlations against the unknown truth, SCALED error standard
    deviations, and scaling coefficients for a given data frame.

    Parameters
    ----------
    df : pd.DataFrame
        Input Dataframe
    ref_ind : int
        column index of reference data set for estimating scaling coefficients

    """

    cov = df.dropna().cov().values

    ind = (0, 1, 2, 0, 1, 2)
    no_ref_ind = np.where(np.arange(3) != ref_ind)[0]

    snr = np.array([np.abs(((cov[i, i] * cov[ind[i + 1], ind[i + 2]]) /
                            (cov[i, ind[i + 1]] * cov[i, ind[i + 2]]) - 1)) ** (-1)
                    for i in np.arange(3)])

    R2 = 1. / (1 + snr**(-1))

    err_var = np.array([np.abs(cov[i, i] - (cov[i, ind[i + 1]] * cov[i, ind[i + 2]]) / cov[ind[i + 1], ind[i + 2]])
                        for i in np.arange(3)])

    beta = np.abs(np.array([cov[ref_ind, no_ref_ind[no_ref_ind != i][0]] / cov[i, no_ref_ind[no_ref_ind != i][0]]
                            if i != ref_ind else 1 for i in np.arange(3)]))

    return R2, np.sqrt(err_var) * beta, beta


def TCA(df, ref_ind=2, alpha=0.95, bootstraps=1000):
    """
    Calculates triple collocation based squared correlations against the unknown truth, SCALED error standard
    deviations, and scaling coefficients for a given data frame.

    Parameters
    ----------
    df : pd.DataFrame
        Input Dataframe
    ref_ind : int
        column index of reference data set for estimating scaling coefficients
    alpha : float
        Confidence level
    bootstraps: int
        Number of bootstrap resamples that should be calculated

    """

    # Initialize bootstrap generator to draw resamples from
    bss = bootstrap(df)

    # Create data frame to store results in p = direct TCA estimates, m = median of the bootstrap sampling distribution.
    res = pd.DataFrame(columns=df.columns.values,
                       index=[met + mod for met in ['r2_', 'ubRMSE_', 'beta_'] for mod in ['p','l','m','u']])

    # Calculate TCA metrics from the original sample
    res.loc['r2_p',:], res.loc['ubRMSE_p',:], res.loc['beta_p',:] = TCA_calc(df, ref_ind=ref_ind)

    arr = np.full((3, 3, bootstraps), np.nan)

    ci_l = (1 - alpha) / 2. * 100
    ci_u = 100 - ci_l

    # Calculate TCA metrics from the bootstrap samples
    for i in np.arange(bootstraps):
        arr[:,0,i], arr[:,1,i], arr[:,2,i] = TCA_calc(next(bss), ref_ind=ref_ind)

    ci = np.nanpercentile(arr, [ci_l, 50, ci_u], axis=2)

    res.loc[['r2_l','r2_m','r2_u'], :] = ci[:,:,0]
    res.loc[['ubRMSE_l','ubRMSE_m','ubRMSE_u'], :] = ci[:,:,1]
    res.loc[['beta_l','beta_m','beta_u'], :] = ci[:,:,2]

    return res


def bias(df, dropna=True, alpha=0.95, flatten=True, n_corr=None):
    """"
    Calculates temporal mean biases and its confidence intervals based on Student's t-distribution,
    both with and without auto-correlation corrected sample size.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame whose k columns will be correlated
    dropna : boolean
        If false, temporal matching (dropna-based) will be done for each column-combination individually
    alpha : float [0,1]
        Confidence level for the confidence intervals
    flatten : boolean
        If set, results are returned as pd.Series in case df only holds 2 columns
    n_corr : xr.DataArray (as returned by the method itself)
        Optionally: auto-correlation-corrected sample size to speed up calculation

    Returns
    -------
    res : xr.DataArray
        (k x k x 7) Data Array holding the following statistics for each data set combination of df:
        bias : Temporal mean bias
        n, n_corr : original and auto-correlation corrected sample size
        CI_l, CI_l_corr, CI_u, CI_u_corr : lower and upper confidence levels with and without sample size correction

    res : pd.Series (if flatten is True and df contains only two columns)
        Series holding the above described statistics for the two input data sets.

    """

    if not isinstance(df,pd.DataFrame):
        print('Error: Input is no pd.DataFrame.')
        return None

    if dropna is True:
        df.dropna(inplace=True)

    df.sort_index(inplace=True)

    cols = df.columns.values

    stats = ['bias', 'n', 'CI_l', 'CI_u', 'n_corr', 'CI_l_corr', 'CI_u_corr']
    dummy = np.full((len(cols),len(cols),len(stats)),np.nan)

    res = xr.DataArray(dummy, dims=['ds1','ds2','stats'], coords={'ds1':cols,'ds2':cols,'stats':stats})

    for ds1 in cols:
        for ds2 in cols:
            if ds1 == ds2:
                continue

            # get sample size
            tmpdf = df[[ds1,ds2]].dropna()
            n = len(tmpdf)
            res.loc[ds1, ds2, 'n'] = n
            res.loc[ds2, ds1, 'n'] = n
            if n < 5:
                continue

            # Calculate bias & ubRMSD
            diff = tmpdf[ds1].values - tmpdf[ds2].values
            bias = diff.mean()
            ubRMSD = diff.std(ddof=1)

            # Calculate confidence intervals
            t_l, t_u = t.interval(alpha, n-1)
            CI_l = bias + t_l * ubRMSD / np.sqrt(n)
            CI_u = bias + t_u * ubRMSD / np.sqrt(n)

            res.loc[ds1, ds2, 'bias'] = bias
            res.loc[ds1, ds2, 'CI_l'] = CI_l
            res.loc[ds1, ds2, 'CI_u'] = CI_u

            if n_corr is not None:
                n_corr_ds = n_corr.loc[ds1, ds2].item()
            else:
                n_corr_ds = correct_n(tmpdf)
            res.loc[ds1, ds2, 'n_corr'] = n_corr_ds
            res.loc[ds2, ds1, 'n_corr'] = n_corr_ds
            if n_corr_ds < 2:
                continue

            # Confidence intervals with corrected sample size
            t_l, t_u = t.interval(alpha, n_corr_ds - 1)
            CI_l = bias + t_l * ubRMSD / np.sqrt(n_corr_ds)
            CI_u = bias + t_u * ubRMSD / np.sqrt(n_corr_ds)

            res.loc[ds1, ds2, 'CI_l_corr'] = CI_l
            res.loc[ds1, ds2, 'CI_u_corr'] = CI_u

    if flatten is True:
        if len(cols) == 2:
            res = pd.Series(res.loc[cols[0], cols[1], :], index=stats, dtype='float32')

    return res


def ubRMSD(df, dropna=True, alpha=0.95, flatten=True, n_corr=None):
    """"
    Calculates the unbiased Root-Mean-Square-Difference and its confidence intervals based on the chi-distribution,
    both with and without auto-correlation corrected sample size.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame whose k columns will be correlated
    dropna : boolean
        If false, temporal matching (dropna-based) will be done for each column-combination individually
    alpha : float [0,1]
        Confidence level for the confidence intervals
    flatten : boolean
        If set, results are returned as pd.Series in case df only holds 2 columns
    n_corr : xr.DataArray (as returned by the method itself)
        Optionally: auto-correlation-corrected sample size to speed up calculation

    Returns
    -------
    res : xr.DataArray
        (k x k x 7) Data Array holding the following statistics for each data set combination of df:
        ubRMSD : unbiased Root-Mean-Square-Difference
        n, n_corr : original and auto-correlation corrected sample size
        CI_l, CI_l_corr, CI_u, CI_u_corr : lower and upper confidence levels with and without sample size correction

    res : pd.Series (if flatten is True and df contains only two columns)
        Series holding the above described statistics for the two input data sets.

    """

    if not isinstance(df,pd.DataFrame):
        print('Error: Input is no pd.DataFrame.')
        return None

    if dropna is True:
        df.dropna(inplace=True)

    df.sort_index(inplace=True)

    cols = df.columns.values

    stats = ['ubRMSD', 'n', 'CI_l', 'CI_u', 'n_corr', 'CI_l_corr', 'CI_u_corr']
    dummy = np.full((len(cols),len(cols),len(stats)),np.nan)

    res = xr.DataArray(dummy, dims=['ds1','ds2','stats'], coords={'ds1':cols,'ds2':cols,'stats':stats})

    for ds1 in cols:
        for ds2 in cols:
            if ds1 == ds2:
                continue

            # get sample size
            tmpdf = df[[ds1,ds2]].dropna()
            n = len(tmpdf)
            res.loc[ds1, ds2, 'n'] = n
            res.loc[ds2, ds1, 'n'] = n
            if n < 5:
                continue

            # Calculate bias & ubRMSD
            diff = tmpdf[ds1].values - tmpdf[ds2].values
            ubRMSD = diff.std(ddof=1)

            # Calculate confidence intervals
            chi_l, chi_u = chi.interval(alpha, n-1)
            CI_l = ubRMSD * np.sqrt(n-1) / chi_u
            CI_u = ubRMSD * np.sqrt(n-1) / chi_l

            res.loc[ds1, ds2, 'ubRMSD'] = ubRMSD
            res.loc[ds1, ds2, 'CI_l'] = CI_l
            res.loc[ds1, ds2, 'CI_u'] = CI_u

            if n_corr is not None:
                n_corr_ds = n_corr.loc[ds1, ds2].item()
            else:
                n_corr_ds = correct_n(tmpdf)

            res.loc[ds1, ds2, 'n_corr'] = n_corr_ds
            res.loc[ds2, ds1, 'n_corr'] = n_corr_ds
            if n_corr_ds < 5:
                continue

            # Confidence intervals with corrected sample size
            chi_l, chi_u = chi.interval(alpha, n_corr_ds - 1)
            CI_l = ubRMSD * np.sqrt(n_corr_ds - 1) / chi_u
            CI_u = ubRMSD * np.sqrt(n_corr_ds - 1) / chi_l

            res.loc[ds1, ds2, 'CI_l_corr'] = CI_l
            res.loc[ds1, ds2, 'CI_u_corr'] = CI_u

    if flatten is True:
        if len(cols) == 2:
            res = pd.Series(res.loc[cols[0], cols[1], :], index=stats, dtype='float32')

    return res


def Pearson_R(df, dropna=True, alpha=0.95, flatten=True, n_corr=None):
    """"
    Calculates the Pearson correlation coefficient and its confidence intervals based on
    Fischer's z-transformation, both with and without auto-correlation corrected sample size.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame whose k columns will be correlated
    dropna : boolean
        If false, temporal matching (dropna-based) will be done for each column-combination individually
    alpha : float [0,1]
        Confidence level for the confidence intervals
    flatten : boolean
        If set, results are returned as pd.Series in case df only holds 2 columns
    n_corr : xr.DataArray (as returned by the method itself)
        Optionally: auto-correlation-corrected sample size to speed up calculation

    Returns
    -------
    res : xr.DataArray
        (k x k x 8) Data Array holding the following statistics for each data set combination of df:
        R, p : Pearson correlation coefficient and corresponding significance level
        n, n_corr : original and auto-correlation corrected sample size
        CI_l, CI_l_corr, CI_u, CI_u_corr : lower and upper confidence levels with and without sample size correction

    res : pd.Series (if flatten is True and df contains only two columns)
        Series holding the above described statistics for the two input data sets.

    """

    if not isinstance(df,pd.DataFrame):
        print('Error: Input is no pd.DataFrame.')
        return None

    if dropna is True:
        df.dropna(inplace=True)

    df.sort_index(inplace=True)

    cols = df.columns.values

    stats = ['R', 'p', 'n', 'CI_l', 'CI_u', 'n_corr', 'CI_l_corr', 'CI_u_corr']
    dummy = np.full((len(cols),len(cols),len(stats)),np.nan)

    res = xr.DataArray(dummy, dims=['ds1','ds2','stats'], coords={'ds1':cols,'ds2':cols,'stats':stats})

    for ds1 in cols:
        for ds2 in cols:
            if ds1 == ds2:
                continue

            # get sample size
            tmpdf = df[[ds1,ds2]].dropna()
            n = len(tmpdf)
            res.loc[ds1, ds2, 'n'] = n
            res.loc[ds2, ds1, 'n'] = n
            if n < 5:
                continue

            # Calculate correlation and -significance
            R, p = pearsonr(tmpdf[ds1].values,tmpdf[ds2].values)

            # Fisher's z-transform for confidence intervals
            z = 0.5 * np.log((1+R)/(1-R))
            z_l, z_u = norm.interval(alpha, loc=z, scale=(n - 3) ** (-0.5))
            CI_l = (np.exp(2*z_l) - 1) / (np.exp(2*z_l) + 1)
            CI_u = (np.exp(2*z_u) - 1) / (np.exp(2*z_u) + 1)

            res.loc[ds1, ds2, 'R'] = R
            res.loc[ds1, ds2, 'p'] = p
            res.loc[ds1, ds2, 'CI_l'] = CI_l
            res.loc[ds1, ds2, 'CI_u'] = CI_u

            if n_corr is not None:
                n_corr_ds = n_corr.loc[ds1, ds2].item()
            else:
                n_corr_ds = correct_n(tmpdf)

            res.loc[ds1, ds2, 'n_corr'] = n_corr_ds
            res.loc[ds2, ds1, 'n_corr'] = n_corr_ds
            if n_corr_ds < 5:
                continue

            # Confidence intervals with corrected sample size
            z_l, z_u = norm.interval(alpha, loc=z, scale=(n_corr_ds - 3) ** (-0.5))
            CI_l = (np.exp(2 * z_l) - 1) / (np.exp(2 * z_l) + 1)
            CI_u = (np.exp(2 * z_u) - 1) / (np.exp(2 * z_u) + 1)

            res.loc[ds1, ds2, 'CI_l_corr'] = CI_l
            res.loc[ds1, ds2, 'CI_u_corr'] = CI_u

    if flatten is True:
        if len(cols) == 2:
            res = pd.Series(res.loc[cols[0], cols[1], :], index=stats, dtype='float32')

    return res
