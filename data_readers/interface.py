
import numpy as np
import pandas as pd

from pytesmo.temporal_matching import df_match

from validation_good_practice.ancillary.paths import Paths

class reader():
    """
    Class for reading and collocating data from different data sources

    Parameters
    ----------
    sensors : list of strings
        A list of sensors that should be read & collocated.
        Per default, all available data will be read (currently, ASCAT, SMOS, SMAP, MERRA-2 and ISMN stations)

    """

    def __init__(self, sensors=None):

        if sensors is None:
            self.sensors = ['ASCAT','SMOS','SMAP','MERRA2', 'ISMN']
        else:
            self.sensors = sensors

        self.root = Paths().data_root


    def read(self, gpi, calc_anom_st=True, calc_anom_lt=True):
        """
        Method for reading data over a particular grid cell.

        Parameters
        ----------
        gpi : int
            EASEv2 grid cell for which data should be read.
            Data first needs to be resampled and stored as .csv files (per grid cell) already!
            (see the reader routines of the individual data sets).
        calc_anom_st : boolean
            if set, short-term anomalies are calculated
        calc_anom_lt : boolean
            if set, long-term anomalies are calculated

        Returns
        ----------
        mode: list of strings
            list of names for the different types of collocated data sets

            Possible options: 'grid_abs', 'grid_anom_st', 'grid_anom_lt', 'ismn_abs', 'ismn_anom_st', 'ismn_anom_lt'

            grid_* refers to collocated data sets WITHOUT ISMN station data
            ismn_* refers  to collocated data sets INCLUDING ISMN station data (if available)
            *_abs refers to absolute soil moisture time series
            *_anom_st refers to short-term anomalies
            *_anom_lt refers to long-term anomalies

        dfs: list of pd.DataFrames
            collocated data sets with and without ismn station data for absolute time series as well as short-term and
            long-term anomalies

        """

        # Read all individual data sets and store them in a list of pd.Series
        Ser_list = list()
        for sensor in np.atleast_1d(self.sensors):
            fname = self.root / sensor / 'timeseries' / ('%i.csv' % gpi)
            if fname.exists():
                Ser = pd.read_csv(fname, index_col=0, header=None, names=[sensor])
                Ser.index = pd.DatetimeIndex(Ser.index)
                if sensor == 'ASCAT':
                    Ser /= 100.
            else:
                Ser = pd.Series(name=sensor)
            Ser_list.append(Ser)

        # convert list of pd.Series into pd.DataFrames with and without ISMN data.
        if 'ISMN' in self.sensors:
            df_ismn_abs = pd.concat(Ser_list, axis='columns')
            df_grid_abs = df_ismn_abs.copy().drop('ISMN', axis='columns')
        else:
            df_grid_abs = pd.concat(Ser_list, axis='columns')

        # Collocate data without ISMN data
        df_grid_abs = collocate(df_grid_abs)

        # store resulting data sets with and without ISMN data and optionally anomalies into a list of data frames.
        mode = ['grid_abs']
        dfs = [df_grid_abs]

        if calc_anom_st is True:
            mode.append('grid_anom_st')
            dfs.append(calc_anomaly(df_grid_abs))

        if calc_anom_lt is True:
            mode.append('grid_anom_lt')
            dfs.append(calc_anomaly(df_grid_abs, longterm=True))

        if 'ISMN' in self.sensors:
            mode.append('ismn_abs')
            if len(df_ismn_abs['ISMN'].dropna()) > 10: # check if at least a few ISMN measurements are available.
                df_ismn_abs = collocate(df_ismn_abs)
            else:
                df_ismn_abs = None
            dfs.append(df_ismn_abs)

            if calc_anom_st is True:
                mode.append('ismn_anom_st')
                dfs.append(calc_anomaly(df_ismn_abs))

            if calc_anom_lt is True:
                mode.append('ismn_anom_lt')
                dfs.append(calc_anomaly(df_ismn_abs, longterm=True))

        return mode, dfs


def calc_anomaly(in_df, longterm=False):
    """
    Calculates anomalies for each column in a pd.DataFrame

    Parameters
    ----------
    in_df : pd.DataFrame
        Input data frame
    longterm : Boolean
        If set, long-term anomalies will be calculated. If not (default) short-term anomalies will be calculated.

    """

    if in_df is None:
        return None

    df = in_df.copy()

    # Calculate anomalies for each column
    for col in df:
        df[col] = calc_anom(df[col], longterm=longterm)

    return df


def calc_anom(Ser, longterm=False):
    """
    Calculates anomalies for a pd.Series

    Parameters
    ----------
    Ser : pd.Series
        Input Series
    longterm : Boolean
        If set, long-term anomalies will be calculated. If not (default) short-term anomalies will be calculated.

    """

    xSer = Ser.dropna().copy()
    if len(xSer) == 0:
        return xSer

    doys = xSer.index.dayofyear.values
    doys[xSer.index.is_leap_year & (doys > 59)] -= 1 # in leap years, subtract 1 for all DOYs after Feb 28
    climSer = pd.Series(index=xSer.index,name=xSer.name)

    if longterm is True:
        climSer[:] = calc_clim(xSer)[doys]
    else:
        years = xSer.index.year
        for yr in np.unique(years):
            clim = calc_clim(xSer[years == yr])
            climSer[years == yr] = clim[doys[years == yr]].values

    return xSer - climSer


def calc_clim(Ser, window_size=35):
    """
    Calculates the climatology for a pd.Series using a moving-average window

    Parameters
    ----------
    Ser : pd.Series
        Input Series
    window_size : int
        size [days] of the moving-average window

    """

    xSer = Ser.dropna().copy()
    doys = xSer.index.dayofyear.values

    # in leap years, subtract 1 for all DOYs after Feb 28
    doys[xSer.index.is_leap_year & (doys > 59)] -= 1

    clim_doys = np.arange(365) + 1
    clim = pd.Series(index=clim_doys)

    for doy in clim_doys:
        # Avoid artifacts at start/end of year
        tmp_doys = doys.copy()
        if doy < window_size / 2.:
            tmp_doys[tmp_doys > 365 - (np.ceil(window_size / 2.) - doy)] -= 365
        if doy > 365 - (window_size / 2. - 1):
            tmp_doys[tmp_doys < np.ceil(window_size / 2.) - (365 - doy)] += 365

        clim[doy] = xSer[(tmp_doys >= doy - np.floor(window_size / 2.)) & \
                         (tmp_doys <= doy + np.floor(window_size / 2.))].values.mean()

    return clim


def collocate(df):
    """
    Collocates the columns of a pd.DataFrame. Data points are resampled to reference time stemps with 24 hr distance,
    which are optimized to maximize the number of matches.

    Parameters
    ----------
    df : pd.DataFrame
        Input Dataframe

    """

    res_df = pd.DataFrame(columns=df.columns.values)

    # Test each hour of the day as potential reference time step to optimize the number of collocated data points
    for d in np.arange(24):

        # Create reference time steps for the respective reference hour of the day of this iteration
        ref_df = pd.DataFrame(
            index=pd.date_range(df.index.min().date(), df.index.max().date()) + pd.Timedelta(d, 'h'))

        # Find the NN to the reference time steps for each data set
        args = [df[col].dropna() for col in df]
        matched = df_match(ref_df, *args, window=0.5)
        if len(df.columns) ==1:
            ref_df[df.columns.values[0]] = matched[df.columns.values[0]]
        else:
            for i, col in enumerate(df):
                ref_df[col] = matched[i][col]
        ref_df.dropna(inplace=True)

        # Check if collocation at this hour gave more temporal matches than collocation at the previous hour
        if len(ref_df) > len(res_df):
            res_df = ref_df.copy()

    return res_df
