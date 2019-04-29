
import numpy as np
import pandas as pd

from pytesmo.temporal_matching import df_match

from validation_good_practice.ancillary.paths import Paths

class reader():

    def __init__(self, sensors=None):

        if sensors is None:
            self.sensors = ['ASCAT','SMOS','SMAP','MERRA2', 'ISMN']
        else:
            self.sensors = sensors

        self.root = Paths().data_root


    def read(self, gpi, sensors=None):

        if sensors is None:
            sensors = self.sensors
        Ser_list = list()
        for sensor in np.atleast_1d(sensors):
            fname = self.root / sensor / 'timeseries' / ('%i.csv' % gpi)
            if fname.exists():
                Ser = pd.read_csv(fname, index_col=0, header=None, names=[sensor])
                Ser.index = pd.DatetimeIndex(Ser.index)
                if sensor == 'ASCAT':
                    Ser /= 100.
            else:
                Ser = pd.Series(name=sensor)
            Ser_list.append(Ser)

        df_ismn_abs = pd.concat(Ser_list, axis='columns')
        df_grid_abs = df_ismn_abs.copy().drop('ISMN', axis='columns')

        if len(df_ismn_abs['ISMN'].dropna()) > 10:
            df_ismn_abs = collocate(df_ismn_abs)
            df_ismn_anom_st = calc_anomaly(df_ismn_abs)
            df_ismn_anom_lt = calc_anomaly(df_ismn_abs, longterm=True)
        else:
            df_ismn_abs = None
            df_ismn_anom_st = None
            df_ismn_anom_lt = None

        df_grid_abs = collocate(df_grid_abs)
        df_grid_anom_st = calc_anomaly(df_grid_abs)
        df_grid_anom_lt = calc_anomaly(df_grid_abs, longterm=True)

        mode = ['grid_abs', 'grid_anom_st', 'grid_anom_lt', 'ismn_abs', 'ismn_anom_st', 'ismn_anom_lt']
        dfs = df_grid_abs, df_grid_anom_st, df_grid_anom_lt, df_ismn_abs, df_ismn_anom_st, df_ismn_anom_lt

        return mode, dfs


def calc_anomaly(in_df, longterm=False):

    df = in_df.copy()

    for col in df:
        df[col] = calc_anom(df[col], longterm=longterm)

    return df


def calc_anom(Ser, longterm=False):

    xSer = Ser.dropna().copy()
    if len(xSer) == 0:
        return xSer

    doys = xSer.index.dayofyear.values
    doys[xSer.index.is_leap_year & (doys > 59)] -= 1
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

    xSer = Ser.dropna().copy()
    doys = xSer.index.dayofyear.values

    # in leap years, subtract 1 for all days after Feb 28
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

    dts = np.arange(24)

    res_df = pd.DataFrame(columns=df.columns.values)
    for d in dts:
        ref_df = pd.DataFrame(
            index=pd.date_range(df.index.min().date(), df.index.max().date()) + pd.Timedelta(d, 'h'))
        args = [df[col].dropna() for col in df]
        matched = df_match(ref_df, *args, window=0.5)
        if len(df.columns) ==1:
            ref_df[df.columns.values[0]] = matched[df.columns.values[0]]
        else:
            for i, col in enumerate(df):
                ref_df[col] = matched[i][col]
        ref_df.dropna(inplace=True)

        if len(ref_df) > len(res_df):
            res_df = ref_df.copy()

    return res_df


if __name__=='__main__':

    # gpi = 72512
    gpi = 88926

    io = reader()

    df_grid_abs, df_grid_anom_st, df_grid_anom_lt, df_ismn_abs, df_ismn_anom_st, df_ismn_anom_lt = io.read(gpi)

    pass