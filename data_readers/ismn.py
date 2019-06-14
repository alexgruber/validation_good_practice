
import numpy as np
import pandas as pd

from ismn.interface import ISMN_Interface

from validation_good_practice.ancillary.grid import EASE2
from validation_good_practice.ancillary.paths import Paths

def generate_station_list():
    """ This routine generates a list of available ISMN stations and the EASEv2 grid point they are located in. """

    paths = Paths()

    io = ISMN_Interface(paths.ismn / 'downloaded' / 'CONUS_20100101_20190101')

    # get metadata indices of all stations that measure soil moisture within the first 10 cm
    idx = io.get_dataset_ids('soil moisture', min_depth=0.0, max_depth=0.1)
    df = pd.DataFrame({'network': io.metadata[idx]['network'],
                       'station': io.metadata[idx]['station'],
                       'lat': io.metadata[idx]['latitude'],
                       'lon': io.metadata[idx]['longitude'],
                       'ease2_gpi': np.zeros(len(idx)).astype('int')}, index=idx)

    # merge indices for stations that have multiple sensors within the first 10 cm
    duplicate_idx = df.groupby(df.columns.tolist()).apply(lambda x: '-'.join(['%i'% i for i in x.index])).values
    df.drop_duplicates(inplace=True)
    df.index = duplicate_idx

    # create EASEv2 grid domain
    grid = EASE2()
    lons, lats = np.meshgrid(grid.ease_lons, grid.ease_lats)
    lons = lons.flatten()
    lats = lats.flatten()

    # find EASEv2 grid points in which the individual stations are located
    for i, (idx, data) in enumerate(df.iterrows()):
        print('%i / %i' % (i, len(df)))
        r = (lons - data.lon) ** 2 + (lats - data.lat) ** 2
        df.loc[idx, 'ease2_gpi'] = np.where((r - r.min()) < 0.0001)[0][0]

    df.to_csv(paths.ismn / 'station_list.csv')


def resample_ismn():
    """
    This resamples ISMN data onto the EASE2 grid and stores data for each grid cell into .csv files.
    If single grid cells contain multiple stations, they are averaged.

    A grid look-up table needs to be created first (method: ancillary.grid.create_lut).

    """

    paths = Paths()

    io = ISMN_Interface(paths.ismn / 'downloaded' / 'CONUS_20100101_20190101')

    # get all stations / sensors for each grid cell.
    lut = pd.read_csv(paths.ismn / 'station_list.csv',index_col=0)
    lut = lut.groupby('ease2_gpi').apply(lambda x: '-'.join([i for i in x.index]))

    dir_out = paths.ismn / 'timeseries'

    for cnt, (gpi, indices) in enumerate(lut.iteritems()):
        print('%i / %i' % (cnt, len(lut)))

        fname = dir_out / ('%i.csv' % gpi)

        idx = indices.split('-')

        # Only one station within grid cell
        if len(idx) == 1:
            try:
                ts = io.read_ts(int(idx[0]))
                ts = ts[ts['soil moisture_flag'] == 'G']['soil moisture'] # Get only "good" data based on ISMN QC
                ts.tz_convert(None).to_csv(fname, float_format='%.4f')
            except:
                print('Corrupt file: ' + io.metadata[int(idx[0])]['filename'])

        # Multiple stations within grid cell
        else:
            df = []
            for i in idx:
                try:
                    ts = io.read_ts(int(i))
                    df += [ts[ts['soil moisture_flag'] == 'G']['soil moisture']] # Get only "good" data based on ISMN QC
                except:
                    print('Corrupt file: ' + io.metadata[int(i)]['filename'])
            if len(df) == 0:
                continue

            df = pd.concat(df, axis=1)
            df.columns = np.arange(len(df.columns))

            # match temporal mean and standard deviation to those of the station with the maximum temporal coverage
            n = np.array([len(df[i].dropna()) for i in df])
            ref = np.where(n==n.max())[0][0]
            for col in df:
                if col != ref:
                    df[col] = (df[col] - df[col].mean())/df[col].std() * df[ref].std() + df[ref].mean()

            # Average measurements of all stations
            df.mean(axis='columns').tz_convert(None).to_csv(fname, float_format='%.4f')


if __name__=='__main__':

    # generate_station_list()
    resample_timeseries()


