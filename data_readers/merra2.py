
import numpy as np
import pandas as pd
import xarray as xr

from validation_good_practice.ancillary.paths import Paths

def resample_merra2(part=1, parts=1):
    """
    This resamples MERRA-2 data from the MERRA grid onto the EASE2 grid and stores data for each grid cell into .csv files.

    A grid look-up table needs to be created first (method: ancillary.grid.create_lut).

    Parameters
    ----------
    part : int
        Data subset to be processed - Data can be resampled in subsets for parallelization to speed-up the processing.
    parts : int
        Number of parts in which to split the data for parallel processing.
        Per default, all data are resampled at once.

    """

    paths = Paths()

    dir_out = paths.merra2 / 'timeseries'

    path = paths.merra2 / 'raw' / '2015-2018'
    files = np.array(sorted(path.glob('*')))
    ds = xr.open_mfdataset(files)
    lats = ds.lat.values
    lons = ds.lon.values
    dates = pd.to_datetime(ds.time.values)

    # get a list of all CONUS gpis
    gpi_lut = pd.read_csv(paths.lut, index_col=0)[['merra2_lon','merra2_lat']]

    # split domain for parallelization
    subs = (np.arange(parts + 1) * len(gpi_lut) / parts).astype('int')
    subs[-1] = len(gpi_lut)
    start = subs[part - 1]
    end = subs[part]

    gpi_lut = gpi_lut.iloc[start:end,:]

    # find and write all EASE2 NN grid points
    for i, (gpi, lut) in enumerate(gpi_lut.iterrows()):
        print("%i / %i" % (i, len(gpi_lut)))

        ind_lat = np.where(lats == lut['merra2_lat'])[0][0]
        ind_lon = np.where(lons == lut['merra2_lon'])[0][0]

        ts = ds['TSOIL1'][:, ind_lat, ind_lon] - 273.15
        swe = ds['SNOMAS'][:, ind_lat, ind_lon]
        ind_valid = ((ts>=4)&(swe==0)).values

        Ser = pd.Series(ds['SFMC'][ind_valid, ind_lat, ind_lon].values, index=dates[ind_valid])
        fname = dir_out / ('%i.csv' % gpi)
        Ser.to_csv(fname, float_format='%.4f')

