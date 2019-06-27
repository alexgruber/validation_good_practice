
import os

import numpy as np
import pandas as pd

from netCDF4 import Dataset

from validation_good_practice.ancillary.paths import Paths

def resample_smos():
    """
    This resamples SMOS data from the SMOS grid onto the EASE2 grid and stores data for each grid cell into .csv files.

    A grid look-up table needs to be created first (method: ancillary.grid.create_lut).

    """

    paths = Paths()

    # Collect all .nc files
    path = paths.smos_raw
    nc_files = sorted(path.glob('**/*.nc'))

    # Get time stamp as the mean of start-of-orbit and end-of-orbit
    sdate = pd.to_datetime([str(f)[-44:-29] for f in nc_files])
    edate = pd.to_datetime([str(f)[-28:-13] for f in nc_files])
    dates = (sdate + (edate - sdate) / 2.).round('min')

    # get a list of all CONUS gpis
    gpi_lut = pd.read_csv(paths.lut, index_col=0)['smos_gpi']
    ease_gpis = gpi_lut.index.values

    # Array with ALL possible dates and ALL CONUS gpis
    res_arr = np.full((len(dates),len(ease_gpis)), np.nan)

    # Fill in result array from orbit files
    for i, f in enumerate(nc_files):
        print("%i / %i" % (i, len(nc_files)))

        ds = Dataset(f)
        smos_gpis = ds.variables['Grid_Point_ID'][:]

        # Check for valid data within orbit files
        for res_ind, ease_gpi in enumerate(ease_gpis):
            smos_ind = np.where(smos_gpis == gpi_lut.loc[ease_gpi])[0]
            if len(smos_ind) > 0:

                # extract soil moisture data
                sm = float(ds.variables['Soil_Moisture'][smos_ind])
                if np.isnan(sm) | (sm < 0.):
                    continue

                # Mask for RFI and Chi-2 flag
                rfi = float(ds.variables['RFI_Prob'][smos_ind])
                chi_2_p = float(ds.variables['Chi_2_P'][smos_ind])
                valid = (rfi < 0.1) & (chi_2_p > 0.05)

                if valid:
                    res_arr[i, res_ind] = sm

        ds.close()

    # Write out valid time series of all CONIS GPIS into separate .csv files
    dir_out = paths.smos / 'timeseries'
    if not dir_out.exists():
        dir_out.mkdir()

    for i, gpi in enumerate(ease_gpis):
        Ser = pd.Series(res_arr[:,i],index=dates).dropna()
        if len(Ser) > 0:
            Ser = Ser.groupby(Ser.index).last() # Make sure that no time duplicates exist!
            fname = dir_out / ('%i.csv' % gpi)
            Ser.to_csv(fname,float_format='%.4f')
