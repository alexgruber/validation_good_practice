
import h5py

import numpy as np
import pandas as pd

from validation_good_practice.ancillary.paths import Paths

def reformat_smap():
    """
    This extracts raw SMAP EASEv2 data and stores it into .csv files for later processing.

    A grid look-up table needs to be created first (method: ancillary.grid.create_lut).

    """

    paths = Paths()

    # generate idx. array to map ease col/row to gpi
    n_row = 406; n_col = 964
    idx_arr = np.arange(n_row*n_col,dtype='int64').reshape((n_row,n_col))

    # get a list of all CONUS gpis
    ease_gpis = pd.read_csv(paths.lut, index_col=0).index.values

    # Collect orbit file list and extract date info from file name
    fdir = paths.smap_raw
    files = sorted(fdir.glob('*'))
    dates = pd.to_datetime([str(f)[-29:-14] for f in files]).round('min')

    # Array with ALL possible dates and ALL CONUS gpis
    res_arr = np.full((len(dates),len(ease_gpis)), np.nan)

    # Fill in result array from orbit files
    for i, f in enumerate(files):
        print("%i / %i" % (i, len(files)))

        tmp = h5py.File(fdir / f)
        row = tmp['Soil_Moisture_Retrieval_Data']['EASE_row_index'][:]
        col = tmp['Soil_Moisture_Retrieval_Data']['EASE_column_index'][:]
        idx = idx_arr[row,col]

        # Check for valid data within orbit files
        for res_ind, gpi in enumerate(ease_gpis):
            sm_ind = np.where(idx == gpi)[0]
            if len(sm_ind) > 0:
                qf = tmp['Soil_Moisture_Retrieval_Data']['retrieval_qual_flag'][sm_ind[0]]
                if (qf == 0)|(qf == 8):
                    res_arr[i, res_ind] = tmp['Soil_Moisture_Retrieval_Data']['soil_moisture'][sm_ind[0]]

        tmp.close()

    # Write out valid time series of all CONIS GPIS into separate .csv files
    dir_out = paths.smap / 'timeseries'
    if not dir_out.exists():
        dir_out.mkdir()

    for i, gpi in enumerate(ease_gpis):
        Ser = pd.Series(res_arr[:,i],index=dates).dropna()
        if len(Ser) > 0:
            Ser = Ser.groupby(Ser.index).last() # Make sure that no time duplicates exist!
            fname = dir_out / ('%i.csv' % gpi)
            Ser.to_csv(fname,float_format='%.4f')
