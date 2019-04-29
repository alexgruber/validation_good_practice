

import numpy as np
import pandas as pd

from netCDF4 import Dataset, num2date

from validation_good_practice.ancillary.paths import Paths

class HSAF_io(object):

    def __init__(self, version='h113', ext='h114'):

        paths = Paths()

        self.data_path = paths.ascat / version
        self.version = version.upper()

        grid = Dataset(paths.ascat / 'warp5_grid' / 'TUW_WARP5_grid_info_2_2.nc')
        self.gpis = grid['gpi'][:][grid['land_flag'][:]==1]
        self.cells = grid['cell'][:][grid['land_flag'][:]==1]
        grid.close()

        self.loaded_cell = None
        self.fid = None

        if ext is not None:
            self.ext = HSAF_io(version=ext, ext=None)
        else:
            self.ext = None

    def load(self, cell):

        fname = self.data_path / self.version + ('_%04i.nc' % cell)
        if not fname.exists():
            print('File not found: ' + fname)
            return False

        try:
            if self.fid is not None:
                self.fid.close()
            self.fid = Dataset(fname)
        except:
            print('Corrupted cell: %i' % cell)
            return False

        self.loaded_cell = cell

        return True

    def read(self, gpi):

        if not gpi in self.gpis:
            print('GPI not found')
            return None

        cell = self.cells[self.gpis==gpi][0]

        if self.loaded_cell != cell:
            loaded = self.load(cell)
            if loaded is False:
                return None

        loc = np.where(self.fid['location_id'][:] == gpi)[0][0]
        start = self.fid['row_size'][0:loc].sum()
        end = start + self.fid['row_size'][loc]

        corr_flag = self.fid['corr_flag'][start:end]
        conf_flag = self.fid['conf_flag'][start:end]
        proc_flag = self.fid['proc_flag'][start:end]
        ssf = self.fid['ssf'][start:end]
        ind_valid = ((corr_flag==0)|(corr_flag==4)) & (conf_flag == 0) & (proc_flag == 0) & (ssf == 1)

        if len(np.where(ind_valid)[0]) == 0:
            print('No valid data for gpi %i' % gpi)
            return None

        sm = self.fid['sm'][start:end][ind_valid]
        time = num2date(self.fid['time'][start:end][ind_valid],
                        units=self.fid['time'].units)

        ts = pd.Series(sm, index=time)
        ts = ts.groupby(ts.index).mean()

        if self.ext is not None:
            ts_ext = self.ext.read(gpi)
            ts = pd.concat((ts,ts_ext))

        return ts

    def close(self):
        if self.fid is not None:
            self.fid.close()
        if self.ext is not None:
            self.ext.close()

def reshuffle_ascat():

    paths = Paths()

    # get a list of all CONUS gpis
    gpi_lut = pd.read_csv(paths.lut, index_col=0)[['ascat_gpi']]

    io = HSAF_io()

    # Store NN of EASE2 grid points in CSV files
    dir_out = paths.ascat / 'resampled'
    for gpi, lut in gpi_lut.iterrows():
        Ser = io.read(lut['ascat_gpi'])
        if Ser is not None:
            Ser = Ser['2015-01-01':'2018-12-31']
            if len(Ser) > 10:
                Ser.index = Ser.index.round('min')
                fname = dir_out / ('%i.csv' % gpi)
                Ser.to_csv(fname, float_format='%.4f')


if __name__=='__main__':
    reshuffle_ascat()

