
import numpy as np
import pandas as pd

from netCDF4 import Dataset, num2date

from validation_good_practice.ancillary.paths import Paths

class HSAF_io(object):
    """
    Class for reading ASCAT soil moisture data as downloaded from H SAF (http://hsaf.meteoam.it/soil-moisture.php)

    Parameters
    ----------
    version : str
        H SAF product version
    ext : str
        data set extension to a particular H SAF data record
        If provided, the H SAF data record and its extension will be read and concatenated into one data frame
    """

    def __init__(self, version='h113', ext='h114'):

        paths = Paths()

        self.data_path = paths.ascat / version
        self.version = version.upper()

        # mask out all non-land grid cells
        grid = Dataset(paths.ascat / 'warp5_grid' / 'TUW_WARP5_grid_info_2_2.nc')
        self.gpis = grid['gpi'][:][grid['land_flag'][:]==1]
        self.cells = grid['cell'][:][grid['land_flag'][:]==1]
        grid.close()

        self.loaded_cell = None
        self.fid = None

        # read data set extension if provided.
        if ext is not None:
            self.ext = HSAF_io(version=ext, ext=None)
        else:
            self.ext = None

    def load(self, cell):
        """
        ASCAT data are stored such that multiple gridpoints are stored within single cell files.
        This method loads a cell file and holds it in the memory to increase reading speed.

        Parameters
        ----------
        cell : int
            number of the cell that should be read
        """

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

        # store information which cell is currently loaded in the memory.
        self.loaded_cell = cell

        return True

    def read(self, gpi):
        """
        Reads the time series over a particular grid point (in the ASCAT DGG grid).

        Parameters
        ----------
        gpi : int
            grid point index to be read
        """

        if not gpi in self.gpis:
            print('GPI not found')
            return None

        # check with cell the GPI is in
        cell = self.cells[self.gpis==gpi][0]

        # check if the cell has been loaded into the memory already
        if self.loaded_cell != cell:
            loaded = self.load(cell)
            if loaded is False:
                return None

        # find gpi within cell file
        loc = np.where(self.fid['location_id'][:] == gpi)[0][0]
        start = self.fid['row_size'][0:loc].sum()
        end = start + self.fid['row_size'][loc]

        # extract ASCAT quality flags and find all valid measurements.
        corr_flag = self.fid['corr_flag'][start:end]
        conf_flag = self.fid['conf_flag'][start:end]
        proc_flag = self.fid['proc_flag'][start:end]
        ssf = self.fid['ssf'][start:end]
        ind_valid = ((corr_flag==0)|(corr_flag==4)) & (conf_flag == 0) & (proc_flag == 0) & (ssf == 1)

        if len(np.where(ind_valid)[0]) == 0:
            print('No valid data for gpi %i' % gpi)
            return None

        # extract soil moisture and date information.
        sm = self.fid['sm'][start:end][ind_valid]
        time = num2date(self.fid['time'][start:end][ind_valid],
                        units=self.fid['time'].units)

        # Occationally there are duplicate time steps. This averages these duplicates.
        ts = pd.Series(sm, index=time)
        ts = ts.groupby(ts.index).mean()

        # Append data set extension if available
        if self.ext is not None:
            ts_ext = self.ext.read(gpi)
            ts = pd.concat((ts,ts_ext))

        return ts

    def close(self):
        """ Cleanup the memory when closing the reader class instance."""

        if self.fid is not None:
            self.fid.close()
        if self.ext is not None:
            self.ext.close()


def resample_ascat():
    """
    This resamples ASCAT data from the DGG grid onto the EASE2 grid and stores data for each grid cell into .csv files.

    A grid look-up table needs to be created first (method: ancillary.grid.create_lut).

    """

    paths = Paths()

    # get a list of all CONUS gpis
    gpi_lut = pd.read_csv(paths.lut, index_col=0)[['ascat_gpi']]

    io = HSAF_io()

    # Store NN of EASE2 grid points into CSV files
    dir_out = paths.ascat / 'resampled'
    for gpi, lut in gpi_lut.iterrows():
        Ser = io.read(lut['ascat_gpi'])
        if Ser is not None:
            Ser = Ser['2015-01-01':'2018-12-31']
            if len(Ser) > 10:
                Ser.index = Ser.index.round('min') # round time steps to full minutes.
                fname = dir_out / ('%i.csv' % gpi)
                Ser.to_csv(fname, float_format='%.4f')
