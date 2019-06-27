
from pathlib import Path

class Paths(object):
    """ This class contains the paths where data are stored and results should be written to."""

    def __init__(self):

        self.result_root = Path('/work/validation_good_practice')

        self.data_root = Path('/data_sets')

        self.lut = self.data_root / 'EASE2_grid' /'grid_lut.csv'

        self.ascat = self.data_root / 'ASCAT'
        self.smos = self.data_root / 'SMOS'
        self.smap = self.data_root / 'SMAP'
        self.merra2 = self.data_root / 'MERRA2'
        self.ismn = self.data_root / 'ISMN'

        # ASCAT raw data should be in self.ascat with the folder names matching the H SAF version number as downloaded
        self.smos_raw = self.smos  / 'raw' / 'MIR_SMUDP2_nc'
        self.smap_raw = self.smos  / 'raw'
        self.merra2_raw = self.merra2 / 'raw' / '2015-2018'
        self.ismn_raw = self.ismn / 'downloaded' / 'CONUS_20100101_20190101'

