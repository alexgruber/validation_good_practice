# DATA READERS

This directory contains routines for reformating, and temporally and spatially collocating various data sets. Currently included are ASCAT, SMOS, SMAP, MERRA-2 and the ISMN.


ASCAT data supported are L2 H113 and H114 data from H SAF (http://hsaf.meteoam.it/soil-moisture.php)

SMOS data supported are L2 soil moisture data from https://smos-diss.eo.esa.int/. Grid information can be found at http://www.cesbio.ups-tlse.fr/SMOS_blog/?tag=dgg

SMAP data supported are SPL2SMP (36 km passive radiometer) data (https://nsidc.org/data/spl2smp/)

MERRA2 data supported are land surface parameters downloaded from https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/data_access/

ISMN data are downloaded from https://ismn.geo.tuwien.ac.at/en/

## Source code description

Each data set has its own reading routine (ascat.py, smos.py, smap.py, merra2.py, ismn.py). The purpose of these routines is to resample all data sets onto the EASE v2 grid and store them into separate .csv files (one for each data set and for each grid cell). 

Note that before doing so, a look-up table between grids and an ismn station list which contains the EASE v2 grid cells that cover the respective stations needs to be created.

```
from validation_good_practice.ancillary.grid import create_lut

from validation_good_practice.data_readers.ascat import resample_ascat
from validation_good_practice.data_readers.smos import resample_smos
from validation_good_practice.data_readers.smap import reformat_smap
from validation_good_practice.data_readers.merra2 import resample_merra2
from validation_good_practice.data_readers.ismn import generate_station_list, resample_ismn

create_lut()
generate_station_list()

resample_ascat()
resample_smos()
reformat_smap()
resample_merra2()
resample_ismn()
```

After data sets have been resampled and stored as .csv files, the reader class within interface.py can be used to read, collocate, and decompose the data sets for individual grid cells. E.g.:

```
from validation_good_practice.data_readers.interface import reader

io = reader()

gpi = 83065
modes, data = io.read(gpi)
```

For more information see the documentation within the individual routines.
