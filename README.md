# Validation Good Practice Example

This package holds the source code that has been used to create the results shown in Appendix A of the soil moisture validation good practice paper:
 
"Validation practices for satellite soil moisture products: What are (the) errors?"

Included are all processing steps starting from the downloaded L2 soil moisture data as described in the paper and in the readme of validation_good_practice.data readers.

The following external python packages are required, which can  be installed using conda or pip:

`numpy, pandas, xarray, scipy, pytesmo, netCDF4, h5py, dask, matplotlib, Basemap, ismn`

Results in the paper can be reproduced as follows:

## Path definition

All necessary data and result paths are defined in routine validation_good_practice.ancillary.paths.py. These obviously need to be adjusted to the machine where data are processed.

## Pre-processing

Pre-processing includes the spatial resampling of the downloaded L2 data:

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

## Metric calculation

The routine validation_good_pratice.interface.py contains a single routine that calculates performance metrics and confidence intervals for all data sets (i.e. absolute time series and anomalies with and without collocation with ISMN data):

```
from validation_good_practice.validate import run

run()
```

Notice that the routine is set up to split the study domain into subsets (per default 30) for parallel processing. This needs to be adjusted based on the number of kernels available. 

## Visualization

Plotting routines are contained in validation_good_pratice.plots.py. All plots that are shown in the paper can be generated through:

```
from validation_good_practice.plots import generate_plots

generate_plots()
```

### MATLAB

Validation metrics are also available in MATLAB and can be found at:

`validation_good_practice._matlab_`

## Contact
For any questions or feedback please contact alexander.gruber@kuleuven.be


