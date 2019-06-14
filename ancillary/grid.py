
import pyproj

import numpy as np
import pandas as pd

from netCDF4 import Dataset

from validation_good_practice.ancillary.paths import Paths

class EASE2(object):
    """
    Class that contains EASE2 grid parameters

    Attributes
    ----------
    ease_lats : np.array
        Array containing the latitudes of grid rows
    ease_lons : np.array
        Array containing the longitudes of grid columns
    """

    def __init__(self):

        proj = pyproj.Proj(("+proj=cea +lat_0=0 +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m"))

        map_scale = 36032.220840584
        x_min, y_max = proj(-180, 90)
        x_max, y_min = proj(180, -90)

        # Calculate number of grid cells in x-dimension. Map scale is defined such that an exact integer number of
        # grid cells fits in longitude direction.
        x_extent = x_max - x_min
        x_dim = round(x_extent / map_scale)

        # Calculate exact x and y dimensions accounting for rounding error in map-scale (in the 10th digit or so)
        x_pix_size = x_extent / x_dim
        y_pix_size = (map_scale ** 2) / x_pix_size

        # Generate arrays with all x/y center coordinates in map-space, centered around 0
        x_arr_pos = np.arange(x_pix_size / 2, x_max, x_pix_size)
        x_arr_neg = np.arange(-x_pix_size / 2, x_min, -x_pix_size)
        x_arr = np.concatenate([x_arr_neg[::-1], x_arr_pos])

        y_arr_pos = np.arange(y_pix_size / 2, y_max, y_pix_size)
        y_arr_neg = np.arange(-y_pix_size / 2, y_min, -y_pix_size)
        y_arr = np.concatenate([y_arr_pos[::-1], y_arr_neg])

        # north/south-most M36 grid cells do not fit within valid region, i.e. extend beyond +- 90 degrees
        y_arr = y_arr[1:-1]

        # Convert all grid cell coordinates from map-space to lat/lon
        self.ease_lons = proj(x_arr, np.zeros(x_arr.shape), inverse=True)[0]
        self.ease_lats = proj(np.zeros(y_arr.shape), y_arr, inverse=True)[1]



def create_lut():
    """
     Creates a look-up table that maps ASCAT, SMOS, and MERRA grid points onto the EASEv2 grid.
     This is required for the resampling routines within the individual data readers

     """

    # Set to False if a particular data set should be excluded.
    initiate = True
    add_ascat = True
    add_smos = True
    add_merra = True

    paths = Paths()

    fname = paths.lut

    # Rough bounding coordinates to pre-clip CONUS for speeding up calculations
    lonmin = -125.
    lonmax = -66.5
    latmin = 24.5
    latmax = 49.5

    if initiate is True:
        grid = EASE2()
        lons, lats = np.meshgrid(grid.ease_lons, grid.ease_lats)
        cols, rows = np.meshgrid(np.arange(len(grid.ease_lons)), np.arange(len(grid.ease_lats)))

        lut = pd.DataFrame({'ease2_col': cols.flatten(),
                            'ease2_row': rows.flatten(),
                            'ease2_lon': lons.flatten(),
                            'ease2_lat': lats.flatten(),
                            'ascat_gpi': -1,
                            'ascat_lon': np.nan,
                            'ascat_lat': np.nan,
                            'smos_gpi': -1,
                            'smos_lon': np.nan,
                            'smos_lat': np.nan,
                            'merra2_lon': np.nan,
                            'merra2_lat': np.nan})

        lut = lut[(lut.ease2_lon >= lonmin)&(lut.ease2_lon <= lonmax)&
                  (lut.ease2_lat >= latmin)&(lut.ease2_lat <= latmax)]
    else:
        lut = pd.read_csv(fname, index_col=0)


    # ------------------------------------------------------------------------------------------------------------------
    # A list of ASCAT gpis over the USA can be exported from https://www.geo.tuwien.ac.at/dgg/index.php
    # This list is used here to restrict EASE2-grid cells to CONUS only.

    if add_ascat is True:
        ascat_gpis = pd.read_csv(paths.ascat / 'warp5_grid' / 'pointlist_United States of America_warp.csv', index_col=0)
        ascat_gpis = ascat_gpis[(ascat_gpis.lon >= lonmin) & (ascat_gpis.lon <= lonmax) &
                                (ascat_gpis.lat >= latmin) & (ascat_gpis.lat <= latmax)]

        ascat_gpis['ease2_gpi'] = -1
        ascat_gpis['r'] = -1

        # Get ease grid indices and distence for each ASCAT grid cell
        for i, (idx, data) in enumerate(ascat_gpis.iterrows()):
            print('%i / %i' % (i, len(ascat_gpis)))

            r = (lut.ease2_lon - data.lon)**2 + (lut.ease2_lat - data.lat)**2
            ascat_gpis.loc[idx, 'ease2_gpi'] = lut[(r-r.min())<0.0001].index.values[0]
            ascat_gpis.loc[idx, 'r'] = r[(r-r.min())<0.0001].values[0]

        # Find the nearest matched ASCAT grid cell for each EASE grid cells
        for i, (idx, data) in enumerate(lut.iterrows()):
            print('%i / %i' % (i, len(lut)))

            matches = ascat_gpis[ascat_gpis.ease2_gpi==idx]
            if len(matches) > 0:
                match = matches[(matches.r-matches.r.min())<0.0001]
                lut.loc[idx, 'ascat_gpi'] = match.index.values[0]
                lut.loc[idx, 'ascat_lon'] = match['lon'].values[0]
                lut.loc[idx, 'ascat_lat'] = match['lat'].values[0]

        # Remove grid cells the don't have a closest ASCAT cell
        lut = lut[lut.ascat_gpi!=-1]

    # ------------------------------------------------------------------------------------------------------------------
    # Read SMOS grid information and clip CONUS
    # Grid information can be found at: http://www.cesbio.ups-tlse.fr/SMOS_blog/?tag=dgg

    if add_smos is True:
        smos = pd.read_csv(paths.smos / 'smos_grid.txt', delim_whitespace=True, names=['gpi','lon','lat','alt','wf'])
        smos = smos[(smos.lon >= lonmin) & (smos.lon <= lonmax) &
                    (smos.lat >= latmin) & (smos.lat <= latmax)]

        # Find closest SMOS gpis and append to the EASE lookup-table
        for i, (idx, data) in enumerate(lut.iterrows()):
            print('%i / %i' % (i, len(lut)))

            r = (smos.lon - data.ease2_lon)**2 + (smos.lat - data.ease2_lat)**2
            lut.loc[idx, 'smos_gpi'] = smos[(r-r.min())<0.0001]['gpi'].values[0]
            lut.loc[idx, 'smos_lon'] = smos[(r-r.min())<0.0001]['lon'].values[0]
            lut.loc[idx, 'smos_lat'] = smos[(r-r.min())<0.0001]['lat'].values[0]

    # ------------------------------------------------------------------------------------------------------------------
    # Read MERRA-2 grid information (lats/lons taken from a CONUS netcdf image subset)

    if add_merra is True:
        merra = Dataset(paths.merra2 / 'raw' / '2015-2018' / 'MERRA2_400.tavg1_2d_lnd_Nx.20150101.SUB.nc')
        lons, lats = np.meshgrid(merra.variables['lon'][:].data, merra.variables['lat'][:].data)
        lons = lons.flatten()
        lats = lats.flatten()

        # Find closest MERRA gpis and append coordinates to the EASE lookup-table
        for i, (idx, data) in enumerate(lut.iterrows()):
            print('%i / %i' % (i, len(lut)))

            r = (lons - data.ease2_lon) ** 2 + (lats - data.ease2_lat) ** 2
            lut.loc[idx, 'merra2_lon'] = lons[np.where((r - r.min())<0.0001)]
            lut.loc[idx, 'merra2_lat'] = lats[np.where((r - r.min())<0.0001)]


    lut.to_csv(fname, float_format='%.6f')
