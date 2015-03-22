"""Rasterised bathymetry tools

The rasterised bathymetry module provides methods and objects designed to
manipulate bathymetry rasters.

The main functions responsible for loading bathymetry data are:

    - :py:func:load_bathymetry
    - :py:func:load_features

the following helper functions provide convenient method for opening and
displaying information about the bathymetry:

    - :py:func:load_bathymetry_file
    - :py:func:load_bathymetry_meta

The raster data can be manipulated using the following functions:

    - :py:func:sparse_to_full
    - :py:func:full_to_sparse
    - :py:func:subsample_sparse
    - :py:func:select_extent

Raster data can be visualised using:

    - :py:func:plot_raster

.. sectionauthor:: Asher Bender <a.bender@acfr.usyd.edu.au>
.. codeauthor:: Asher Bender <a.bender@acfr.usyd.edu.au>

"""
import os
import pickle
from bz2 import BZ2File

import numpy as np
from scipy.stats import mode

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def check_all_files_exist(file_list):
    """Check a list of files existence. Return True if all files exist.

    Args:
        file_list (list): List of file paths to check for existence.

    Returns:
        bool: `True` if all files in the list exist. `False` otherwise.

    """

    # Assume all files exist.
    exist = True

    # Iterate through list of files and test to ensure they exist.
    for f in file_list:
        if not os.path.isfile(f):
            exist = False
            break

    return exist


def load_bathymetry_file(fname):
    """Load bathymetry data from the disk.

    :py:func:load_bathymetry_file loads BZ2-compressed, pickled bathymetry data
    from the disk.

    Args:
        fname (str): Path to bathymetry file.

    Returns:
        np.array: Array like object containing pickled bathymetry data.

    Raises:
        IOError: If the file does not exist or could not be opened for an
            unexpected reason.

    """

    # File exists, attempt to open.
    if os.path.isfile(fname):
        try:
            with BZ2File(fname, 'r') as f:
                return pickle.load(f)

        # Could not open file, re-throw error.
        except Exception as e:
            msg = 'Could not open the file {0}. '
            msg += "Ensure this is a valid '.pkl.bz2' file. "
            msg += 'The error thrown was:\n\n{2}'
            raise IOError(msg.format(fname, str(e)))

    # The bathymetry information does not exist. Throw error.
    else:
        msg = 'Could not locate the required bathymetry information: '
        msg += "'{0}'."
        raise IOError(fname)


def meta_from_bins(x_bins, y_bins, zone, meta=None, resolution=None,
                   lat_bins=None, lon_bins=None, verbose=False):
    """Create bathymetry meta-data from X and Y bins.

    Creates bathymetry meta-data X and Y bin information. The bathymetry is
    returned as a dictionary containing the following key, value pairs:

        meta = {'x_bins': array(N,),
                'y_bins': array(M,),
                'zone': str(),
                'x_lim': [min(bathy['x_bins']), min(bathy['x_bins'])],
                'y_lim': [min(bathy['y_bins']), min(bathy['y_bins'])],
                'extent': [bathy['x_bins'][0] - bathy['resolution']/2,
                           bathy['x_bins'][1] + bathy['resolution']/2,
                           bathy['y_bins'][0] - bathy['resolution']/2,
                           bathy['y_bins'][1] + bathy['resolution']/2],
                'rows': M,
                'cols': N,
                'shape': [M, N]
                'size': M * N,
                'resolution': float()}

    if WGS84 information is provided, the following optional fields will also
    be returned:

        meta = {'lon_bins': array(N,),
                'lat_bins': array(M,),
                'lat_lim': [min(bathy['lat_bins']), min(bathy['lat_bins'])],
                'lon_lim': [min(bathy['lon_bins']), min(bathy['lon_bins'])]}

    Args:
        x_bins (np.array): array of X co-ordinates for each column in
            bathymetry raster.
        y_bins (np.array): array of Y co-ordinates for each row in bathymetry
            raster.
        zone (string): the UTM zone associated with the northings (y_bins) and
            eastings (x_bins).
        meta (dict, optional): if specified the bathymetry meta-data will be
            added to the provided dictionary.
        resolution (float, optional): if set to `None` the resolution will be
            calculated from the bins. If provided, the value will be stored and
            returned in the meta-data.
        lat_bins (np.array, optional): array of latitudes for each row in
            bathymetry raster (must be specified with lon_bins).
        lon_bins (np.array, optional): array of longitudes for each column in
            bathymetry raster (must be specified with lat_bins).
        verbose (bool, optional): If set to True the contents of the bathymetry
            meta-data dictionary will be summarised on stdout.

    Returns:
        dict: Dictionary object containing the bathymetry meta-data. See
              function description for format details.

    Raises:
        Exception: If lat_bins or lon_bins are provided and do not have the
            same number of elements as y_bins and x_bins.

    """

    # If no dictionary is provided, create an empty dictionary.
    if meta is None:
        meta = dict()

    # Use raster container to store size information.
    meta['x_bins'] = x_bins
    meta['y_bins'] = y_bins
    meta['zone'] = zone

    # Set the resolution. If the resolution is not provide, calculate the
    # resolution from the difference in bin elements.
    if resolution is None:
        meta['resolution'] = float(np.abs(meta['x_bins'][1] - meta['x_bins'][0]))
    else:
        meta['resolution'] = float(resolution)

    # Store latitude and longitude bins if provided.
    if lat_bins is not None and lon_bins is not None:
        meta['lat_bins'] = lat_bins
        if lat_bins.size != y_bins.size:
            msg = 'Latitude and Y-bins must have the same number of elements.'
            raise Exception(msg)

        meta['lon_bins'] = lon_bins
        if lon_bins.size != x_bins.size:
            msg = 'Longitude and X-bins must have the same number of elements.'
            raise Exception(msg)

        # Store lat/lon convenience fields.
        meta['lat_lim'] = [meta['lat_bins'].min(), meta['lat_bins'].max()]
        meta['lon_lim'] = [meta['lon_bins'].min(), meta['lon_bins'].max()]

    # Store 'convenience' fields.
    meta['x_lim'] = [meta['x_bins'].min(), meta['x_bins'].max()]
    meta['y_lim'] = [meta['y_bins'].min(), meta['y_bins'].max()]
    meta['rows'] = meta['y_bins'].size
    meta['cols'] = meta['x_bins'].size
    meta['shape'] = [meta['rows'], meta['cols']]
    meta['size'] = meta['rows'] * meta['cols']

    # Consider X-bin and Y-bin values to mark the centre of bathymetry pixels.
    radius = meta['resolution'] / 2.0
    meta['extent'] = [float(meta['x_lim'][0]) - radius,
                      float(meta['x_lim'][1]) + radius,
                      float(meta['y_lim'][0]) - radius,
                      float(meta['y_lim'][1]) + radius]

    # Summarise bathymetry data.
    if verbose:
        summarise_bathymetry(meta)

    return meta


def summarise_bathymetry(bathy):
    """Print a summary of the bathymetry information.

    Args:
        bathymetry (dict): Bathymetry meta-data (see :py:func:meta_from_bins).

    """

    # Determine whether to print latitude and longitude information.
    wgs84 = False
    if 'lat_bins' in bathy and 'lon_bins' in bathy:
        wgs84 = True

    print 'Bathymetry Summary:'
    print '    X-bins:      [{0},]'.format(bathy['cols'])
    print '    Y-bins:      [{0},]'.format(bathy['rows'])
    print '    UTM-zone:    {0}'.format(bathy['zone'])
    if wgs84:
        print '    lon-bins:    [{0},]'.format(bathy['cols'])
        print '    lat-bins:    [{0},]'.format(bathy['rows'])
        print '    X-lim:       [{0[0]}, {0[1]}]'.format(bathy['x_lim'])
        print '    Y-lim:       [{0[0]}, {0[1]}]'.format(bathy['y_lim'])
    if wgs84:
        print '    lon-lim:     [{0[0]}, {0[1]}]'.format(bathy['lon_lim'])
        print '    lat-lim:     [{0[0]}, {0[1]}]'.format(bathy['lat_lim'])
    print '    extent:      [{0[0]}, {0[1]}, {0[2]}, {0[3]}]'.format(bathy['extent'])
    print '    rows:        {0}'.format(bathy['rows'])
    print '    cols:        {0}'.format(bathy['cols'])
    print '    shape:       [{0[0]}, {0[1]}]'.format(bathy['shape'])
    print '    size:        {0}'.format(bathy['size'])
    print '    resolution:  {0}'.format(bathy['resolution'])


def load_bathymetry_meta(bathymetry_path, verbose=True):
    """Load bathymetry meta-data from the disk.

    Loads bathymetry meta-data from a path WITHOUT loading depth
    information. :py:func:load_bathymetry_meta expects the data to be stored in
    BZ2 compressed pickle files in the following components:

        bathymetry_path/
            |----resolution.pkl.bz2
            |----utm_zone.txt
            |----x_bins.pkl.bz2
            |----y_bins.pkl.bz2

    where the .pkl.bz2 files:

        - resolution: Contains a single float specifying the (square) size of
                      each bathymetry pixel in metres.

        - utm_zone: Contains a string specifying which UTM zone the eastings
                    (x_bins) and northings (y_bins) are specified in.

        - x_bins: Contains a numpy vector storing the local easting, in metres,
                  for each column of bathymetry pixels

        - y_bins: Contains a numpy vector storing the local northing, in
                  metres, for each row of bathymetry pixels

    The bathymetry meta-data is returned as a dictionary containing the same
    key-value pairs as the output of :py:func:meta_from_bins.

    Args:
        bathymetry_path (str): Path to where bathymetry data is stored.
        verbose (bool, optional): If set to True the contents of the bathymetry
            meta-data dictionary will be summarised on stdout.

    Returns:
        dict: Dictionary object containing the bathymetry meta-data (for the
              format of this output see :py:func:meta_from_bins).

    Raises:
        Exception: If mandatory all files could not be located.
        IOError: If the path does not exist or a file could not be opened for
            an unexpected reason.

    """

    # Ensure the path exists.
    if not os.path.exists(bathymetry_path):
        raise IOError('Could not locate the path: {0}'.format(bathymetry_path))

    # Create function for constructing file names from the path.
    mkfname = lambda feat: os.path.join(bathymetry_path, feat + '.pkl.bz2')

    # Ensure mandatory fields exist.
    required = [mkfname(info) for info in ['resolution', 'x_bins', 'y_bins']]
    required.append(os.path.join(bathymetry_path, 'utm_zone.txt'))
    if not check_all_files_exist(required):
        msg = "The files 'zone.txt', 'resolution.pkl.bz2', 'x_bins.pkl.bz2' "
        msg += "and 'y_bins.pkl.bz2' must exist in the path {0}."
        raise Exception(msg.format(bathymetry_path))

    # Iterate through bathymetry information and load into dictionary.
    bathy = dict()
    for info in ['resolution', 'x_bins', 'y_bins']:
        try:
            bathy[info] = load_bathymetry_file(mkfname(info))
        except:
            raise

    # Load UTM zone information.
    with open(required[-1], 'r') as f:
        bathy['zone'] = f.read().strip()

    # Load WGS84 cell co-ordinates if they exist.
    WGS84 = [mkfname(info) for info in ['lon_bins', 'lat_bins']]
    if check_all_files_exist(WGS84):
        for fname in WGS84:
            try:
                bathy[info] = load_bathymetry_file(fname)
            except:
                raise

    # Create meta-data with WGS84 information.
    if all(i in bathy for i in ['lon_bins', 'lat_bins']):
        bathy = meta_from_bins(bathy['x_bins'], bathy['y_bins'], bathy['zone'],
                               resolution=bathy['resolution'],
                               lat_bins=bathy['lat_bins'],
                               lon_bins=bathy['lon_bins'], verbose=verbose)

    # Create meta-data without WGS84 information.
    else:
        bathy = meta_from_bins(bathy['x_bins'], bathy['y_bins'], bathy['zone'],
                               resolution=bathy['resolution'], verbose=verbose)

    return bathy


def load_bathymetry(bathymetry_path, invalid=np.nan, verbose=True):
    """Load bathymetry raster from the disk.

    Loads bathymetry information from a path. :py:func:load_bathymetry expects
    the data to be stored in BZ2 compressed pickle files in the following
    components:

        bathymetry_path/
            |----depth.pkl.bz2
            |----index.pkl.bz2
            |----resolution.pkl.bz2
            |----utm_zone.txt
            |----x_bins.pkl.bz2
            |----y_bins.pkl.bz2
            |----<optional> lat_bins.pkl.bz2
            |----<optional> lon_bins.pkl.bz2

    where the .pkl.bz2 files:

        - depth: Contains a numpy vector storing valid bathymetric
                 data. Out-of-range or invalid returns are NOT stored. Each
                 element in the 'depth' vector corresponds to an element in the
                 'index' vector - this position information is used to
                 reconstruct the raster.

        - index: Contains a numpy vector of integers specifying the
                 column-major order (Fortran-like) index ordering of the
                 bathymetry pixels. Note that the shape of the raster is given
                 by the size of 'x_bins' (cols) and 'y_bins' (rows).

        - resolution: Contains a single float specifying the (square) size of
                      each bathymetry pixel in metres.

        - utm_zone: Contains a string specifying which UTM zone the eastings
                    (x_bins) and northings (y_bins) are specified in.

        - x_bins: Contains a numpy vector storing the local easting, in metres,
                  for each column of bathymetry pixels

        - y_bins: Contains a numpy vector storing the local northing, in
                  metres, for each row of bathymetry pixels

        - lat_bins: Contains a numpy vector storing the WGS84 latitude for each
                  row of bathymetry pixels

        - lon_bins: Contains a numpy vector storing the WGS84 longitude for
                  each column of bathymetry pixels

    The bathymetry is returned as a dictionary containing the same key-value
    pairs as the output of :py:func:meta_from_bins. An additional key 'depth'
    is added which contains a [MxN] numpy array of depth values.

    Args:
        bathymetry_path (str): Path to where bathymetry data is stored.
        invalid (value, optional): Value to used to represent pixels where
            bathymetry is not available.
        verbose (bool, optional): If set to True the contents of the bathymetry
            dictionary will be summarised on stdout.

    Returns:
        dict: Dictionary object containing the bathymetry data. The format of
              this dictionary is detailed in :py:func:meta_from_bins, the
              dictionary contains an additional key 'depth' which contains the
              bathymetry data.

    Raises:
        Exception: If a file could not be opened for an unexpected reason.
        IOError: If the path or a required bathymetry file does not exist.

    """

    # Attempt to load bathymetry meta-data.
    try:
        bathy = load_bathymetry_meta(bathymetry_path, verbose=False)
    except:
        raise

    # Iterate through bathymetry information and load into dictionary.
    for info in ['depth', 'index']:
        fname = os.path.join(bathymetry_path, info + '.pkl.bz2')
        try:
            bathy[info] = load_bathymetry_file(fname)
        except:
            raise

    # Reshape sparse bathymetry data into a dense raster.
    bathy['depth'] = sparse_to_full(bathy['index'],
                                    bathy['depth'],
                                    bathy['rows'],
                                    bathy['cols'],
                                    invalid=invalid)

    # Summarise bathymetry data.
    if verbose:
        summarise_bathymetry(bathy)

    return bathy


def load_features(features, bathymetry_path, transform=False, verbose=True):
    """Compiles bathymetry features into a single feature vector.

    Loads bathymetry features from a path and compiles them into a single
    vector.

    The features which are load and returned are specified by a string. The
    string is a sequence of commands and optional arguments. The commands
    represent features to load. Supported commands are:

        D: Depth (no argument required)
        S: slope    (argument required - integer scale)
        R: rugosity (argument required - integer scale)
        A: aspect   (argument required - integer scale)

    The following example:

        'DR2S4'

    would return a matrix where the first column contains depth, the second
    column contains rugosity at a scale of 2 points and the final column
    contains slope at a scale of 4.

    :py:func:load_features expects the data to be stored in BZ2 compressed
    pickle files in the following components:

        bathymetry_path/
            |    |----depth.pkl.bz2
            |    |----index.pkl.bz2
            |    |----utm_zone.txt
            |    |----x_bins.pkl.bz2
            |    |----y_bins.pkl.bz2
            |    |----<optional> lat_bins.pkl.bz2
            |    |----<optional> lon_bins.pkl.bz2
            |
            +----<scale>
                 |----aspect.pkl.bz2
                 |----index.pkl.bz2
                 |----rugosity.pkl.bz2
                 |----slope.pkl.bz2

    where:

        - <scale>: is the integer scale at which the bathymetry features were
                   calculated. The string is three characters long. Unused
                   characters are padded with 0's (e.g. 002).

    and the .pkl.bz2 files:

        - depth/aspect/rugosity/slope: Contains a numpy vector storing valid
                 bathymetric features. Out-of-range or invalid returns are NOT
                 stored. Each element in the vector corresponds to an element
                 in the 'index' vector' found at the same directory level.

        - index: Contains a numpy vector of integers specifying the
                 column-major order (Fortran-like) index ordering of the
                 bathymetry pixels. Note that the shape of the raster is given
                 by the size of 'x_bins' (cols) and 'y_bins' (rows).

    for a description of the other files in this directory structure see
    :py:func:load_bathymetry.

    Note::

        There is no guarantee that all features will be available at any
        particular location. As a result, rows in the output which do not have
        a full compliment of features (entire row of valid data) are discarded.

    Args:
        features (str): String specifying which features are loaded (see
            documentation text).
        bathymetry_path (str): Path to where bathymetry data is stored.
        transform (bool, optional): If set to True, rugosity will be returned
            as log(rugosity - 1) and slope will be returned as log(slope).
        verbose (bool, optional): If set to True progress will be output on
            stdout.

    Returns:
        tuple: A tuple where the first element contains the matrix of
               bathymetry features. The second element contains a list of human
               readable feature names. The final element contains a linear
               index to the original location in the bathymetry raster.

    Raises:
        Exception: If a file could not be opened for an unexpected reason.
        IOError: If the path or a required bathymetry file does not exist.

    """

    # Define short hand notation for valid features.
    valid_features = ['D', 'A', 'R', 'S']

    # Attempt to load bathymetry meta-data.
    try:
        bathy = load_bathymetry_meta(bathymetry_path, verbose=False)
    except:
        raise

    i = 0
    feature_type = list()
    feature_scale = list()
    end_of_string = False
    while not end_of_string:
        char = features[i]

        # If character is a known feature, save the feature type.
        if char in valid_features:
            feature_type.append(char)
            i += 1

            # The depth feature has no scale.
            if char == 'D':
                feature_scale.append(0)

        # If the character is not a recognised feature, assume the character is
        # numeric and is specifying a scale.
        else:
            scale = ''
            while (i < len(features)) and (features[i] not in valid_features):
                scale += features[i]
                i += 1

            # Convert scale from string to an integer.
            try:
                feature_scale.append(int(scale))
            except:
                msg = "In the features specified '{0}', "
                msg += "the string {1} is not numeric."
                raise Exception(msg.format(features, scale))

        # Reached end of string terminate parsing.
        if i >= len(features):
            end_of_string = True

    # Iterate through features and load data.
    index_cache = dict()
    feature_name = list()
    feature_list = list()
    eps = np.finfo(float).eps
    for feat, scale in zip(feature_type, feature_scale):

        # Feature is depth.
        if feat == 'D':
            feature_name.append('Depth')

            if verbose:
                print 'Loading depth'

            # Load depth and expand into column vector.
            iname = os.path.join(bathymetry_path, 'index.pkl.bz2')
            dname = os.path.join(bathymetry_path, 'depth.pkl.bz2')
            data = load_bathymetry_file(dname)
            index = load_bathymetry_file(iname)

        # Feature is aspect, rugosity or slope.
        else:
            f_path = os.path.join(bathymetry_path, '{0:0>3n}'.format(scale))

            # Load index of feature.
            if scale not in index_cache:
                iname = os.path.join(f_path, 'index.pkl.bz2')
                index = load_bathymetry_file(iname)
                index_cache[scale] = index
            else:
                index = index_cache[scale]

            # Insert aspect vector.
            if feat == 'A':
                if verbose:
                    print 'Loading aspect,   scale {0:>3n}'.format(scale)

                feature_name.append('Aspect {0:>3n}'.format(scale))
                dname = os.path.join(f_path, 'aspect.pkl.bz2')
                data = load_bathymetry_file(dname)

            # Insert rugosity vector.
            elif feat == 'R':
                if verbose:
                    print 'Loading rugosity, scale {0:>3n}'.format(scale)

                feature_name.append('Rugosity {0:>3n}'.format(scale))
                dname = os.path.join(f_path, 'rugosity.pkl.bz2')
                data = load_bathymetry_file(dname)
                if transform:
                    data = np.log(data - 1 + eps)

            # Insert slope vector.
            elif feat == 'S':
                if verbose:
                    print 'Loading slope,    scale {0:>3n}'.format(scale)

                feature_name.append('Slope {0:>3n}'.format(scale))
                dname = os.path.join(f_path, 'slope.pkl.bz2')
                data = load_bathymetry_file(dname)
                if transform:
                    data = np.log(data + eps)

        # Copy data full raster. Note that the indices are stored in column
        # major order (Fortran-like).
        column = np.nan * np.ones(bathy['size'])
        column[index] = data
        feature_list.append(column)
        del(index)
        del(data)

    # Concatenate features into a single vector.
    features = np.vstack(feature_list).T

    # Find rows which contain a full set of features (due to differences in
    # scale some rows may be incomplete).
    index = np.all(np.isnan(features) == False, axis=1)

    # Prune rows which contain invalid elements.
    features = features[index, :]

    return features, feature_name, np.where(index)[0]


def full_to_sparse(raster, invalid=np.nan):
    """Convert a dense raster into sparse data.

    Args:
        raster (np.array): (MxN) array of raster data.
            `sparse` entries are located in the raster.
        invalid (value, optional): Value to used to represent pixels where
            raster data is not available.

    Returns:

        tuple: The first element is a column major order (Fortran-like) index
            where `sparse` entries are located in the raster. The second
            element is an array of sparse raster data.

    """

    # Flatten raster data into column major order (Fortran-like).
    raster = raster.flatten(order='F')

    # Find locations in the bathymetry that are valid.
    if np.isnan(invalid):
        # Special case. For some reason the following does not work:
        #
        #     raster != np.nan
        #
        index = np.where(~np.isnan(raster))[0]
    else:
        index = np.where(raster != invalid)[0]

    return index, raster[index]


def sparse_to_full(index, sparse, rows, cols, invalid=np.nan):
    """Convert sparse data into a dense raster.

    Args:
        index (np.array): Column major order (Fortran-like) index where
            `sparse` entries are located in the raster.
        sparse (np.array): array of sparse raster data,
        rows (int): number of rows in the raster.
        cols (int): number of columns in the raster.
        invalid (value, optional): Value to used to represent pixels where
            raster is not available.

    Returns:
        np.array: (rows x cols) dense raster containing sparse data.

    """

    # Copy observed values into full raster. Note that the indices are stored
    # in column major order (Fortran-like).
    raster = invalid * np.ones(rows * cols)
    raster[index] = sparse

    return raster.reshape((rows, cols), order='F')


def cartesian_to_bathymetry(bathymetry, easting, northing):
    """Convert Cartesian co-ordinates to bathymetry row, column subscripts.

    Args:
        bathymetry (dict): Bathymetry meta-data (see :py:func:meta_from_bins).
        easting (np.array): Eastings in cartesian co-ordinates. These values
            must lie within the easting limits of the raster.
        northing (np.array): Northings in cartesian co-ordinates. These values
            must lie within the northing limits of the raster.

    Returns:
        tuple: The first element is a np.array of rows and a the second element
               is a np.array of columns where the Eastings and Northings occur
               in the bathymetry raster. The output rows and columns are the
               same length as the input Northing and Eastings.

    Raises:
        Exception: If the input eastings and northings contain an unequal
            number of elements OR it any easting-northing pairs occur outside
            the limits of the bathymetry.

    """

    # Ensure inputs are the same length (via number of elements).
    if (easting.size != northing.size):
        msg = 'The input eastings and northings must contain the same number '
        msg += 'of elements.'
        raise Exception(msg)

    # Ensure input locations lie within the bathymetry.
    if (np.any(easting < min(bathymetry['x_lim'])) or
        np.any(easting > max(bathymetry['x_lim'])) or
        np.any(northing < min(bathymetry['y_lim'])) or
        np.any(northing > max(bathymetry['y_lim']))):
        msg = 'All eastings and northings must lie within the limits of '
        msg += 'the bathymetry.'
        raise Exception(msg)

    # Convert AUV northings and eastings into a matrix subscripts.
    resolution = bathymetry['resolution']
    cols = np.floor((easting - min(bathymetry['x_lim'])) / resolution)
    rows = np.floor((northing - min(bathymetry['y_lim'])) / resolution)
    rows = bathymetry['rows'] - rows - 1

    return rows.astype(int), cols.astype(int)


def label_bathymetry_pixels(easting, northing, classes, bathymetry, valid):
    """Label bathymetry pixels with AUV class observations.

    :py:func:label_bathymetry_pixels classifies bathymetry pixels given a set
    of co-ordinates in Cartesian space and a corresponding vector of
    classifications. If a Cartesian co-ordinate occurs over a pixel which has
    been marked as invalid, the Cartesian co-ordinate is discarded. Pixels
    which have been observed multiple times are labelled with the most
    frequently occurring label.

    Args:
        easting (np.array): Eastings in Cartesian co-ordinates. These values
            must lie within the easting limits of the raster.
        northing (np.array): Northings in Cartesian co-ordinates. These values
            must lie within the northing limits of the raster.
        classes (np.array): Classes observed at Cartesian co-ordinates.
        bathymetry (dict): Bathymetry meta-data (see :py:func:meta_from_bins).
        valid (np.array): array of indices in the bathymetry raster which the
            AUV observations are able to observe. If a bathymetry pixel is not
            referenced in this array, it will not appear in the labelled
            output.

    Returns:
        tuple: The first element is a np.array of indices, in column major
               order (Fortran-like), where AUV observations were made. The
               second element is a np.array of pixel class assignments.

    Raises:
        Exception: If the input eastings and northings contain an unequal
            number of elements OR it any easting-northing pairs occur outside
            the limits of the bathymetry.

    """

    # Convert cartesian co-ordinates to bathymetry rows/cols.
    try:
        rows, cols = cartesian_to_bathymetry(bathymetry, easting, northing)
    except:
        raise
    index = np.ravel_multi_index([rows, cols],
                                 dims=bathymetry['shape'],
                                 order='F')

    # Remove cells which do not traverse valid indices.
    AUV_in_feature = np.in1d(index, valid)
    index = index[AUV_in_feature]
    classes = classes[AUV_in_feature]

    # Find unique bathymetric cells visited by AUV surveys.
    unique_index, idx, counts = np.unique(index,
                                          return_index=True,
                                          return_counts=True)

    # Calculate the most frequently observed class for cells with multiple
    # observations.
    C = classes[idx]
    for i in np.where(counts != 1)[0]:
        C[i] = mode(classes[index == unique_index[i]])[0][0]

    return unique_index.flatten(), C.flatten()


def subsample_sparse(subsample, index, bathymetry, verbose=True):
    """Sub-sample a sparse matrix.

    Return the index to elements in sparse bathymetry which create a
    sub-sampled raster.

    Args:
        subsample (int): integer scale to sub-sample (every N-th point).
        index (np.array): array of indices indicating which elements of the
            bathymetry raster contain valid values.
        bathymetry (dict): Bathymetry meta-data (see :py:func:meta_from_bins).
        verbose (bool, optional): If set to True the contents of the bathymetry
            meta-data dictionary will be summarised on stdout.

    Returns: tuple: The first element is a np.array of indices, referring to
        elements in the sparse bathymetry that occur in the sub-sampled
        raster. The second element is a corresponding np.array that indicates
        where the sub-sampled sparse bathymetry occurs in the sub-sampled
        raster. The final element of the tuple is a bathymetry meta-data
        dictionary containing size information of the sub-sampled bathymetry
        raster.

    """

    # Grid sub-sampled row and column subscripts.
    sub_rows = np.arange(0, bathymetry['rows'], subsample, dtype=np.int)
    sub_cols = np.arange(0, bathymetry['cols'], subsample, dtype=np.int)

    # Generate meta-data for sub-sampled raster.
    sub_raster = meta_from_bins(bathymetry['x_bins'][sub_cols],
                                bathymetry['y_bins'][sub_rows],
                                bathymetry['zone'],
                                verbose=verbose)

    # Create mask of invalid locations in the raster given the index.
    invalid = np.ones(bathymetry['size'], dtype=bool)
    invalid[index] = False
    invalid = invalid.reshape(bathymetry['shape'], order='C')

    # Create a matrix that containing the linear index of each pixel in the
    # full raster. Mask off invalid areas as NaNs.
    full_index = np.arange(bathymetry['size'], dtype=float)
    full_index[invalid.flatten()] = np.nan
    full_index = full_index.reshape(bathymetry['shape'], order='F')

    # Create a sub-sampled matrix that containing the linear index of each
    # pixel in the full raster.
    sub_index = full_index[sub_rows, :]
    sub_index = sub_index[:, sub_cols]
    sub_index = sub_index.flatten(order='F')

    # Index to valid features which occur in the sub-sampled raster.
    feature_to_sub = np.where(np.in1d(index, sub_index))[0]

    # Location of valid feature in sub-sampled raster.
    sub_index = np.where(~np.isnan(sub_index))[0]

    # Return valid indices in array of input indices and the corresponding
    # location in the sub-sampled matrix.
    return feature_to_sub, sub_index, sub_raster


def select_extent(extent, index, bathymetry, verbose=True):
    """Select a rectangular extent from sparse raster data.

    Return the index to elements in sparse bathymetry which occupy a
    rectangular region in Cartesian space.

    Args:
        extent (np.array): vector specifying the rectangular extent to
            extract. The vector is specified in the form
            [x-min, x-max, y-min, y-max].
        index (np.array): array of indices indicating which elements of the
            bathymetry raster contain valid values.
        bathymetry (dict): Bathymetry meta-data (see :py:func:meta_from_bins).
        verbose (bool, optional): If set to True the contents of the bathymetry
            meta-data dictionary will be summarised on stdout.

    Returns: tuple: The first element is a np.array of indices, referring to
        elements in the sparse bathymetry that occur in the sub-sampled
        raster. The second element is a corresponding np.array that indicates
        where the sub-sampled sparse bathymetry occurs in the sub-sampled
        raster. The final element of the tuple is a bathymetry meta-data
        dictionary containing size information of the sub-sampled bathymetry
        raster.

    """

    # Ensure extent is specified correctly.
    if len(extent) != 4:
        msg = 'extent must contain four elements [x-min, x-max, y-min, y-max].'
        raise Exception(msg)
    if (extent[0] >= extent[1]):
        msg = 'X-min must be smaller than X-max.'
        raise Exception(msg)
    if (extent[2] >= extent[3]):
        msg = 'Y-min must be smaller than Y-max.'
        raise Exception(msg)

    # Get rows and columns within extent.
    sub_cols = ((bathymetry['x_bins'] >= extent[0]) &
                (bathymetry['x_bins'] <= extent[1]))
    sub_rows = ((bathymetry['y_bins'] >= extent[2]) &
                (bathymetry['y_bins'] <= extent[3]))

    # Generate meta-data for sub-sampled raster.
    sub_raster = meta_from_bins(bathymetry['x_bins'][sub_cols],
                                bathymetry['y_bins'][sub_rows],
                                bathymetry['zone'],
                                resolution=bathymetry['resolution'],
                                verbose=verbose)

    # Create mask of valid locations in the raster given the index.
    valid = np.zeros(bathymetry['size'], dtype=bool)
    valid[index] = True
    valid = valid.reshape(bathymetry['shape'], order='F')

    # Create mask of valid locations in the raster given the row and column
    # constraints.
    valid[~sub_rows, :] = False
    valid[:, ~sub_cols] = False

    # Create a matrix that containing the linear index of each pixel in the
    # full raster. Mask off invalid areas as NaNs.
    full_index = np.arange(bathymetry['size'], dtype=float)
    full_index[~valid.flatten(order='F')] = np.nan
    full_index = full_index.reshape(bathymetry['shape'], order='F')

    # Create a sub-sampled matrix that containing the linear index of each
    # pixel in the full raster.
    sub_index = full_index[sub_rows, :]
    sub_index = sub_index[:, sub_cols]
    sub_index = sub_index.flatten(order='F')

    # Index to valid features which occur in the sub-sampled raster.
    feature_to_sub = np.where(np.in1d(index, sub_index))[0]

    # Location of valid feature in sub-sampled raster.
    sub_index = np.where(~np.isnan(sub_index))[0]

    # Return valid indices in array of input indices and the corresponding
    # location in the sub-sampled matrix.
    return feature_to_sub, sub_index, sub_raster


# --------------------------------------------------------------------------- #
#                             Plotting Functions
# --------------------------------------------------------------------------- #


def plot_raster(raster, ax=None, extent=None, title=None, clabel=None,
                no_cbar=False, no_axis_label=False, no_ticks=False,
                horz_cbar=False, **kwargs):
    """Plot raster as an image.

    Args:
        raster (np.array): (MxN) raster to plot.
        ax (matplotlib.axes.Axes, optional): Axes to plot raster. If set to
            `None`, a new axis will be created for plotting.
        extent (list, optional): extent of plot [x_min, x_max, y_min, y_max] in
             Cartesian space. If provided the X and Y labels will be labelled
             automatically.
        title (str, optional): Title of plot. If set to `None`, no title will
            be printed.
        clabel (str, optional): Label of colour bar. If set to `None`, no
            colour bar label will be rendered.
        no_cbar (bool, optional): If set to `True` the colour bar will NOT be
            rendered.  If set to `False` the colour bar WILL be rendered.
        no_axis_label (bool, optional): If set to `True` the X/Y labels will
            NOT be rendered.  If set to `False` the X/Y labels WILL be
            rendered.
        no_ticks (bool, optional): If set to `True` the X/Y tick marks will NOT
            be rendered.  If set to `False` the X/Y tick marks WILL be
            rendered.
        horz_cbar (bool, optional): If set to `True` the colour bar will
            rendered horizontally below the figure. This option is best used
            with `no_axis_label=True` and `no_ticks=True`.
        **kwargs: Are passed into matplotlib.pyplot.imshow when the raster is
            rendered.

    Returns:
        matplotlib.axes.Axes: axes containing raster image.

    """

    # Create axes if none is provided.
    if ax is None:
        ax = plt.subplot(111)

    # Ensure the raster is plotted in 'real-world' co-ordinates.
    if extent is not None:
        kwargs['extent'] = extent

    # Plot raster.
    im = ax.imshow(raster, **kwargs)

    # Assign title to image.
    if title:
        plt.title(title)

    # Label axes.
    plt.grid('on')
    if (extent is not None) and not no_axis_label:
        plt.xlabel('Local Easting (m)')
        plt.ylabel('Local Northing (m)')

    # Remove tick marks.
    if no_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Create colour bar.
    if not no_cbar:

        # Vertical colour bar.
        if not horz_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            if clabel is not None:
                cbar.set_label(clabel, rotation=90)

        # Horizontal colour bar.
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
            if clabel is not None:
                cbar.set_label('', rotation=0)

    # Force extent of plot.
    if extent is not None:
        ax.axis(extent)

    return ax
