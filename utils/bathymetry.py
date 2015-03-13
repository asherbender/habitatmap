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
    - :py:func:sparse_to_raster

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

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_bathymetry_file(fname):
    """Load bathymetry data from the disk.

    :py:func:load_bathymetry_file loads BZ2-compressed, pickled bathymetry data
    from the disk.

    Args:
        fname (str): Path to bathymetry file.

    Returns:
        np.array: Array like object containing pickled bathymetry data.

    Raises:
        Exception: If the file could not be opened for an unexpected reason.
        IOError: If the file does not exist.

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
            raise Exception(msg.format(fname, str(e)))

    # The bathymetry information does not exist. Throw error.
    else:
        msg = 'Could not locate the required bathymetry information: '
        msg += "'{0}'."
        raise IOError(fname)


def meta_from_bins(x_bins, y_bins, meta=None, resolution=None, verbose=False):
    """Create bathymetry meta-data from X and Y bins.

    Creates bathymetry meta-data X and Y bin information. The bathymetry is
    returned as a dictionary containing the following key, value pairs:

        meta = {'x_bins': array(N,),
                'y_bins': array(M,),
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

    Args:
        x_bins (np.array): array of X co-ordinate for each column in bathymetry
            raster.
        y_bins (np.array): array of Y co-ordinate for each row in bathymetry
            raster.
        meta (dict, optional): if specified the bathymetry meta-data will be
            added to the provided dictionary.
        resolution (float, optional): if set to `None` the resolution will be
            calculated from the bins. If provided, the value will be stored in the
            meta-data.
        verbose (bool, optional): If set to True the contents of the bathymetry
            meta-data dictionary will be summarised on stdout.

    Returns:
        dict: Dictionary object containing the bathymetry meta-data. See
              function description for format details.

    """

    # If no dictionary is provided, create an empty dictionary.
    if meta is None:
        meta = dict()

    # Use raster container to store size information.
    meta['x_bins'] = x_bins
    meta['y_bins'] = y_bins

    # Set the resolution. If the resolution is not provide, calculate the
    # resolution from the difference in bin elements.
    if resolution is None:
        meta['resolution'] = np.abs(meta['x_bins'][1] - meta['x_bins'][0])
    else:
        meta['resolution'] = resolution

    # Store 'convenience' fields.
    meta['x_lim'] = [meta['x_bins'].min(), meta['x_bins'].max()]
    meta['y_lim'] = [meta['y_bins'].min(), meta['y_bins'].max()]
    meta['rows'] = meta['y_bins'].size
    meta['cols'] = meta['x_bins'].size
    meta['shape'] = [meta['rows'], meta['cols']]
    meta['size'] = meta['rows'] * meta['cols']

    # Consider X-bin and Y-bin values to mark the centre of bathymetry pixels.
    radius = meta['resolution'] / 2.0
    meta['extent'] = [meta['x_lim'][0] - radius, meta['x_lim'][1] + radius,
                      meta['y_lim'][0] - radius, meta['y_lim'][1] + radius]

    # Summarise bathymetry data.
    if verbose:
        print 'Bathymetry Summary:'
        print '    X-bins:      [{0},]'.format(meta['cols'])
        print '    Y-bins:      [{0},]'.format(meta['rows'])
        print '    X-lim:       [{0[0]}, {0[1]}]'.format(meta['x_lim'])
        print '    Y-lim:       [{0[0]}, {0[1]}]'.format(meta['y_lim'])
        print '    extent:      [{0[0]}, {0[1]}, {0[2]}, {0[3]}]'.format(meta['extent'])
        print '    rows:        {0}'.format(meta['rows'])
        print '    cols:        {0}'.format(meta['cols'])
        print '    shape:       [{0[0]}, {0[1]}]'.format(meta['shape'])
        print '    size:        {0}'.format(meta['size'])
        print '    resolution:  {0}'.format(meta['resolution'])

    return meta


def load_bathymetry_meta(bathymetry_path, verbose=True):
    """Load bathymetry meta-data from the disk.

    Loads bathymetry meta-data from a path WITHOUT loading depth
    information. :py:func:load_bathymetry_meta expects the data to be stored in
    BZ2 compressed pickle files in the following components:

        bathymetry_path/
            |----resolution.pkl.bz2
            |----x_bins.pkl.bz2
            |----y_bins.pkl.bz2

    where the .pkl.bz2 files:

        - resolution: Contains a single float specifying the (square) size of
                      each bathymetry pixel in metres.

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
        Exception: If a file could not be opened for an unexpected reason.
        IOError: If the path or a required bathymetry file does not exist.

    """

    # Ensure the path exists.
    if not os.path.exists(bathymetry_path):
        raise IOError('Could not locate the path: {0}'.format(bathymetry_path))

    # Iterate through bathymetry information and load into dictionary.
    bathy = dict()
    for info in ['resolution', 'x_bins', 'y_bins']:
        fname = os.path.join(bathymetry_path, info + '.pkl.bz2')
        try:
            bathy[info] = load_bathymetry_file(fname)
        except:
            raise

    # Create meta-data from x/y bins and resolution.
    bathy = meta_from_bins(bathy['x_bins'], bathy['y_bins'],
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
            |----x_bins.pkl.bz2
            |----y_bins.pkl.bz2

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

        - x_bins: Contains a numpy vector storing the local easting, in metres,
                  for each column of bathymetry pixels

        - y_bins: Contains a numpy vector storing the local northing, in
                  metres, for each row of bathymetry pixels

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
    bathy['depth'] = sparse_to_raster(bathy['index'],
                                      bathy['depth'],
                                      bathy['rows'],
                                      bathy['cols'],
                                      invalid=invalid)

    # Summarise bathymetry data.
    if verbose:
        print 'Bathymetry Summary:'
        print '    depth:       [{0[0]}x{0[1]}]'.format(bathy['depth'].shape)
        print '    X-bins:      [{0[0]},]'.format(bathy['x_bins'].shape)
        print '    Y-bins:      [{0[0]},]'.format(bathy['y_bins'].shape)
        print '    X-lim:       [{0[0]}, {0[1]}]'.format(bathy['x_lim'])
        print '    Y-lim:       [{0[0]}, {0[1]}]'.format(bathy['y_lim'])
        print '    extent:      [{0[0]}, {0[1]}, {0[2]}, {0[3]}]'.format(bathy['extent'])
        print '    rows:        {0}'.format(bathy['rows'])
        print '    cols:        {0}'.format(bathy['cols'])
        print '    size:        {0}'.format(bathy['size'])
        print '    resolution:  {0}'.format(bathy['resolution'])

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
            |    |----x_bins.pkl.bz2
            |    |----y_bins.pkl.bz2
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

        - x_bins: Contains a numpy vector storing the local easting, in metres,
                  for each column of bathymetry pixels

        - y_bins: Contains a numpy vector storing the local northing, in
                  metres, for each row of bathymetry pixels

    Note::

        There is no gaurantee that all features will be available at any
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


def sparse_to_raster(index, sparse, rows, cols, invalid=np.nan):
    """Convert sparse data into a dense raster.

    Args:
        index (np.array): Column major order (Fortran-like) index where
            `sparse` entries are located in the raster.
        sparse (np.array): array of sparse raster data,
        rows (int): number of rows in the raster.
        cols (int): number of columns in the raster.
        invalid (value, optional): Value to used to represent pixels where
            bathymetry is not available.

    Returns:
        np.array: dense raster containing sparse data.

    """

    # Copy observed values into full raster. Note that the indices are stored
    # in column major order (Fortran-like).
    raster = invalid * np.ones(rows * cols)
    raster[index] = sparse

    return raster.reshape((rows, cols), order='F')


def intersect(A, B, index=False):
    """Find the intersection of two arrays.

    Return the sorted, unique values that are in both of the input arrays. The
    input arrays `A` and `B` must be sorted and unique. Where this function
    differs from :py:func:np.intersect1d is that an index array for each input
    can optionally be returned. For example when using:

        C, IA, IB = intersect(A, B, index=True)

    the following is true:

        C = A[IA] and C = B[IB]

    Args:
        A (np.array): First input array.
        B (np.array): Second input array.
        index (bool, optional): If set to `True` an additional two arrays will
            be returned. The first array is a boolean mask the same length as A
            that is `True` where an element of A is in B and `False` otherwise.
            The second array is a boolean mask the same length as B that is
            `True` where an element of B is in A and `False` otherwise.

    Returns:
        np.array: Sorted array of common and unique elements. If index=True, a
                  tuple is returned where the first element is the set
                  intersection. The second element is a boolean mask to the
                  first array such that C = A[IA]. The last element is a
                  boolean mask to the second array such that C = B[IB].


    Raises:
        Exception: If either of the input arrays are not sorted or unique.

    """

    if not np.all(A == np.unique(A)):
        raise Exception('The first input must be sorted and unique.')

    if not np.all(B == np.unique(B)):
        raise Exception('The second input must be sorted and unique.')

    # Return a boolean array the same length as A that is True where an element
    # of A is in B and False otherwise (vice-versa for B).
    index_A = np.in1d(A, B, assume_unique=False)
    index_B = np.in1d(B, A, assume_unique=False)

    if not index:
        return A[index_A]
    else:
        return A[index_A], index_A, index_B


def cartesian_to_bathymetry(bathymetry, easting, northing):
    """Convert Cartesian co-ordinates to bathymetry row, column subscripts.

    Args:
        bathymetry (dict): Bathymetry meta-data (see
            :py:func:load_bathymetry_meta).
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
        bathymetry (dict): Bathymetry meta-data (see
            :py:func:load_bathymetry_meta).
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


def subsample_sparse_raster(subsample, index, bathymetry, verbose=True):
    """Sub-sample a sparse matrix.

    Return the index to elements in sparse bathymetry which create a
    sub-sampled raster.

    Args:
        subsample (int): integer scale to sub-sample (every N-th point).
        index (np.array): array of indices indicating which elements of the
            bathymetry raster contain valid values.
        bathymetry (dict): Bathymetry meta-data (see
            :py:func:load_bathymetry_meta).
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


# --------------------------------------------------------------------------- #
#                                Consider moving
# --------------------------------------------------------------------------- #

def feasible_region(template, raster):

    # Get limits of raster and template.
    x_min = template[:, 0].min()
    x_max = raster.cols - template[:, 0].max()
    y_max = raster.rows - template[:, 1].min()
    y_min = template[:, 1].max()

    # Mask off rows and columns where the template would 'fall off' the raster.
    row_mask = np.arange(raster.rows)
    col_mask = np.arange(raster.cols)
    feasible = np.ones(raster.shape, dtype=bool)
    feasible[:, col_mask < x_min] = False
    feasible[:, col_mask > x_max] = False
    feasible[row_mask < y_min, :] = False
    feasible[row_mask > y_max, :] = False

    return feasible


def plot_feasible_region(feasible, raster, limits, **kwargs):
    """Overlay feasible region on bathymetry (unfeasible in red)."""

    # Convert feasible region into an RGB image.
    feasible = feasible[:, :, np.newaxis].repeat(3, axis=2)
    feasible[:, :, 0] = ~feasible[:, :, 0]
    feasible[:, :, 2] = 0

    # Plot bathymetry in gray-scale and feasible region on top.
    ax = plot_raster(raster, limits, cmap=cm.gray)
    ax.imshow(feasible, extent=limits, interpolation='none', **kwargs)

    return ax


# plt.imshow(sub_index)

# sub_raster = utils.bathymetry.meta_from_bins(bathymetry['x_bins'][sub_cols],
#                                              bathymetry['y_bins'][sub_rows])


# cols_grid, rows_grid = np.meshgrid(sub_cols, sub_rows)


# # Convert subsampled data to square matrix.
# sub_to_full = np.ravel_multi_index([rows_grid,
#                                     cols_grid],
#                                    dims=bathymetry['shape'], order='F')

# # Ensure unique values.
# U = np.unique(feature_index)
# feature_to_sub = np.where(np.in1d(U, sub_to_full))[0]


# # Mask off values which do not occur in feature_index.
# valid = np.in1d(sub_to_full, U).reshape(sub_to_full.shape, order='F')
# sub_to_full[~valid] = -1

# # Linear index
# sub_index = np.where(sub_to_full.flatten() >= 0)
