"""Rasterised bathymetry tools

The rasterised bathymetry module provides methods and objects designed to
manipulate bathymetry rasters.

The main function responsible for loading bathymetry files is:

    - :py:func:load_bathymetry

the following helper functions provide convenient method for opening and
displaying information about the bathymetry:

    - :py:func:load_bathymetry_file
    - :py:func:load_bathymetry_meta

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
        :class:`np.array`: Array like object containing pickled bathymetry
                           data.

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


def load_bathymetry_meta(bathymetry_path, verbose=True):
    """Load bathymetry meta-data from the disk.

    Loads bathymetry meta-data from a path WITHOUT loading depth
    information. :py:func:load_bathymetry_meta expects the data to be stored in
    BZ2 compressed pickle files in the following components:

        bathymetry_path/
            \____resolution.pkl.bz2
            \____x_bins.pkl.bz2
            \____y_bins.pkl.bz2

    where the .pkl.bz2 files:

        - resolution: Contains a single float specifying the (square) size of
                      each bathymetry pixel in metres.

        - x_bins: Contains a numpy vector storing the local easting, in metres,
                  for each column of bathymetry pixels

        - y_bins: Contains a numpy vector storing the local northing, in
                  metres, for each row of bathymetry pixels

    The bathymetry is returned as a dictionary containing the following key,
    value pairs:

        info = {'x_bins': array(N,),
                'y_bins': array(M,),
                'x_lim': [min(bathy['x_bins']), min(bathy['x_bins'])],
                'y_lim': [min(bathy['y_bins']), min(bathy['y_bins'])],
                'extent': [bathy['y_bins'][0] - bathy['resolution']/2,
                           bathy['y_bins'][1] + bathy['resolution']/2,
                           bathy['x_bins'][0] - bathy['resolution']/2,
                           bathy['x_bins'][1] + bathy['resolution']/2],
                'rows': M,
                'cols': N,
                'size': M * N,
                'resolution': float()}

    Args:
        bathymetry_path (str): Path to where bathymetry data is stored.
        verbose (bool): If set to True the contents of the bathymetry
                        information dictionary will be summarised on stdout.

    Returns:
        :class:`dict`: Dictionary object containing the bathymetry meta-data.

    Raises:
        Exception: If the a file could not be opened for an unexpected reason.
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

    # Store 'convenience' fields.
    bathy['y_lim'] = [bathy['x_bins'].min(), bathy['x_bins'].max()]
    bathy['x_lim'] = [bathy['y_bins'].min(), bathy['y_bins'].max()]
    bathy['rows'] = bathy['y_bins'].size
    bathy['cols'] = bathy['x_bins'].size
    bathy['size'] = bathy['rows'] * bathy['cols']

    # Consider X-bin and Y-bin values to mark the centre of bathymetry pixels.
    radius = bathy['resolution'] / 2.0
    bathy['extent'] = [bathy['y_lim'][0] - radius, bathy['y_lim'][1] + radius,
                       bathy['x_lim'][0] - radius, bathy['x_lim'][1] + radius]

    # Summarise bathymetry data.
    if verbose:
        print 'Bathymetry Summary:'
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


def load_bathymetry(bathymetry_path, invalid=np.nan, dtype=np.float,
                    verbose=True):
    """Load bathymetry raster from the disk.

    Loads bathymetry information from a path. :py:func:load_bathymetry expects
    the data to be stored in BZ2 compressed pickle files in the following
    components:

        bathymetry_path/
            \____depth.pkl.bz2
            \____index.pkl.bz2
            \____resolution.pkl.bz2
            \____x_bins.pkl.bz2
            \____y_bins.pkl.bz2

    where the .pkl.bz2 files:

        - depth: Contains a numpy vector storing valid bathymetric
                 data. Out-of-range or invalid returns are NOT stored. Each
                 element in the 'depth' vector corresponds to an element in the
                 index 'vector' - this position information is used to
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

    The bathymetry is returned as a dictionary containing the following key,
    value pairs:

        bathymetry = {'depth': array(M, N),
                      'x_bins': array(N,),
                      'y_bins': array(M,),
                      'x_lim': [min(bathy['x_bins']), min(bathy['x_bins'])],
                      'y_lim': [min(bathy['y_bins']), min(bathy['y_bins'])],
                      'extent': [bathy['y_bins'][0] - bathy['resolution']/2,
                                 bathy['y_bins'][1] + bathy['resolution']/2,
                                 bathy['x_bins'][0] - bathy['resolution']/2,
                                 bathy['x_bins'][1] + bathy['resolution']/2],
                      'rows': M,
                      'cols': N,
                      'size': M * N,
                      'resolution': float()}

    Args:
        bathymetry_path (str): Path to where bathymetry data is stored.
        invalid (value): Value to used to represent pixels where bathymetry is
                         not available.
        dtype (np.dtype): Data type used to store bathymetry raster.
        verbose (bool): If set to True the contents of the bathymetry
                        dictionary will be summarised on stdout.

    Returns:
        :class:`dict`: Dictionary object containing the bathymetry data.

    Raises:
        Exception: If the a file could not be opened for an unexpected reason.
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

    # Copy observed values into full raster. Note that the indices are stored
    # in column major order (Fortran-like).
    depth = invalid * np.ones((bathy['size']))
    depth[bathy['index']] = bathy['depth']
    bathy['depth'] = depth.reshape((bathy['rows'], bathy['cols']), order='F')
    del(bathy['index'])

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


def load_features(features, path_prefix, transform=False, verbose=True):
    """Compiles bathymetry features into a single feature vector"""

    # Define short hand notation for valid features.
    valid_features = ['D', 'A', 'R', 'S']

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
    feature_list = list()
    feature_name = list()
    feature_cache = dict()
    eps = np.finfo(float).eps
    for feat, scale in zip(feature_type, feature_scale):

        # Feature is depth.
        if feat == 'D':
            if verbose:
                print 'Selecting depth'
            feature_file = path_prefix + '.pkl'
            with open(feature_file, 'rb') as f:
                data = pickle.load(f)

            feature_name.append('Depth')
            data = data['Z']

        # Feature is aspect, rugosity or slope.
        else:
            feature_file = path_prefix + '_{0:03n}.pkl'.format(scale)
            if scale not in feature_cache:
                with open(feature_file, 'rb') as f:
                    feature_cache[scale] = pickle.load(f)

            # Insert aspect vector.
            if feat == 'A':
                feature_name.append('Aspect {0:>3n}'.format(scale))

                if verbose:
                    print 'Selecting aspect,   scale {0:>3n}'.format(scale)

                data = feature_cache[scale]['aspect']

            # Insert rugosity vector.
            elif feat == 'R':
                feature_name.append('Rugosity {0:>3n}'.format(scale))

                if verbose:
                    print 'Selecting rugosity, scale {0:>3n}'.format(scale)

                if transform:
                    data = np.log(feature_cache[scale]['rugosity'] - 1 + eps)
                else:
                    data = feature_cache[scale]['rugosity']

            # Insert slope vector.
            elif feat == 'S':
                feature_name.append('Slope {0:>3n}'.format(scale))

                if verbose:
                    print 'Selecting slope,    scale {0:>3n}'.format(scale)

                if transform:
                    data = np.log(feature_cache[scale]['slope'] + eps)
                else:
                    data = feature_cache[scale]['slope']

        # Store feature vector.
        feature_list.append(data.reshape((data.size, 1)))

    # Concatenate features into a single vector.
    features = np.concatenate(feature_list, axis=1)

    # Find rows which contain a full set of features (due to differences in
    # scale some rows may be incomplete).
    full_row = np.all(np.isnan(features) == False, axis=1)

    return features, feature_name, full_row


def cartesian_to_bathymetry(raster, easting, northing):

    # Convert AUV northings and eastings into a matrix subscripts.
    cols = np.floor((easting - min(raster.x_lim)) / raster.resolution)
    rows = np.floor((northing - min(raster.y_lim)) / raster.resolution)
    rows = raster.rows - rows - 1

    return rows.astype(int), cols.astype(int)


def label_bathymetry_pixels(bathymetry, easting, northing, classes):

    # Convert cartesian eastings/northings to bathymetry subscripts.
    rows, cols = cartesian_to_bathymetry(bathymetry, easting, northing)

    # Convert subscripts to index for 'flattened' bathymetry features.
    index = np.ravel_multi_index([rows, cols],
                                 dims=(bathymetry.rows, bathymetry.cols))

    # Find all bathymetric cells visited by AUV surveys.
    unique_index, idx, counts = np.unique(index,
                                          return_index=True,
                                          return_counts=True)

    # Find unique rows and columns.
    unique_rows = rows[idx]
    unique_cols = cols[idx]

    # Iterate through bathymetric cells with repeat observations and use the
    # most frequently observed class.
    C = classes[idx].flatten()
    for i in np.where(counts != 1)[0]:
        C[i] = int(mode(classes[index == unique_index[i]])[0][0])

    return unique_rows, unique_cols, unique_index, C


def subsample_features(subsample, bathymetry, valid):

    # Grid row and column subscripts.
    sub_rows = np.arange(0, bathymetry.rows, subsample, dtype=np.int)
    sub_cols = np.arange(0, bathymetry.cols, subsample, dtype=np.int)
    cols_grid, rows_grid = np.meshgrid(sub_cols, sub_rows)

    # Create linear index to full sized matrix/raster from sub-sampled
    # subscripts.
    numel = cols_grid.size
    full_index = np.ravel_multi_index([rows_grid.reshape((numel, 1)),
                                       cols_grid.reshape((numel, 1))],
                                      dims=(bathymetry.rows, bathymetry.cols))

    # Find valid indices in sub-sampled matrix.
    sub_index = np.where(valid[full_index])[0].flatten()

    # Find valid indices in full matrix.
    full_index = full_index[sub_index].flatten()

    # Use raster container to store size information.
    x_bins = bathymetry.x_bins[sub_cols]
    y_bins = bathymetry.y_bins[sub_rows]
    resolution = np.abs(x_bins[1] - x_bins[0])
    size = Raster(None, x_bins, y_bins, resolution)
    del(size.raster)

    # Return valid indices to full sized matrix, valid indices to sub-sampled
    # matrix and the shape of the sub-sampled matrix.
    return full_index, sub_index, size


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

# --------------------------------------------------------------------------- #
#                             Plotting Functions
# --------------------------------------------------------------------------- #


def plot_raster(raster, ax=None, extent=None, title=None, clabel=None,
                no_cbar=False, no_axis_label=False, no_ticks=False,
                horz_cbar=False, **kwargs):
    """Plot raster as an image.

    Args:
        raster (numpy.array): (MxN) raster to plot.
        ax (matplotlib.axes.Axes): Axes to plot raster. If set to `None`, a new
                                   axis will be created for plotting.
        extent (list): extent of plot [x_min, x_max, y_min, y_max] in Cartesian
                       space. If provided the X and Y labels will be labelled
                       automatically.
        title (str): Title of plot. If set to `None`, no title will be printed.
        clabel (str): Label of colour bar. If set to `None`, no colour bar
                      label will be rendered.
        no_cbar (bool): If set to `True` the colour bar will NOT be rendered.
                        If set to `False` the colour bar WILL be rendered.
        no_axis_label (bool): If set to `True` the X/Y labels will NOT be
                              rendered.  If set to `False` the X/Y labels WILL
                              be rendered.

        no_ticks (bool): If set to `True` the X/Y tick marks will NOT be
                              rendered.  If set to `False` the X/Y tick marks
                              WILL be rendered.
        horz_cbar (bool): If set to `True` the colour bar will rendered
                          horizontally below the figure. This option is best
                          used with `no_axis_label=True` and `no_ticks=True`.
        **kwargs (any): Are passed into matplotlib.pyplot.imshow when the
                        raster is rendered.

    Returns:
        :class:`matplotlib.axes.Axes`: axes containing raster image.

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


def plot_features(bathymetry, features, cols=1, titles=None, clabels=None,
                  **kwargs):
    """Plot bathymetry feature vectors."""

    # Determine the number of rows in the subplot.
    rows = np.ceil(float(features.shape[1]) / cols)

    # Iterate through feature vectors and plotting.
    axs = list()
    for i in range(features.shape[1]):

        # Create new subplot.
        ax = plt.subplot(rows, cols, i + 1)

        # Reshape feature vector for plotting.
        raster = features[:, i].reshape(bathymetry.shape)

        # Add titles.
        if titles is not None:
            kwargs['title'] = titles[i]

        # Add colour bar.
        if clabels is not None:
            # Hack to try and provide some compatibility between python 2.7 and
            # python 3.0
            if type(clabels) == type(''):
                kwargs['clabel'] = ''
            else:
                kwargs['clabel'] = clabels[i]

        # Plot raster.
        plot_raster(raster, bathymetry.limits, ax=ax, **kwargs)
        axs.append(ax)

    return axs


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
