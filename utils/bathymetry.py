import pickle
import numpy as np
from scipy.stats import mode
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Raster(object):
    """Container for raster data."""

    def __init__(self, raster, x_bins, y_bins, resolution):

        self.raster = raster
        self.x_bins = x_bins.flatten()
        self.y_bins = y_bins.flatten()
        self.resolution = resolution

    @property
    def shape(self):
        return (self.rows, self.cols)

    @property
    def rows(self):
        return self.y_bins.size

    @property
    def cols(self):
        return self.x_bins.size

    @property
    def numel(self):
        return self.rows * self.cols

    @property
    def x_lim(self):
        return [self.x_bins.min(), self.x_bins.max()]

    @property
    def y_lim(self):
        return [self.y_bins.min(), self.y_bins.max()]

    @property
    def limits(self):
        r = self.resolution / 2.0
        return [self.x_lim[0] - r, self.x_lim[1] + r,
                self.y_lim[0] - r, self.y_lim[1] + r]


def load_bathymetry(bathymetry_file, verbose=True):
    """Load bathymetry data from pickle object."""

    # Load bathymetry dictionary from pickle file.
    with open(bathymetry_file, 'rb') as f:
        bathymetry = pickle.load(f)

    # Convert dictionary to object.
    bathymetry = Raster(bathymetry['Z'],
                        bathymetry['x_bins'],
                        bathymetry['y_bins'],
                        bathymetry['resolution'])

    # Summarise bathymetry data.
    if verbose:
        print 'Bathymetry Summary:'
        print '    Z:           [{0[0]}x{0[1]}]'.format(bathymetry.shape)
        print '    X-bins:      [{0[0]},]'.format(bathymetry.x_bins.shape)
        print '    Y-bins:      [{0[0]},]'.format(bathymetry.y_bins.shape)
        print '    X-lim:       [{0[0]}, {0[1]}]'.format(bathymetry.x_lim)
        print '    Y-lim:       [{0[0]}, {0[1]}]'.format(bathymetry.y_lim)
        print '    rows:        {0}'.format(bathymetry.rows)
        print '    cols:        {0}'.format(bathymetry.cols)
        print '    numel:       {0}'.format(bathymetry.numel)
        print '    resolution:  {0}'.format(bathymetry.resolution)

    return bathymetry


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


def plot_raster(raster, limits, ax=None, title=None, clabel=None,
                no_label=False, no_ticks=False, horz_cbar=False,
                **kwargs):
    """Plot raster."""

    # Create axes if none is provided.
    if ax is None:
        ax = plt.subplot(111)

    # Plot raster.
    im = ax.imshow(raster, extent=limits, **kwargs)

    # Assign title to image.
    if title:
        plt.title(title)

    # Label axes.
    plt.grid('on')
    if not no_label:
        plt.xlabel('Local Easting (m)')
        plt.ylabel('Local Northing (m)')

    # Remove tick marks.
    if no_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Create colour bar.
    if clabel is not None:

        # Vertical colour bar.
        if not horz_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(clabel, rotation=90)

        # Horizontal colour bar.
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
            cbar.set_label('', rotation=0)

    # Force limits of plot.
    ax.axis(limits)

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
