import sys
import copy
import pickle
from time import time

# Load linear algebra/science libraries.
import scipy
import numpy as np
import multiprocessing as mp
from scipy.stats import mode
from sklearn.grid_search import ParameterGrid

# Load plotting libraries.
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load Gaussian Process library.
import GPy
from GPy.inference.latent_function_inference.expectation_propagation import EP
from GPy.inference.latent_function_inference.expectation_propagation_dtc import EPDTC


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


def load_raster(raster_file, verbose=True):
    """Load raster data from pickle object."""

    # Load raster dictionary from pickle file.
    with open(raster_file, 'rb') as f:
        raster = pickle.load(f)

    # Convert dictionary to object.
    raster = Raster(raster['Z'],
                    raster['x_bins'],
                    raster['y_bins'],
                    raster['resolution'])

    # Summarise raster data.
    if verbose:
        print 'Raster Summary:'
        print '    Z:           [{0[0]}x{0[1]}]'.format(raster.shape)
        print '    X-bins:      [{0[0]},]'.format(raster.x_bins.shape)
        print '    Y-bins:      [{0[0]},]'.format(raster.y_bins.shape)
        print '    X-lim:       [{0[0]}, {0[1]}]'.format(raster.x_lim)
        print '    Y-lim:       [{0[0]}, {0[1]}]'.format(raster.y_lim)
        print '    rows:        {0}'.format(raster.rows)
        print '    cols:        {0}'.format(raster.cols)
        print '    numel:       {0}'.format(raster.numel)
        print '    resolution:  {0}'.format(raster.resolution)

    return raster


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


def plot_survey_template(template):
    """Plot survey template"""

    trajectory, = plt.plot(template[:, 0], template[:, 1],
                           'b.', label='Trajectory')

    start, = plt.plot(template[0, 0], template[0, 1],
                      'g.', markersize=20,
                      label='Starting location')

    stop, = plt.plot(template[-1, 0], template[-1, 1],
                     'r.', markersize=20,
                     label='Terminal location')

    # Specify format of legend.
    plt.legend(handles=[start, trajectory, stop],
               handler_map={start: HandlerLine2D(numpoints=1),
                            trajectory: HandlerLine2D(numpoints=1),
                            stop: HandlerLine2D(numpoints=1)})

    # Label axes.
    plt.xlabel('Local Easting (m)')
    plt.ylabel('Local Northing (m)')
    plt.grid()


def discretise_survey_template(template, resolution):
    """Discretise survey template into raster 'grid-world'."""

    # Ensure the template is discretised to the raster grid world.
    discretised = np.round(template / resolution).astype(int)

    # Find unique pixels preserving order (pretty inefficient method).
    unique_location = list()
    for row in discretised.tolist():
        if row not in unique_location:
            unique_location.append(row)

    # Convert back to a numpy array containing the template represented as
    # (row, col) index to pixels in the raster.
    return np.array(unique_location, dtype=np.int)


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


def grid_survey_origins(surveys, feasible, raster):
    """Create a grid of survey origins."""

    # Allocate the same number of surveys to each axis.
    axis_surveys = int(np.sqrt(surveys))

    # Increase number of surveys per axis until the total number of surveys
    # breaches the specified limit.
    while True:
        x_org = np.linspace(0, raster.cols - 1, axis_surveys)
        y_org = np.linspace(0, raster.rows - 1, axis_surveys)
        x_org, y_org = np.meshgrid(x_org.astype(int), y_org.astype(int))
        x_org = x_org.flatten()
        y_org = y_org.flatten()
        valid = feasible[y_org, x_org]

        if valid.flatten().sum() >= surveys:
            break
        else:
            axis_surveys += 1

    # Return origins as a [Nx2] matrix.
    return np.vstack((raster.x_bins[x_org[valid]],
                      raster.y_bins[y_org[valid]])).T


def plot_survey_origins(origins, raster, limits, *args, **kwargs):
    """Plot survey origins."""

    ax = plot_raster(raster, limits, cmap=cm.gray)
    ax.plot(origins[:, 0], origins[:, 1], *args, **kwargs)
    ax.axis(limits)


def pixels_observed_by_template(x, y, template, raster):
    """Return pixels covered by a (discrete) template placed at an origin."""

    # Ensure the template is specified as integers.
    if not np.issubdtype(template.dtype, np.integer):
        msg = 'The survey template must be an [Nx2] numpy array of integers'
        msg += ' indexing visited locations in the raster.'
        raise Exception(msg)

    # Convert the Cartesian X/Y origin to raster subscripts.
    x_idx = int((x - raster.x_bins.min()) / raster.resolution)
    y_idx = int((y - raster.y_bins.min()) / raster.resolution)

    # Create template origin.
    origin = np.array([[x_idx, y_idx]], dtype=np.integer)

    # Shift template.
    template = origin + template
    template[:, 1] = raster.rows - template[:, 1] - 1

    return template


def plot_survey_utility(origins, utility, raster, limits, *args, **kwargs):
    """Plot survey utility."""

    ax = plot_raster(raster, limits, cmap=cm.gray)
    sc = ax.scatter(origins[:, 0], origins[:, 1], c=utility, *args, **kwargs)
    ax.axis(limits)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label('Utility', rotation=90)


def chunk_index(length, number_chunks):
    """Yield index to array in chunks."""

    j = -1
    n = int(length / number_chunks)
    for i in xrange(0, length, n):
        j += 1
        if (i + n) <= length:
            idx = range(i, i + n)
        else:
            idx = range(i, length)
        yield j, idx


def gp_train_grid(model, param_grid, verbose=True):
    """Optimise GP hyper-parameters from a grid of starting locations."""

    t0 = time()
    if verbose:
        print 'Optimising hyper-parameters:'

    # Iterate through grid of hyper-parameters.
    optimal = np.inf
    grid = ParameterGrid(param_grid)
    for i, params in enumerate(grid):

        # Create new initial starting conditions.
        string = '    {0:>3n} of {1:>3n}    '.format(i + 1, len(grid))
        for key, value in params.items():
            model.kern[key] = value
            string += '{0}: {1:>8.3f}, '.format(key, value)

        if verbose:
            sys.stdout.write(string)

        # Maximise marginal likelihood with respect to the kernel
        # hyperparameters.
        model.optimize(max_f_eval=10000, messages=False)

        # Store most highly-performing model.
        log_likelihood = -model.log_likelihood()
        if log_likelihood < optimal:
            optimal = log_likelihood
            utility_model = copy.deepcopy(model)

        if verbose:
            print 'f = {0:>10.3f}'.format(-log_likelihood)

    if verbose:
        print '\nElapsed time {0:.3f}s'.format(time() - t0)

    return utility_model


def gpr_chunk_inference(chunks, xs, model, verbose=True):
    """Perform Gaussian process regression in chunks."""

    # Pre-allocate memory for inferred mean and variance.
    f_xs = np.zeros((len(xs), 1))
    V_xs = np.zeros((len(xs), 1))

    t0 = time()
    if verbose:
        print 'Performing inference:'

    # Iterate through the data in chunks performing inference.
    for i, idx in chunk_index(len(xs), chunks):
        if verbose:
            print '   completed {0:>5.1f}%'.format(float(i) / chunks * 100)
        f_xs[idx, :], V_xs[idx, :] = model.predict(xs[idx, :])

    if verbose:
        print '\nElapsed time {0:.3f}s'.format(time() - t0)

    return f_xs, V_xs


# ----------------------------------------------------------------------------
#                               Habitat Mapping
# ----------------------------------------------------------------------------

def load_feature(feature_file, verbose=True):
    """Load bathymetry features."""

    # Load raster dictionary from pickle file.
    with open(feature_file, 'rb') as f:
        features = pickle.load(f)

    # Convert dictionary to object.
    feature = Raster(None,
                     features['x_bins'],
                     features['y_bins'],
                     features['resolution'])

    # Add feature arrays.
    feature.aspect = features['aspect']
    feature.rugosity = features['rugosity']
    feature.slope = features['slope']
    feature.processed = features['processed']
    del(feature.raster)

    # Summarise feature data.
    if verbose:
        print 'Feature Summary:'
        print '    aspect:      [{0[0]}x{0[1]}]'.format(feature.aspect.shape)
        print '    rugosity:    [{0[0]}x{0[1]}]'.format(feature.rugosity.shape)
        print '    slope:       [{0[0]}x{0[1]}]'.format(feature.slope.shape)
        print '    processed:   [{0[0]}x{0[1]}]'.format(feature.processed.shape)
        print '    X-bins:      [{0[0]},]'.format(feature.x_bins.shape)
        print '    Y-bins:      [{0[0]},]'.format(feature.y_bins.shape)
        print '    X-lim:       [{0[0]}, {0[1]}]'.format(feature.x_lim)
        print '    Y-lim:       [{0[0]}, {0[1]}]'.format(feature.y_lim)
        print '    rows:        {0}'.format(feature.rows)
        print '    cols:        {0}'.format(feature.cols)
        print '    numel:       {0}'.format(feature.numel)
        print '    resolution:  {0}'.format(feature.resolution)

    return feature


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


def load_AUV_classes(AUV_file, verbose=True):
    """Load AUV classification of seabed."""

    # Load raster dictionary from pickle file.
    with open(AUV_file, 'rb') as f:
        AUV = pickle.load(f)

    # Determine the number of classes.
    AUV['K'] = len(np.unique(AUV['class']))

    # Summarise AUV data.
    if verbose:
        print 'Classification Summary:'
        print '    easting:     [{0[0]}x{0[1]}]'.format(AUV['easting'].shape)
        print '    northing:    [{0[0]}x{0[1]}]'.format(AUV['northing'].shape)
        print '    probability: [{0[0]}x{0[1]}]'.format(AUV['probability'].shape)
        print '    class:       [{0[0]}x{0[1]}]'.format(AUV['class'].shape)
        print '    K:           {0}'.format(AUV['K'])

    return AUV


def plot_AUV_classes(raster, limits,
                     easting, northing, classes, K,
                     subsample=1, cmap=cm.hsv,
                     **kwargs):
    """Plot AUV classes on top of bathymetry."""

    easting = easting[::subsample]
    northing = northing[::subsample]
    classes = classes[::subsample]

    scax = list()
    sctitle = list()
    colours = cmap(np.linspace(0, 1, K))
    ax = plot_raster(raster, limits, cmap=cm.bone, clabel='depth (m)')
    for i in range(K):
        idx = (classes == (i + 1))
        sctitle.append('Class {0}'.format(i + 1))
        scax.append(ax.scatter(easting[idx], northing[idx], s=15,
                               c=colours[i, :], lw=0))

    ax.legend(scax, sctitle, scatterpoints=1, **kwargs)

    return ax


def cartesian_to_bathymetry(raster, easting, northing):

    # Convert AUV northings and eastings into a matrix subscripts.
    cols = np.floor((easting - min(raster.x_lim)) / raster.resolution)
    rows = np.floor((northing - min(raster.y_lim)) / raster.resolution)
    rows = raster.rows - rows - 1

    return rows.astype(int), cols.astype(int)


def label_bathymetry_pixels(bathymetry, AUV):

    # Convert cartesian eastings/northings to bathymetry subscripts.
    rows, cols = cartesian_to_bathymetry(bathymetry,
                                         AUV['easting'],
                                         AUV['northing'])

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
    classes = AUV['class'][idx].flatten()
    for i in np.where(counts != 1)[0]:
        classes[i] = int(mode(AUV['class'][index == unique_index[i]])[0][0])

    return unique_rows, unique_cols, unique_index, classes


def _gpc_predict(models, xs, verbose=False, **kwargs):

    # Pre-allocate memory.
    K = len(models)
    p = np.zeros((xs.shape[0], K))
    f = np.zeros((xs.shape[0], K))
    v = np.zeros((xs.shape[0], K))

    # Iterate through models.
    for i, model in enumerate(models):
        if verbose:
            print 'Predicting using model {0}'.format(i + 1)

        # Perform inference in the latent function space and then squash
        # the output through the likelihood function.
        f_i, v_i = model._raw_predict(xs, **kwargs)
        p_i = model.likelihood.predictive_values(f_i, v_i)[0]

        # The output of GPy inference are [Nx1] arrays. Numpy's
        # broadcasting cannot be used in this circumstance. Flatten the
        # arrays to allow column assignment.
        f[:, i] = f_i.flatten()
        v[:, i] = v_i.flatten()
        p[:, i] = p_i.flatten()

    # Replace rows of zeros (unlikely) with uncertainty.
    p[np.all(p == 0, axis=1), :] = 1.0 / float(K)

    # Normalise one-vs-all probabilities.
    p = (p.T / np.sum(p, axis=1)).T

    return p, f, v


class GPClassification(object):
    """Create an all-vs-one Gaussian process classifier."""

    def __init__(self, X, Y, kernel=None, sparse=False, inducing=10,
                 verbose=True):
        """Initialise the classification model."""

        # Store level of output.
        self.__verbose = verbose

        # Ensure the inputs and outputs are paired.
        if X.shape[0] != Y.shape[0]:
            raise Exception('The input and outputs must be the same length.')

        # By default, use a squared exponential kernel for the classification
        # model.
        if kernel is None:
            kernel = GPy.kern.RBF(X.shape[1])

        # Find unique set of labels in the data set.
        self.__labels = np.unique(Y.flatten())

        # If the model is SPARSE create inducing points.
        if sparse:

            # There are less inducing points than data points, use full model.
            num_inducing = inducing * X.shape[1]
            if X.shape[0] < num_inducing:
                sparse = False

            # Create set of inducing points by linearly interpolating the
            # dataset.
            else:
                idx = np.linspace(0, len(X)-1, num_inducing).astype(int)
                Z = X[np.unique(idx), :].copy()

        # Create a Gaussian process classifier for each label in the dataset.
        self.__models = list()
        for i, label in enumerate(self.__labels):
            if self.__verbose:
                print 'Creating model for label {0}'.format(label)

            # Perform all-vs-one classification for the current class.
            c = (Y == label).astype(int)

            # Create Gaussian process classifier for current class.
            if not sparse:
                gpc = GPy.core.GP(X=X,
                                  Y=c,
                                  kernel=copy.deepcopy(kernel),
                                  inference_method=EP(),
                                  likelihood=GPy.likelihoods.Bernoulli())

            # Create SPARSE Gaussian process classifier for current class.
            else:
                gpc = GPy.core.SparseGP(X, c, Z,
                                        kernel=copy.deepcopy(kernel),
                                        likelihood=GPy.likelihoods.Bernoulli(),
                                        inference_method=EPDTC())

                # Attempt to optimise the location of the inducing inputs.
                gpc.Z.unconstrain()

            # Store model.
            self.__models.append(gpc)

    def optimise(self, iterations=10, max_iters=100):
        """Optimise the kernel parameters."""

        # Iterate through models.
        t0 = time()
        for i, model in enumerate(self.__models):
            if self.__verbose:
                msg = 'Optimising parameters for label {0}'
                print msg.format(self.__labels[i])

            # perform several optimisation trials.
            for i in range(iterations):
                if self.__verbose:
                    msg = '    iteration {0:>3n} of {1:>3n}, '
                    sys.stdout.write(msg.format(i + 1, iterations))

                # Run approximation method (EP by default) and then optimise
                # the kernel parameters.
                model.optimize('bfgs', max_iters=max_iters)

                if self.__verbose:
                    print 'f = {0:>10.3f}'.format(model.log_likelihood())

            if self.__verbose:
                sys.stdout.write('\n')

        if self.__verbose:
            print '\nElapsed time {0:.3f}s'.format(time() - t0)

    def __predict_chunked(self, xs, chunks, parallel=False, verbose=False,
                          **kwargs):

        # Pre-allocate memory.
        K = len(self.__labels)
        p = np.zeros((xs.shape[0], K))
        f = np.zeros((xs.shape[0], K))
        v = np.zeros((xs.shape[0], K))

        if not parallel:
            for i, idx in chunk_index(len(xs), chunks):
                if verbose:
                    msg = '   completed {0:>5.1f}%'
                    print msg.format(float(i) / chunks * 100)

                # Perform inference on chunk.
                p[idx, :], f[idx, :], v[idx, :] = _gpc_predict(self.__models,
                                                               xs,
                                                               verbose=False)

        else:
            pool = mp.Pool(processes=parallel)
            print 'pool started'

            results = list()
            for i, idx in chunk_index(len(xs), chunks):
                print 'Starting thread %i' % i
                args = (self.__models, xs[idx, :], False,)
                result = pool.apply_async(_gpc_predict, args=args)
                results.append(result)

            # Prevent any more tasks from being submitted to the pool. Once all
            # the tasks have been completed the worker processes will
            # exit. Wait for the worker processes to exit.
            pool.close()
            pool.join()

            # Reconstruct output.
            for i, idx in chunk_index(len(xs), chunks):
                p[idx, :], f[idx, :], v[idx, :] = results[i].get()

        if self.__verbose:
            sys.stdout.write('\n')

        return p, f, v

    def predict(self, xs, chunks=None, parallel=None):

        # Perform inference in one block.
        t0 = time()
        if not chunks:
            p, f, v = self.__predict(xs, verbose=self.__verbose)

        # Iterate through the data in chunks performing inference.
        else:
            p, f, v = self.__predict_chunked(xs, chunks, parallel=parallel,
                                             verbose=self.__verbose)

        if self.__verbose:
            print '\nElapsed time {0:.3f}s'.format(time() - t0)

        return p, f, v


def subsample_features(subsample, bathymetry, valid):

    # Grid row and column subscripts.
    sub_rows = np.arange(0, bathymetry.shape[0], subsample, dtype=np.int)
    sub_cols = np.arange(0, bathymetry.shape[1], subsample, dtype=np.int)
    cols, rows = np.meshgrid(sub_cols, sub_rows)

    # Create linear index to full sized matrix/raster from sub-sampled
    # subscripts.
    full_index = np.ravel_multi_index([rows.reshape((rows.size, 1)),
                                       cols.reshape((cols.size, 1))],
                                      dims=(bathymetry.shape[0],
                                            bathymetry.shape[1]))

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


def plot_probability(p, idx, shape, K, blending='flat', cmap=cm.hsv):
    """Plot most likely class or class probabilities."""

    # Get class assignment.
    c = np.argmax(p, axis=1)

    # Create colour map for plot.
    colours = cmap(np.linspace(0, 1, K))
    colours = colours[c, :]

    # Copy colour data into raster.
    rows, cols = np.unravel_index(idx, shape.shape)
    raster = np.zeros(shape.shape)[:, :, np.newaxis].repeat(4, axis=2)
    for i in range(3):
        raster[rows, cols, i] = colours[:, i]

    # Plot most likely class.
    if blending == 'flat':
        raster[rows, cols, -1] = 1

    # Plot most likely class, shaded according to probability.
    elif blending == 'to_colour':
        raster[rows, cols, -1] = np.max(p, axis=1)

    return plot_raster(raster, shape.limits)


def plot_variance(V, idx, shape, K, cols=1, titles=None, clabels=None,
                  **kwargs):
    """Plot predicted variance."""

    # Determine the number of rows in the subplot.
    rows = np.ceil(float(V.shape[1]) / cols)

    # Iterate through feature vectors and plotting.
    axs = list()
    for i in range(V.shape[1]):

        # Create new subplot.
        ax = plt.subplot(rows, cols, i + 1)

        # Reshape feature vector for plotting.
        raster = np.nan * np.zeros(shape.numel)
        raster[idx] = V[:, i]
        raster = raster.reshape(shape.shape)

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
        plot_raster(raster, shape.limits, ax=ax, **kwargs)
        axs.append(ax)

    return axs
