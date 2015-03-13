import sys
import copy
import bathymetry
from time import time

# Load linear algebra/science libraries.
import numpy as np
import multiprocessing as mp
from sklearn.grid_search import ParameterGrid

import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Load Gaussian Process library.
import GPy
from GPy.inference.latent_function_inference.expectation_propagation import EP
from GPy.inference.latent_function_inference.expectation_propagation_dtc import EPDTC


def chunk_index(length, number_chunks):
    """Yield index to array in chunks."""

    j = -1
    n = int(np.ceil(float(length) / number_chunks))
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


def _gpc_optimise(model, iterations, max_iterations, verbose):

    if verbose:
        print 'Optimising parameters for {0}'.format(model.name)

    # Perform several optimisation trials.
    for i in range(iterations):

        # Run approximation method (EP by default) and then optimise the kernel
        # parameters.
        model.optimize('bfgs', max_iters=max_iterations)

        if verbose:
            msg = '    iteration {0:>3n} of {1:>3n}, f = {2:>10.4f}'
            print msg.format(i + 1, iterations, model.log_likelihood())

    if verbose:
        print ''

    return model


def _gpc_predict(models, xs):
    """Function for training Gaussian process classifier."""

    # Pre-allocate memory.
    K = len(models)
    p = np.zeros((xs.shape[0], K))
    f = np.zeros((xs.shape[0], K))
    v = np.zeros((xs.shape[0], K))

    # Iterate through models.
    for i, model in enumerate(models):

        # Perform inference in the latent function space and then squash
        # the output through the likelihood function.
        f_i, v_i = model._raw_predict(xs)
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

    def __init__(self, X, Y, kernel=None, threads=None, sparse=False,
                 inducing=10, verbose=True):
        """Initialise the classification model."""

        # Store level of output.
        self.__verbose = verbose
        self.__threads = threads

        # Ensure the inputs and outputs are paired.
        if X.shape[0] != Y.shape[0]:
            raise Exception('The input and outputs must be the same length.')

        # Find unique set of labels in the data set.
        self.__labels = np.unique(Y.flatten())

        # Currently we cannot parallelise the sparse model.
        if sparse and (threads is not None):
            raise Exception('The sparse model cannot be parallelise.')

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

        # By default, use a squared exponential kernel for the classification
        # model.
        if kernel is None:
            kernel = GPy.kern.RBF(X.shape[1])

        # Create a Gaussian process classifier for each label in the dataset.
        self.__models = list()
        for i, label in enumerate(self.__labels):
            if self.__verbose:
                print 'Creating model for class {0}'.format(i + 1)

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

            # Name model.
            gpc.name = 'Class {0}'.format(str(self.__labels[i]))

            # Store model.
            self.__models.append(gpc)

    def optimise(self, iterations=10, max_iterations=100, verbose=True):
        """Optimise the kernel parameters."""

        # Iterate through models.
        t0 = time()

        # Perform optimisation on a SINGLE process.
        if not self.__threads:
            for i, model in enumerate(self.__models):
                self.__models[i] = _gpc_optimise(model,
                                                 iterations,
                                                 max_iterations,
                                                 self.__verbose)

        # Perform optimisation on MULTIPLE processes.
        else:
            pool = mp.Pool(processes=self.__threads)

            results = list()
            for i, model in enumerate(self.__models):

                args = (model, iterations, max_iterations, False)
                result = pool.apply_async(_gpc_optimise, args=args)
                results.append(result)

            # Prevent any more tasks from being submitted to the pool. Once all
            # the tasks have been completed the worker processes will
            # exit. Wait for the worker processes to exit.
            pool.close()
            pool.join()

            # Retrieve optimised models..
            for i, result in enumerate(results):
                self.__models[i] = results[i].get()

        if verbose:
            print 'Elapsed time {0:.3f}s'.format(time() - t0)

    def predict(self, xs, chunks=None, verbose=True):

        # Pre-allocate memory.
        t0 = time()
        K = len(self.__labels)
        p = np.zeros((xs.shape[0], K))
        f = np.zeros((xs.shape[0], K))
        v = np.zeros((xs.shape[0], K))

        # Perform inference on a SINGLE process.
        if not self.__threads:
            for i, idx in chunk_index(len(xs), chunks):
                if verbose:
                    msg = 'Completed {0:>5.1f}%'
                    print msg.format(float(i + 1) / chunks * 100)

                # Perform inference on chunk.
                p[idx, :], f[idx, :], v[idx, :] = _gpc_predict(self.__models,
                                                               xs[idx, :])

        # Perform inference on MULTIPLE processes.
        else:
            pool = mp.Pool(processes=self.__threads)

            results = list()
            for i, idx in chunk_index(len(xs), chunks):
                args = (self.__models, xs[idx, :],)
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

        if verbose:
            print '\nElapsed time {0:.3f}s'.format(time() - t0)

        return p, f, v


# --------------------------------------------------------------------------- #
#                             Plotting Functions
# --------------------------------------------------------------------------- #

def plot_probability(p, idx, shape, K, blending='flat', cmap=cm.hsv):
    """Plot most likely class or class probabilities."""

    # Get class assignment.
    c = np.argmax(p, axis=1)

    # Create colour map for plot.
    colours = cmap(np.linspace(0, 1, K))
    colours = colours[c, :]

    # Copy colour data into raster.
    rows, cols = np.unravel_index(idx, shape['shape'], order='F')
    raster = np.zeros(shape['shape'])[:, :, np.newaxis].repeat(4, axis=2)
    for i in range(3):
        raster[rows, cols, i] = colours[:, i]

    # Plot most likely class.
    if blending == 'flat':
        raster[rows, cols, -1] = 1

    # Plot most likely class, shaded according to probability.
    elif blending == 'to_colour':
        raster[rows, cols, -1] = np.max(p, axis=1)

    print raster.shape
    return plt.imshow(raster)
    # return bathymetry.plot_raster(raster, extent=shape['extent'])


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
        raster = np.nan * np.zeros(shape['size'])
        raster[idx] = V[:, i]
        raster = raster.reshape(shape['shape'], order='F')

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
        bathymetry.plot_raster(raster, extent=shape['extent'], ax=ax, **kwargs)
        axs.append(ax)

    return axs
