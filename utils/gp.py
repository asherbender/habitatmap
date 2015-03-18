import os
import sys
import copy
import time
import bathymetry
import parallelisation


# Load linear algebra/science libraries.
import scipy
import numpy as np
from sklearn.grid_search import ParameterGrid

import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Load Gaussian Process library.
import GPy
from GPy.inference.latent_function_inference.expectation_propagation import EP
from GPy.inference.latent_function_inference.expectation_propagation_dtc import EPDTC

import pyGPs


# Define small floating-point accuracy
EPS = np.spacing(1)


def block_index(length, number_chunks):
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


def gpr_chunk_inference(chunks, xs, model, verbose=True):
    """Perform Gaussian process regression in chunks."""

    # Pre-allocate memory for inferred mean and variance.
    f_xs = np.zeros((len(xs), 1))
    V_xs = np.zeros((len(xs), 1))

    t0 = time.time()
    if verbose:
        print 'Performing inference:'

    # Iterate through the data in chunks performing inference.
    for i, idx in block_index(len(xs), chunks):
        if verbose:
            print '   completed {0:>5.1f}%'.format(float(i) / chunks * 100)
        f_xs[idx, :], V_xs[idx, :] = model.predict(xs[idx, :])

    if verbose:
        print '\nElapsed time {0:.3f}s'.format(time.time() - t0)

    return f_xs, V_xs


# --------------------------------------------------------------------------- #
#                                      GPR
# --------------------------------------------------------------------------- #

class Gpr(object):
    """Gaussian Process Regression.

    Args:
      model (class):
      X (np.array):
      Y (np.array):
      kernel (GPy.kernel, optional):
      Z (np.array, optional):

    Attributes:
      X (np.array):
      Y (np.array):

    """

    def __init__(self, X, Y, kernel=None, Z=None):

        # Create default kernel.
        if kernel is None:
            kernel = GPy.kern.RBF(X.shape[1])

        # Use full model by default.
        sparse = False

        # Use sparse model.
        if Z is not None:

            # Create set of inducing points by linearly interpolating the
            # dataset.
            if isinstance(Z, (int, long)):
                num_inducing = Z

                if num_inducing < len(X):
                    idx = np.linspace(0, len(X)-1, num_inducing).astype(int)
                    Z = X[np.unique(idx), :]
                    sparse = True

            # Use input array of inducing points.
            else:
                if Z.shape[1] != X.shape[1]:
                    msg = 'The inducing points Z must have the same '
                    msg += 'number of columns as the inputs X.'
                    raise Exception(msg)

                elif Z.shape[0] >= X.shape[0]:
                    msg = 'There must be less inducing points Z than inputs X.'
                    raise Exception(msg)

                else:
                    sparse = True

        # Use full model.
        if not sparse:
            self.__gpr = GPy.models.GPRegression(X, Y, kernel=kernel)

        # Use sparse model.
        else:
            self.__gpr = GPy.models.SparseGPRegression(X, Y, kernel=kernel,
                                                       Z=Z)

    @property
    def X(self):
        """Training inputs."""
        return self.__gpr.X

    @X.setter
    def X(self, value):
        self.__gpr.X = value

    @property
    def Y(self):
        """Training outputs."""
        return self.__gpr.Y

    @Y.setter
    def Y(self, value):
        self.__gpr.Y = value

    @property
    def kernel(self):
        """Training outputs."""
        return self.__gpr.kern

    @kernel.setter
    def kernel(self, value):
        self.__gpr.kernel = value

    def optimise(self, param_grid=None, verbose=True):
        """Optimise hyper-parameters of Gaussian process."""

        # Perform a small sleep to allow output to stdout - mainly for parallel
        # monitoring parallel tasks.
        time.sleep(0.1)
        if param_grid is None:
            self.__gpr.optimize('bfgs')
        else:
            self.__grid_optimise(param_grid, verbose)

        return self

    def __grid_optimise(self, param_grid, verbose=True):
        """Optimise GP hyper-parameters from a grid of starting locations."""

        t0 = time.time()
        if verbose:
            print 'Optimising hyper-parameters:'

        # Iterate through grid of hyper-parameters.
        optimal = np.inf
        grid = ParameterGrid(param_grid)
        for i, params in enumerate(grid):
            model = copy.deepcopy(self.__gpr)

            # Create new initial starting conditions.
            string = '    {0:>3n} of {1:>3n}    '.format(i + 1, len(grid))
            for key, value in params.items():
                model.kern[key] = value
                string += '{0}: {1:>8.3f}, '.format(key, value)

            if verbose:
                sys.stdout.write(string)

            # Maximise marginal likelihood with respect to the kernel
            # hyper-parameters.
            model.optimize()

            # Store most highly-performing model.
            log_likelihood = -model.log_likelihood()
            if log_likelihood < optimal:
                optimal = log_likelihood
                optimal_model = model

            if verbose:
                print 'f = {0:>10.3f}'.format(-log_likelihood)
                print model

        # Save best model.
        self.__gpr = optimal_model

        if verbose:
            print '\nElapsed time {0:.3f}s'.format(time.time() - t0)

    def log_likelihood(self):
        """Return the log-likelihood of the model."""

        return self.__gpr.log_likelihood()

    def predict(self, xs):
        """Perform inference in Gaussian process."""

        # Perform inference.
        m, v = self.__gpr.predict(xs)
        return m.flatten(), v.flatten()


# --------------------------------------------------------------------------- #
#                                    PLSC
# --------------------------------------------------------------------------- #

def cumgauss_sigmoid(mu, sigma, alpha, beta):

    # Note: The normal cumulative distribution function only differs
    #       from the error function by scaling and translation. With:
    #
    #           Cumulative gaussian = norm.cdf(t) = 0.5 * erfc(-x/sqrt(2))
    #
    #       See:
    #           http://en.wikipedia.org/wiki/Error_function#Related_functions

    # Equation 6.42 (pg 148)
    t = (alpha * mu + beta) / (np.sqrt(1 + sigma * alpha**2))
    return 0.5 * scipy.special.erfc(-t / np.sqrt(2))


class Plsc(Gpr):
    """Probabilistic Least-Squares Classification.

    Args:
      model (class):
      X (np.array):
      Y (np.array):
      kernel (GPy.kernel, optional):
      Z (np.array, optional):
      sigmoid (callable, optional):
      alpha (float, optional):
      beta (float, optional):

    Attributes:
      X (np.array):
      Y (np.array):

    """

    def __init__(self, X, Y, kernel=None, Z=None, sigmoid=None, alpha=1.0,
                 beta=1.0):

        # Transform Y from [0, 1] to [-1, 1].
        Y = 2 * Y - 1.

        # Initialise Gaussian process regression model.
        super(Plsc, self).__init__(X, Y, kernel=kernel, Z=Z)

        # Create default sigmoid function.
        if sigmoid is None:
            self.__sigmoid = cumgauss_sigmoid

        # Store sigmoid function hyper-parameters
        self.__alpha = alpha
        self.__beta = beta

    def __optimise_sigmoid(self, params, y, mu, sigma):
        """Minimise negative sum of log probabilities."""

        # Wrapper function - swap input order for minimize.
        alpha, beta = np.abs(params[0]), params[1]
        p = self.__sigmoid(mu, sigma, alpha, beta)

        # Ensure probabilities represent their own class.
        # (i.e. p(-1) = 1 - p(1))
        p = np.abs(p - 0.5) + 0.5

        # Minimise log probabilities. Equation 5.11 (pg 116).
        log_LOO = -np.sum(np.log(p + EPS))
        if np.isnan(log_LOO):
            return np.inf
        else:
            return log_LOO

    def optimise(self, kernel_grid=None, sigmoid_grid=None, verbose=True):
        """Optimise hyper-parameters of Gaussian process."""

        # Optimise (kernel) parameters of Gaussian process regression.
        super(Plsc, self).optimise(kernel_grid, verbose=verbose)

        # Calculate the LOO-CV predictive mean and variance (eq 5.12, pg 117).
        Kinv = np.linalg.inv(self.kernel.K(self.X))
        mu_i = np.diag(self.Y - Kinv.dot(self.Y) / Kinv)
        sigma_i = 1 / np.diag(Kinv)

        # Optimise (kernel) parameters of Gaussian process regression.
        if sigmoid_grid is None:
            solution = scipy.optimize.minimize(self.__optimise_sigmoid,
                                               (self.__alpha, self.__beta),
                                               (self.Y, mu_i, sigma_i))

        else:
            optimal = np.inf
            grid = ParameterGrid(sigmoid_grid)
            for i, params in enumerate(grid):
                string = '    {0:>3n} of {1:>3n}    '.format(i + 1, len(grid))
                for key, value in params.items():
                    string += '{0}: {1:>8.3f}, '.format(key, value)

                if verbose:
                    sys.stdout.write(string)

                iteration = scipy.optimize.minimize(self.__optimise_sigmoid,
                                                    (params['alpha'],
                                                     params['beta']),
                                                    (self.Y, mu_i, sigma_i))

                if verbose:
                    print 'f = {0}'.format(iteration.fun)

                if iteration.fun < optimal:
                    optimal = iteration.fun
                    solution = iteration

        self.__alpha, self.__beta = solution.x
        return self

    def predict(self, xs):
        """Perform inference in Gaussian process."""

        # Perform prediction in 'latent' space (prior to squashing).
        mu, sigma = super(Plsc, self).predict(xs)

        # Squash output through sigmoid to approximate probabilities.
        p = self.__sigmoid(mu, sigma, self.__alpha, self.__beta)

        return p, mu, sigma

# --------------------------------------------------------------------------- #
#                                   EP-GPC
# --------------------------------------------------------------------------- #


class EpGpc(object):
    """Create an all-vs-one Gaussian process classifier."""

    def __init__(self, X, Y, kernel=None, Z=None):
        """Initialise the classification model."""

        # Create default kernel.
        if kernel is None:
            kernel = GPy.kern.RBF(X.shape[1])

        # Use full model by default.
        sparse = False

        # Use sparse model.
        if Z is not None:

            # Create set of inducing points by linearly interpolating the
            # dataset.
            if isinstance(Z, (int, long)):
                num_inducing = Z

                if num_inducing < len(X):
                    idx = np.linspace(0, len(X)-1, num_inducing).astype(int)
                    Z = X[np.unique(idx), :]
                    sparse = True

            # Use input array of inducing points.
            else:
                if Z.shape[1] != X.shape[1]:
                    msg = 'The inducing points Z must have the same '
                    msg += 'number of columns as the inputs X.'
                    raise Exception(msg)

                elif Z.shape[0] >= X.shape[0]:
                    msg = 'There must be less inducing points Z than inputs X.'
                    raise Exception(msg)

                else:
                    sparse = True

        # Use full model.
        if not sparse:
            self.__gpc = GPy.core.GP(X=X,
                                     Y=Y,
                                     kernel=copy.deepcopy(kernel),
                                     inference_method=EP(),
                                     likelihood=GPy.likelihoods.Bernoulli())

        # Use sparse model.
        else:
            self.__gpc = GPy.core.SparseGP(X, Y, Z,
                                           kernel=copy.deepcopy(kernel),
                                           likelihood=GPy.likelihoods.Bernoulli(),
                                           inference_method=EPDTC())

            # Attempt to optimise the location of the inducing inputs.
            self.__gpc.Z.unconstrain()

    @property
    def X(self):
        """Training inputs."""
        return self.__gpc.X

    @X.setter
    def X(self, value):
        self.__gpc.X = value

    @property
    def Y(self):
        """Training outputs."""
        return self.__gpc.Y

    @Y.setter
    def Y(self, value):
        self.__gpc.Y = value

    @property
    def kernel(self):
        """Training outputs."""
        return self.__gpc.kern

    @kernel.setter
    def kernel(self, value):
        self.__gpc.kernel = value

    def optimise(self, iterations=10, max_iterations=100, verbose=True):
        """Optimise hyper-parameters of Gaussian process."""

        # Retrieve process ID.
        pid = os.getpid()

        # Perform several optimisation trials.
        for i in range(iterations):

            # Run approximation method (EP by default) and then optimise the
            # kernel parameters.
            self.__gpc.optimize('bfgs', max_iters=max_iterations)

            if verbose:
                msg = 'PID {0:n}: {1}, iteration {2:>3n} of {3:>3n}, f = {4:>10.4f}'
                print msg.format(pid, self.__gpc.name, i + 1, iterations,
                                 self.__gpc.log_likelihood())

        return self

    def log_likelihood(self):
        """Return the log-likelihood of the model."""

        return self.__gpc.log_likelihood()

    def predict(self, xs, verbose=True):
        """Perform inference in Gaussian process."""

        # Perform inference in the latent function space and then squash
        # the output through the likelihood function.
        f, v = self.__gpc._raw_predict(xs)
        p = self.__gpc.likelihood.predictive_values(f, v)[0]

        return p.flatten(), f.flatten(), v.flatten()


# --------------------------------------------------------------------------- #
#                          All-vs-one classification
# --------------------------------------------------------------------------- #

class AllVsOne(object):
    """Implementation of All-vs-One classification.

    Args:
      model (class):
      X (np.array):
      Y (np.array):
      args (list, optional):
      kwargs (dict, optional):

    Attributes:
      X (np.array):
      Y (np.array):

    """

    def __init__(self, model, X, Y, args=(), kwargs={}, threads=None,
                 verbose=True):

        # Ensure training data are of equal length.
        if X.shape[0] != Y.shape[0]:
            msg = 'The target inputs and outputs must be the same length.'
            raise Exception(msg)

        self.__threads = threads
        self.__verbose = verbose

        # Determine number of input classes.
        self.__labels = np.unique(Y)
        self.__K = len(self.__labels)
        if self.__K <= 2:
            msg = 'Only two classes detected. Use binary classifier.'
            raise Exception(msg)

        # Initialise models.
        self.__models = list()
        for k in self.__labels:
            print 'Creating model for class {0}'.format(k)
            C = (Y == k).astype(float)
            gpc = model(X, C, *args, **kwargs)
            gpc.name = 'Class {0}'.format(k)
            self.__models.append(gpc)

    def optimise(self, *args, **kwargs):
        """Optimise hyper-parameters of Gaussian process."""

        t0 = time.time()

        # Perform optimisation on a SINGLE process.
        if not self.__threads:
            for k, model in enumerate(self.__models):
                if self.__verbose:
                    print 'Training model {0}'.format(self.__labels[k])
                self.__models[k] = model.optimise(*args, **kwargs)

        # Perform optimisation on MULTIPLE processes.
        else:
            # Create list of jobs to execute in parallel.
            jobs = list()
            for k, model in enumerate(self.__models):
                jobs.append({'target': model.optimise,
                             'args': args,
                             'kwargs': kwargs})

            # Send jobs to queue.
            output = parallelisation.pool(jobs, num_processes=self.__threads)

            # Collect and store output.
            for k, model in enumerate(output):
                self.__models[k] = model

        if self.__verbose:
            print 'Elapsed time {0:.3f}s'.format(time.time() - t0)

    def log_likelihood(self):
        """Return log-likelihood of the models."""

        return np.array([l.log_likelihood() for l in self.__models])

    def __predict(self, xs, blocks=1, verbose=True):
        """Perform inference in Gaussian process."""

        # Pre-allocate memory.
        p = np.zeros((xs.shape[0], self.__K))
        f = np.zeros((xs.shape[0], self.__K))
        v = np.zeros((xs.shape[0], self.__K))

        # Iterate through blocks of data.
        for i, idx in block_index(len(xs), blocks):
            if verbose and blocks > 1:
                msg = 'Performing inference on block {0} of {1}'
                print msg.format(i + 1, blocks)

            # Iterate through models.
            for k, model in enumerate(self.__models):
                if verbose and blocks == 1:
                    msg = 'Performing inference in model {0}'
                    print msg.format(self.__labels[k])

                p[idx, k], f[idx, k], v[idx, k] = model.predict(xs[idx, :])

        # Replace rows of zeros (unlikely) with uncertainty.
        p[np.all(p == 0, axis=1), :] = 1.0 / float(self.__K)

        # Normalise one-vs-all probabilities.
        p = (p.T / np.sum(p, axis=1)).T

        return p, f, v

    def predict(self, xs, blocks=1):

        t0 = time.time()

        # Perform optimisation on a SINGLE process.
        if not self.__threads:
            p, f, v = self.__predict(xs, blocks=blocks, verbose=self.__verbose)

        # Perform optimisation on MULTIPLE processes.
        else:

            # Create list of jobs to execute in parallel.
            jobs = list()
            for i, idx in block_index(len(xs), blocks):
                jobs.append({'target': self.__predict,
                             'args': [xs[idx, :],]})
                                      # 1, False]})

            # Send jobs to queue.
            output = parallelisation.pool(jobs, num_processes=self.__threads)

            # Reconstruct output.
            p = np.zeros((xs.shape[0], self.__K))
            f = np.zeros((xs.shape[0], self.__K))
            v = np.zeros((xs.shape[0], self.__K))
            for i, idx in block_index(len(xs), blocks):
                p[idx, :], f[idx, :], v[idx, :] = output[i]

        if self.__verbose:
            print 'Elapsed time {0:.3f}s'.format(time.time() - t0)

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













# --------------------------------------------------------------------------- #
#                                   EP-GPC
# --------------------------------------------------------------------------- #

class EPGPC(object):
    """Create an all-vs-one Gaussian process classifier."""

    def __init__(self, X, Y, kernel=None, mean=None):
        """Initialise the classification model."""

        # Set training data. Note that the targets are re-scaled from {0., 1.}
        # to {-1., 1.} (to match pyGPs, internal representation).
        self.X = X
        self.Y = 2 * (Y - 0.5)

        # Create default mean function.
        if kernel is None:
            mean = pyGPs.mean.Zero()

        # Create default kernel.
        if kernel is None:
            kernel = pyGPs.cov.RBF(log_ell=0., log_sigma=1.)

        # Create model.
        self.__gpc = pyGPs.GPC()
        self.__gpc.setPrior(mean=mean, kernel=kernel)
        self.__gpc.setData(self.X, self.Y)
        self.__gpc.setOptimizer('BFGS')

    @property
    def X(self):
        """Training inputs."""
        return self.__X

    @X.setter
    def X(self, value):
        if not isinstance(value, (np.ndarray)) or value.ndim != 2:
            raise Exception('The target inputs must be a 2D numpy array.')
        else:
            self.__X = value

    @property
    def Y(self):
        """Training outputs."""
        return self.__Y

    @Y.setter
    def Y(self, value):
        if not isinstance(value, (np.ndarray)) or value.ndim != 2:
            raise Exception('The target outputs must be a numpy array.')
        else:
            self.__Y = value

    def optimise(self):
        """Optimise hyper-parameters of Gaussian process."""

        self.__gpc.optimize()
        return self

    def log_likelihood(self):
        """Return the log-likelihood of the model."""

        # Calculate the posterior. Note that it may be available from
        # self.__gpc.nlZ
        nlZ, post = self.__gpc.getPosterior(self, der=False)
        return nlZ

    def predict(self, xs, verbose=True):
        """Perform inference in Gaussian process."""

        N = xs.shape[0]
        ymu, ys2, fmu, fs2, lp = self.__gpc.predict(xs, ys=np.ones((N, 1)))
        return np.exp(lp).flatten(), fmu.flatten(), fs2.flatten()


class EPGPC_FITC(EPGPC):

    def __init__(self, X, Y, Z=None, kernel=None, mean=None):
        """Initialise the classification model."""

        # Set training data. Note that the targets are re-scaled from {0., 1.}
        # to {-1., 1.} (to match pyGPs, internal representation).
        self.X = X
        self.Y = 2 * (Y - 0.5)

        # Create default mean function.
        if kernel is None:
            mean = pyGPs.mean.Zero()

        # Create default kernel.
        if kernel is None:
            kernel = pyGPs.cov.RBF(log_ell=0., log_sigma=1.)

        # Create default inducing points.
        if Z is None:
            num_inducing = 10 * X.shape[1]
            idx = np.linspace(0, len(X)-1, num_inducing).astype(int)
            Z = X[np.unique(idx), :]

        # Create model.
        self.__gpc = pyGPs.GPC_FITC()
        self.__gpc.setPrior(mean=mean, kernel=kernel, inducing_points=Z)
        self.__gpc.setData(self.X, self.Y)
        self.__gpc.setOptimizer('BFGS')


    def optimise(self):
        """Optimise hyper-parameters of Gaussian process."""

        self.__gpc.optimize()
        return self

    def log_likelihood(self):
        """Return the log-likelihood of the model."""

        # Calculate the posterior. Note that it may be available from
        # self.__gpc.nlZ
        nlZ, post = self.__gpc.getPosterior(self, der=False)
        return nlZ

    def predict(self, xs, verbose=True):
        """Perform inference in Gaussian process."""

        N = xs.shape[0]
        ymu, ys2, fmu, fs2, lp = self.__gpc.predict(xs, ys=np.ones((N, 1)))
        return np.exp(lp).flatten(), fmu.flatten(), fs2.flatten()
