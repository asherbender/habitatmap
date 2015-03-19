import time
import pyGPs
import bathymetry
import parallelisation

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def block_index(length, number_chunks):
    """Yield index to array in block.

    Args:
        length (int): length of array.
        number_chunks (int): number of blocks to break array into.

    Yields:

        tuple: The first element is an integer representing the current block
            number. The second element is a list of indices marking the
            location of the 'block' in the original array.

    """

    j = -1
    n = int(np.ceil(float(length) / number_chunks))
    for i in xrange(0, length, n):
        j += 1
        if (i + n) <= length:
            idx = range(i, i + n)
        else:
            idx = range(i, length)
        yield j, idx


def block_inference(function, xs, args=list(), kwargs=dict(), blocks=10,
                    parallel=False, verbose=True):

    # Ensure blocks is a positive number.
    if blocks <= 0:
        raise Exception('blocks must be a positive integer')
    else:
        blocks = int(blocks)

    t0 = time.time()

    # Perform inference on a SINGLE process.
    if parallel is False:
        job_output = list()
        for i, idx in block_index(len(xs), blocks):
            if verbose:
                msg = 'Performing inference on block {0:>3} of {1:>3}.'
                print msg.format(i + 1, blocks)
            job_output.append(function(xs[idx, :], *args, **kwargs))

    # Perform inference on MULTIPLE processes.
    else:

        # Create list of jobs to execute in parallel.
        jobs = list()
        for i, idx in block_index(len(xs), blocks):
            jobs.append({'target': function,
                         'args': [xs[idx, :], ] + args,
                         'kwargs': kwargs})

        # Send jobs to queue.
        job_output = parallelisation.pool(jobs, num_processes=parallel)

    # Copy jobs into single output.
    output = list(job_output[0])
    for i in range(1, blocks):
        for j in range(len(output)):
            output[j] = np.concatenate((output[j], job_output[i][j]))

    if verbose:
        print 'Elapsed time {0:.3f}s'.format(time.time() - t0)

    return output

# --------------------------------------------------------------------------- #
#                          Gaussian Process Regression
# --------------------------------------------------------------------------- #


class GPR(object):
    """Gaussian process regression.

    Args:
        X (np.array): (NxD) training inputs.
        Y (np.array): (Nx1) training outputs.
        mean (pyGPs.mean, optional): mean function to use during modelling. If
            `None` is specified, a zero-mean function will be used.
        kernel (pyGPs.cov, optional): covariance function to use during
            modelling. If `None` is specified, a squared-exponential covariance
            function will be used.

    Attributes:
        X (np.array): (NxD) training inputs.
        Y (np.array): (Nx1) training output.

    """

    def __init__(self, X, Y, mean=None, kernel=None):

        # Set training data.
        self.X = X
        self.Y = Y

        # Create default mean function.
        if kernel is None:
            mean = pyGPs.mean.Zero()

        # Create default kernel.
        if kernel is None:
            kernel = pyGPs.cov.RBF(log_ell=0., log_sigma=1.)

        # Create model.
        self._gpr = pyGPs.GPR()
        self._gpr.setPrior(mean=mean, kernel=kernel)
        self._gpr.setData(self.X, self.Y)

    @property
    def X(self):
        """Training inputs."""
        return self._X

    @X.setter
    def X(self, value):
        if not isinstance(value, (np.ndarray)) or value.ndim != 2:
            raise Exception('The target inputs must be a 2D numpy array.')
        else:
            self._X = value

    @property
    def Y(self):
        """Training outputs."""
        return self._Y

    @Y.setter
    def Y(self, value):
        if not isinstance(value, (np.ndarray)) or value.ndim != 2:
            raise Exception('The target outputs must be a numpy array.')
        else:
            self._Y = value

    def optimise(self):
        """Optimise hyper-parameters of Gaussian process classifier."""

        self._gpr.optimize()
        return self

    def log_likelihood(self):
        """Return the log-likelihood of the model.

        Returns:
            float: negative-log likelihood of the model.

        """

        # Calculate the posterior. Note that it may be available from
        # self.__gpc.nlZ
        nlZ, post = self._gpr.getPosterior(self, der=False)
        return nlZ

    def predict(self, xs, verbose=True):
        """Perform inference in Gaussian process regression model.

        Args:
            xs (np.array): (MxD) input queries.

        Returns:
            tuple: The first element is a numpy array of predictive means. The
                second element is a numpy array of predictive variances in the
                latent function.

        """

        N = xs.shape[0]
        ymu, ys2, fmu, fs2, lp = self._gpr.predict(xs, ys=np.ones((N, 1)))
        return fmu.flatten(), fs2.flatten()


class GPR_FITC(GPR):
    """Sparse Gaussian process regression.

    Args:
        X (np.array): (NxD) training inputs.
        Y (np.array): (Nx1) training outputs.
        Z (np.array or int, optional): If set to `None` 10*D will be evenly
           sampled from the rows of the training inputs `X` to use as inducing
           inputs. If specified as an integer, a number of samples will be
           evenly sampled from the rows of the training inputs `X` to use as
           inducing inputs. If specified as an (QxD) numpy array, the locations
           specified by the input will be used as inducing point.
        mean (pyGPs.mean, optional): mean function to use during modelling. If
            `None` is specified, a zero-mean function will be used.
        kernel (pyGPs.cov, optional): covariance function to use during
            modelling. If `None` is specified, a squared-exponential covariance
            function will be used.

    Attributes:
        X (np.array): (NxD) training inputs.
        Y (np.array): (Nx1) training output.

    """

    def __init__(self, X, Y, Z=None, kernel=None, mean=None):
        """Initialise the classification model."""

        # Set training data.
        self.X = X
        self.Y = Y

        # If no inducing points are specified, sample ten points per dimensions
        # for inducing points.
        if Z is None:
            Z = 10 * X.shape[1]

        # Create set of inducing points by linearly interpolating the dataset.
        if isinstance(Z, (int, long)):
            num_inducing = Z

            if num_inducing < len(X):
                idx = np.linspace(0, len(X)-1, num_inducing).astype(int)
                Z = X[np.unique(idx), :]

        # Use input array of inducing points.
        else:
            if Z.shape[1] != X.shape[1]:
                msg = 'The inducing points Z must have the same '
                msg += 'number of columns as the inputs X.'
                raise Exception(msg)

            elif Z.shape[0] >= X.shape[0]:
                msg = 'There must be less inducing points Z than inputs X.'
                raise Exception(msg)

        # Create default mean function.
        if kernel is None:
            mean = pyGPs.mean.Zero()

        # Create default kernel.
        if kernel is None:
            kernel = pyGPs.cov.RBF(log_ell=0., log_sigma=1.)

        # Create model.
        self._gpr = pyGPs.GPR_FITC()
        self._gpr.setPrior(mean=mean, kernel=kernel, inducing_points=Z)
        self._gpr.setData(self.X, self.Y)


# --------------------------------------------------------------------------- #
#                        Gaussian Process Classification
# --------------------------------------------------------------------------- #

class EPGPC(object):
    """Expectation propagation, Gaussian process (binary) classification.

    Args:
        X (np.array): (NxD) training inputs.
        Y (np.array): (Nx1) training output where the values take on binary
            assignments (zero for class one and one for class two).
        mean (pyGPs.mean, optional): mean function to use during modelling. If
            `None` is specified, a zero-mean function will be used.
        kernel (pyGPs.cov, optional): covariance function to use during
            modelling. If `None` is specified, a squared-exponential covariance
            function will be used.

    Attributes:
        X (np.array): (NxD) training inputs.
        Y (np.array): (Nx1) training output.

    """

    def __init__(self, X, Y, mean=None, kernel=None):
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
        self._gpc = pyGPs.GPC()
        self._gpc.setPrior(mean=mean, kernel=kernel)
        self._gpc.setData(self.X, self.Y)
        self._gpc.setOptimizer('BFGS')

    @property
    def X(self):
        """Training inputs."""
        return self._X

    @X.setter
    def X(self, value):
        if not isinstance(value, (np.ndarray)) or value.ndim != 2:
            raise Exception('The target inputs must be a 2D numpy array.')
        else:
            self._X = value

    @property
    def Y(self):
        """Training outputs."""
        return self._Y

    @Y.setter
    def Y(self, value):
        if not isinstance(value, (np.ndarray)) or value.ndim != 2:
            raise Exception('The target outputs must be a numpy array.')
        else:
            self._Y = value

    def optimise(self):
        """Optimise hyper-parameters of Gaussian process classifier."""

        self._gpc.optimize()
        return self

    def log_likelihood(self):
        """Return the log-likelihood of the model.

        Returns:
            float: negative-log likelihood of the model.

        """

        # Calculate the posterior. Note that it may be available from
        # self.__gpc.nlZ
        nlZ, post = self._gpc.getPosterior(self, der=False)
        return nlZ

    def predict(self, xs, verbose=True):
        """Perform inference in Gaussian process classifier.

        Args:
            xs (np.array): (MxD) input queries.

        Returns: tuple: The first element is a numpy array of predictive
            probabilities. The second element is a numpy array of predictive
            means in the latent function. The final element is a numpy array of
            predictive variances in the latent function.

        """

        N = xs.shape[0]
        ymu, ys2, fmu, fs2, lp = self._gpc.predict(xs, ys=np.ones((N, 1)))
        return np.exp(lp).flatten(), fmu.flatten(), fs2.flatten()


class EPGPC_FITC(EPGPC):
    """Sparse expectation propagation, Gaussian process (binary) classification.

    Args:
        X (np.array): (NxD) training inputs.
        Y (np.array): (Nx1) training output where the values take on binary
            assignments (zero for class one and one for class two).
        Z (np.array or int, optional): If set to `None` 10*D will be evenly
           sampled from the rows of the training inputs `X` to use as inducing
           inputs. If specified as an integer, a number of samples will be
           evenly sampled from the rows of the training inputs `X` to use as
           inducing inputs. If specified as an (QxD) numpy array, the locations
           specified by the input will be used as inducing point.
        mean (pyGPs.mean, optional): mean function to use during modelling. If
            `None` is specified, a zero-mean function will be used.
        kernel (pyGPs.cov, optional): covariance function to use during
            modelling. If `None` is specified, a squared-exponential covariance
            function will be used.

    Attributes:
        X (np.array): (NxD) training inputs.
        Y (np.array): (Nx1) training output.

    """

    def __init__(self, X, Y, Z=None, kernel=None, mean=None):
        """Initialise the classification model."""

        # Set training data. Note that the targets are re-scaled from {0., 1.}
        # to {-1., 1.} (to match pyGPs, internal representation).
        self.X = X
        self.Y = 2 * (Y - 0.5)

        # If no inducing points are specified, sample ten points per dimensions
        # for inducing points.
        if Z is None:
            Z = 10 * X.shape[1]

        # Create set of inducing points by linearly interpolating the dataset.
        if isinstance(Z, (int, long)):
            num_inducing = Z

            if num_inducing < len(X):
                idx = np.linspace(0, len(X)-1, num_inducing).astype(int)
                Z = X[np.unique(idx), :]

        # Use input array of inducing points.
        else:
            if Z.shape[1] != X.shape[1]:
                msg = 'The inducing points Z must have the same '
                msg += 'number of columns as the inputs X.'
                raise Exception(msg)

            elif Z.shape[0] >= X.shape[0]:
                msg = 'There must be less inducing points Z than inputs X.'
                raise Exception(msg)

        # Create default mean function.
        if kernel is None:
            mean = pyGPs.mean.Zero()

        # Create default kernel.
        if kernel is None:
            kernel = pyGPs.cov.RBF(log_ell=0., log_sigma=1.)

        # Create model.
        self._gpc = pyGPs.GPC_FITC()
        self._gpc.setPrior(mean=mean, kernel=kernel, inducing_points=Z)
        self._gpc.setData(self.X, self.Y)
        self._gpc.setOptimizer('BFGS')


# --------------------------------------------------------------------------- #
#                          All-vs-one classification
# --------------------------------------------------------------------------- #

class OneVsAll(object):
    """Implementation of All-vs-One classification.

    In one-vs-all (OVA) classification a binary classifier is trained to
    classify one class against all others. This is done for each
    class. Predictions are made by performing inference in each model and
    combining and normalising the output of each classifier into a single
    multinomial.

    Args:
        model (class): Pointer to binary classifier to be used in multi-class
            classification.
        X (np.array): (NxD) training inputs.
        Y (np.array): (Nx1) training output where the values take on discrete
            1-of-K integer values. For example, in a five-class problem each
            element in Y must take on one of the following values {1,2,3,4,5}.
        args (list, optional): list of arguments used to initialise each binary
            classification model.
        kwargs (dict, optional): dictionary of optional keyword-value pairs
            used to initialise each binary classification model.
        threads (int, optional): number of processes to use during training and
            inference. If set to `None`, all operations will be performed on a
            single process.
        verbose (bool, optional): If set to True the activity will be displayed
            on stdout.

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
            if self.__verbose:
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

        # Perform inference on a SINGLE process.
        if not self.__threads:
            p, f, v = self.__predict(xs, blocks=blocks, verbose=self.__verbose)

        # Perform inference on MULTIPLE processes.
        else:

            # Create list of jobs to execute in parallel.
            jobs = list()
            for i, idx in block_index(len(xs), blocks):
                jobs.append({'target': self.__predict,
                             'args': [xs[idx, :]]})

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
