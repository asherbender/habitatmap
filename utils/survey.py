import pickle
import numpy as np
import bathymetry
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

# --------------------------------------------------------------------------- #
#                             Plotting Functions
# --------------------------------------------------------------------------- #

def plot_AUV_classes(raster, limits,
                     easting, northing, classes, K,
                     subsample=1, cmap=cm.hsv,
                     **kwargs):
    """Plot AUV classes on top of bathymetry."""

    easting = easting[::subsample]
    northing = northing[::subsample]
    classes = classes[::subsample]

    # Plot bathymetry raster (in grey-scale).
    ax = bathymetry.plot_raster(raster, limits,
                                cmap=cm.bone,
                                clabel='depth (m)')

    # Plot classified easting and northings. Each class is plotted in a new
    # colour.
    scax = list()
    sctitle = list()
    colours = cmap(np.linspace(0, 1, K))
    for i in range(K):
        idx = (classes == (i + 1))
        sctitle.append('Class {0}'.format(i + 1))
        scax.append(ax.scatter(easting[idx], northing[idx], s=15,
                               c=colours[i, :], lw=0))

    # Render legend.
    ax.legend(scax, sctitle, scatterpoints=1, **kwargs)

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


def plot_survey_origins(origins, raster, limits, *args, **kwargs):
    """Plot survey origins."""

    ax = bathymetry.plot_raster(raster, limits, cmap=cm.gray)
    ax.plot(origins[:, 0], origins[:, 1], *args, **kwargs)
    ax.axis(limits)


def plot_survey_utility(origins, utility, raster, limits, *args, **kwargs):
    """Plot survey utility."""

    ax = bathymetry.plot_raster(raster, limits, cmap=cm.gray)
    sc = ax.scatter(origins[:, 0], origins[:, 1], c=utility, *args, **kwargs)
    ax.axis(limits)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label('Utility', rotation=90)
