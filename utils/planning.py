import numpy as np
import bathymetry
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def shift_template(easting, northing, template, raster):
    """Shift indices in a discrete template to a Cartesian location.

    Args:
        easting (float): X co-ordinate to place discrete survey template.
        northing (float): Y co-ordinate to place discrete survey template.
        template (np.array): Array (dtype=np.int) representing a survey
            template.
        raster (dict): Dictionary containing bathymetry size information (see
            :py:func:meta_from_bins)

    Returns:

        np.array: The discrete survey template shifted to the specified
            Cartesian location. Note that easting and northing location is
            specified in Cartesian space. The returned template is a matrix of
            indices referencing the location (rows, cols) of the survey
            template in the raster.

    Raises:
        Exception: If the survey template is not a [Nx2] numpy array of
            integers indicating which locations in a raster that the template
            spans.

    """

    # Ensure the template is specified as integers.
    if not np.issubdtype(template.dtype, np.integer):
        msg = 'The survey template must be an [Nx2] numpy array of integers'
        msg += ' indexing visited locations in the raster.'
        raise Exception(msg)

    # Convert the Cartesian X/Y origin to raster subscripts.
    origin_col = int((easting - raster['x_lim'][0]) / raster['resolution'])
    origin_row = int((northing - raster['y_lim'][0]) / raster['resolution'])

    # Create template origin.
    origin = np.array([[origin_col, origin_row]], dtype=np.integer)

    # Shift template to origin.
    template = origin + template
    template[:, 1] = raster['rows'] - template[:, 1] - 1

    return template


def feasible_region(template, bathy):

    # Get limits of bathymetry and template.
    x_min = template[:, 0].min()
    x_max = bathy['cols'] - template[:, 0].max()
    y_max = bathy['rows'] - template[:, 1].min()
    y_min = template[:, 1].max()

    # Mask off rows and columns where the template would 'fall off' the
    # bathymetry.
    row_mask = np.arange(bathy['rows'])
    col_mask = np.arange(bathy['cols'])
    feasible = np.ones(bathy['shape'], dtype=bool)
    feasible[:, col_mask < x_min] = False
    feasible[:, col_mask > x_max] = False
    feasible[row_mask < y_min, :] = False
    feasible[row_mask > y_max, :] = False

    return feasible


def grid_survey_origins(surveys, feasible, raster):
    """Create a grid of survey origins."""

    # Allocate the same number of surveys to each axis.
    axis_surveys = int(np.sqrt(surveys))

    # Increase number of surveys per axis until the total number of surveys
    # breaches the specified limit.
    while True:
        x_org = np.linspace(0, raster['cols'] - 1, axis_surveys)
        y_org = np.linspace(0, raster['rows'] - 1, axis_surveys)
        x_org, y_org = np.meshgrid(x_org.astype(int), y_org.astype(int))
        x_org = x_org.flatten()
        y_org = y_org.flatten()
        valid = feasible[y_org, x_org]

        if valid.flatten().sum() >= surveys:
            break
        else:
            axis_surveys += 1

    # Return origins as a [Nx2] matrix.
    return np.vstack((raster['x_bins'][x_org[valid]],
                      raster['y_bins'][y_org[valid]])).T

# --------------------------------------------------------------------------- #
#                             Plotting Functions
# --------------------------------------------------------------------------- #


def plot_feasible_region(feasible, raster, extent, **kwargs):
    """Overlay feasible region on bathymetry (unfeasible in red)."""

    # Convert feasible region into an RGB image.
    feasible = feasible[:, :, np.newaxis].repeat(3, axis=2)
    feasible[:, :, 0] = ~feasible[:, :, 0]
    feasible[:, :, 2] = 0

    # Plot bathy in gray-scale and feasible region on top.
    ax = bathymetry.plot_raster(raster, extent=extent, cmap=cm.gray)
    ax.imshow(feasible, extent=extent, interpolation='none', **kwargs)

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


def plot_survey_origins(origins, raster, extent, *args, **kwargs):
    """Plot survey origins."""

    ax = bathymetry.plot_raster(raster, extent=extent, cmap=cm.gray)
    ax.plot(origins[:, 0], origins[:, 1], *args, **kwargs)
    return ax


def plot_survey_utility(origins, utility, raster, extent, *args, **kwargs):
    """Plot survey utility."""

    ax = bathymetry.plot_raster(raster, extent=extent, cmap=cm.gray, no_cbar=True)
    sc = ax.scatter(origins[:, 0], origins[:, 1], c=utility, *args, **kwargs)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label('Utility', rotation=90)
