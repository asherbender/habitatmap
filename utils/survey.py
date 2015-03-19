import pickle
import numpy as np
import bathymetry
import matplotlib.cm as cm


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


# --------------------------------------------------------------------------- #
#                             Plotting Functions
# --------------------------------------------------------------------------- #


def plot_AUV_classes(raster, limits,
                     easting, northing, classes, K,
                     subsample=1, cmap=cm.hsv,
                     **kwargs):
    """Plot AUV classes on top of bathymetry."""

    # Allow data to be sub-sampled.
    easting = easting[::subsample]
    northing = northing[::subsample]
    classes = classes[::subsample]

    # Plot bathymetry raster (in grey-scale).
    ax = bathymetry.plot_raster(raster, extent=limits,
                                clabel='depth (m)',
                                cmap=cm.bone)

    # Plot classified Easting and Northings. Each class is plotted in a new
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
