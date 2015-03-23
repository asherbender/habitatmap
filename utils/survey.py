import os
import utm
import pickle
import collections
import numpy as np
import bathymetry
import matplotlib.cm as cm


class stereo_pose_dct(collections.OrderedDict):

    def __init__(self):

        super(stereo_pose_dct, self).__init__()

        self['version'] = None
        self['augmented poses'] = None
        self['state vector size'] = None
        self['closure hypotheses'] = None
        self['loop closures'] = None
        self['origin latitude'] = None
        self['origin longitude'] = None
        self['origin easting'] = None
        self['origin northing'] = None
        self['utm zone'] = None
        self['pose id'] = list()
        self['timestamp'] = list()
        self['latitude'] = list()
        self['longitude'] = list()
        self['X'] = list()
        self['Y'] = list()
        self['Z'] = list()
        self['roll'] = list()
        self['pitch'] = list()
        self['yaw'] = list()
        self['left image'] = list()
        self['right image'] = list()
        self['altitude'] = list()
        self['image radius'] = list()
        self['trajectory cross-over'] = list()
        self['length'] = None
        self['image directory'] = None
        self['campaign name'] = None
        self['dive name'] = None
        self['renav name'] = None

    def __summarise(self):

        msg  = 'version:               {0:n}\n'.format(self['version'])
        msg += 'augmented poses:       {0:n}\n'.format(self['augmented poses'])
        msg += 'state vector size:     {0:n}\n'.format(self['state vector size'])
        msg += 'closure hypotheses:    {0:n}\n'.format(self['closure hypotheses'])
        msg += 'loop closures:         {0:n}\n'.format(self['loop closures'])
        msg += 'origin latitude:       {0:n}\n'.format(self['origin latitude'])
        msg += 'origin longitude:      {0:n}\n'.format(self['origin longitude'])
        msg += 'origin easting:        {0:n}\n'.format(self['origin easting'])
        msg += 'origin northing:       {0:n}\n'.format(self['origin northing'])
        msg += 'utm zone:              {0}\n'.format(self['utm zone'])
        msg += 'pose id:               ({0:n}, )\n'.format(len(self['pose id']))
        msg += 'timestamp:             ({0:n}, )\n'.format(len(self['timestamp']))
        msg += 'latitude:              ({0:n}, )\n'.format(len(self['latitude']))
        msg += 'longitude:             ({0:n}, )\n'.format(len(self['longitude']))
        msg += 'X:                     ({0:n}, )\n'.format(len(self['X']))
        msg += 'Y:                     ({0:n}, )\n'.format(len(self['Y']))
        msg += 'Z:                     ({0:n}, )\n'.format(len(self['Z']))
        msg += 'roll:                  ({0:n}, )\n'.format(len(self['roll']))
        msg += 'pitch:                 ({0:n}, )\n'.format(len(self['pitch']))
        msg += 'yaw:                   ({0:n}, )\n'.format(len(self['yaw']))
        msg += 'left image:            ({0:n}, )\n'.format(len(self['left image']))
        msg += 'right image:           ({0:n}, )\n'.format(len(self['right image']))
        msg += 'altitude:              ({0:n}, )\n'.format(len(self['altitude']))
        msg += 'image radius:          ({0:n}, )\n'.format(len(self['image radius']))
        msg += 'trajectory cross-over: ({0:n}, )\n'.format(len(self['trajectory cross-over']))
        msg += 'length:                {0:n}\n'.format(self['length'])
        msg += 'image directory:       {0}\n'.format(self['image directory'])
        msg += 'campaign name:         {0}\n'.format(self['campaign name'])
        msg += 'dive name:             {0}\n'.format(self['dive name'])
        msg += 'renav name:            {0}\n'.format(self['renav name'])

        return msg

    def __repr__(self):
        return self.__summarise()

    def __str__(self):
        return self.__summarise()


def _parse_stereo_pose_est(fpath, header_spec, origin_spec, column_spec):

    # If a directory was specified, look for a navigation file.
    if os.path.isdir(fpath):
        fpath = os.path.join(fpath, 'stereo_pose_est.data')

    # Ensure file exists.
    if not os.path.isfile(fpath):
        msg = "The file '{0}' does not exist.".format(fpath)
        raise IOError(msg)

    # Allocate memory for data.
    data = stereo_pose_dct()

    # Read contents of file.
    with open(fpath, 'r') as f:
        contents = f.read().split('\n')

    # -------------------------------------------------------------------------
    # Read header
    # -------------------------------------------------------------------------
    header = list()
    for i, line in enumerate(contents):

        # Process header lines.
        if line.startswith('% '):
            header.append(line)

        # Skip blank lines.
        elif line == '':
            continue

        # Stop processing header
        else:
            break

    # Return header as list and joined text.
    data['header'] = '\n'.join(header)
    header_list = header
    pointer = i

    # Get meta data from header.
    #     <text delimiter>, <type>, <dictionary key>
    for line in header_list:
        for delimiter, meta_type, key in header_spec:
            if delimiter in line:
                data[key] = meta_type(line.replace(delimiter, '').strip())

    # -------------------------------------------------------------------------
    #  Read origin
    # -------------------------------------------------------------------------
    if origin_spec:

        # Get meta data from header.
        #     <text delimiter>, <type>, <dictionary key>
        for delimiter, meta_type, key in origin_spec:

            # Get latitude/longitude.
            line = contents[pointer]
            if delimiter in line:
                data[key] = meta_type(line.replace(delimiter, '').strip())

            pointer += 1

        # Assume name of keys (a bit sloppy). If they exist, get local
        # northing and easting for origin.
        if (('origin latitude' in data.keys()) and
            ('origin longitude' in data.keys())):
            local = utm.from_latlon(data['origin latitude'],
                                    data['origin longitude'])

            data['origin easting'] = local[0]
            data['origin northing'] = local[1]
            data['utm zone'] = str(local[2]) + str(local[3])

    # -------------------------------------------------------------------------
    #  Read data
    # -------------------------------------------------------------------------
    while True:

        # Read row of data and iterate through elements (columns) of data.
        line = contents[pointer].split()
        if line:
            for i, (column_type, key) in enumerate(column_spec):
                data[key].append(column_type(line[i]))

        pointer += 1
        if pointer >= len(contents):
            break

    # Store length.
    data['length'] = len(data['pose id'])

    # Convert columns to numpy arrays.
    for i, (column_type, key) in enumerate(column_spec):
        if len(data[key]) == data['length']:
            if column_type in [int, float, str, bool]:
                dt = np.dtype(column_type)
                data[key] = np.array(data[key], dtype=dt)
            else:
                data[key] = np.array(data[key])

    # -------------------------------------------------------------------------
    #  Store campaign, dive and renav names
    # -------------------------------------------------------------------------
    path_tokens = fpath.split(os.path.sep)
    if len(path_tokens) > 3:
        data['campaign name'] = path_tokens[-4]
        data['dive name'] = path_tokens[-3]
        data['renav name'] = path_tokens[-2]

        # Get time stamp from beginning of dive name.
        timestamp = '_'.join(data['dive name'].split('_')[:2])[1:]
        data['image directory'] = 'i' + timestamp + '_cv'

    return data


def read_stereo_pose_est(fname):
    """Convert seabed_slam data to a dictionary object.

    :py:class:read_stereo_pose_est converts stereo_pose_est.data files
    produced by seabed_slam into a python dictionary.

     A dictionary is returned which contains the following keys:

        data = {'header':                header block of original data file
                'version':               version of stereo_pose_est.data file
                'augmented poses':       number of augmented poses
                'state vector size':     state vector size
                'closure hypotheses':    number of loop closure hypotheses
                'loop closures':         number of loop closures

                'origin latitude':       latitude origin of dive
                'origin longitude':      longitude origin of dive
                'origin easting':        local easting origin of dive
                'origin northing':       local northing origin of dive
                'utm zone':              UTM zone of origin

                'pose id':               integer uniquely identifying pose
                'timestamp':             time in seconds
                'latitude':              in degrees
                'longitude':             in degrees
                'X':                     in meters, local navigation frame
                'Y':                     in meters, local navigation frame
                'Z':                     in meters, local navigation frame
                'roll':                  in rads, (X-axis Euler angle) local navigation frame
                'pitch':                 in rads, (Y-axis Euler angle) local navigation frame
                'yaw':                   in rads, (Z-axis Euler angle) local navigation frame
                'left image':            name of left (color) image
                'right image':           name of right (mono-chrome) image
                'altitude':              in meters
                'image radius':          approximate bounding image radius in meters
                'trajectory cross-over': likely trajectory cross-over point 1 for true, 0 for false

                'length':                number of elements in the navigation data
                'campaign name':         name of campaign
                'dive name':             name of dive
                'renav name':            name of renav (seabed slam run)
               }

    Args:
        fname (str): path to stereo_pose_est.data file.

    Returns:
        dict: containing navigation information.

    """

    # Define header meta-data.
    #     <text delimiter>, <type>, <dictionary key>
    meta = [('% STEREO_POSE_FILE VERSION ', int, 'version'),
            ('%    Number of augmented poses: ', int, 'augmented poses'),
            ('%    State vector size        : ', int, 'state vector size'),
            ('%    Number of hypotheses   : ', int, 'closure hypotheses'),
            ('%    Number of loop closures: ', int, 'loop closures')]

    # Define origin data:
    #     <text delimiter>, <type>, <dictionary key>
    origin = [('ORIGIN_LATITUDE', float, 'origin latitude'),
              ('ORIGIN_LONGITUDE', float, 'origin longitude')]

    # Define column names (dictionary keys) and data types.
    #     <type>, <dictionary key>
    columns = [(int, 'pose id'),
               (str, 'timestamp'),
               (float, 'latitude'),
               (float, 'longitude'),
               (float, 'X'),
               (float, 'Y'),
               (float, 'Z'),
               (float, 'roll'),
               (float, 'pitch'),
               (float, 'yaw'),
               (str, 'left image'),
               (str, 'right image'),
               (float, 'altitude'),
               (float, 'image radius'),
               (bool, 'trajectory cross-over')]

    # Read file.
    try:
        data = _parse_stereo_pose_est(fname, meta, origin, columns)
    except:
        raise

    return data


def index_stereo_pose_est(nav_data, index):
    """Index elements in a stereo_pose_est.data dictionary

   :py:func:index_stereo_pose_est returns a dictionary containing indexed
    elements of a stereo_pose_est dictionary.

    Args:
        nav_data (dict): Dictionary containing navigation data.
        index (np.array): is a vector used to index navigation data in the
            dictionary.

    Returns:
        dict: The output structure is an identical copy of the input structure
            but the fields containing navigation data will only contain data at
            the locations specified by INDEX.

    """

    # Copy data.
    indexed = stereo_pose_dct()

    # Index vector elements.
    for key in nav_data.keys():
        try:
            if len(nav_data[key]) == nav_data['length']:
                indexed[key] = nav_data[key][index]
                length_key = key
            else:
                indexed[key] = nav_data[key]
        except:
            indexed[key] = nav_data[key]

    # Store new length.
    indexed['length'] = len(indexed[length_key])
    return indexed


def load_stereo_pose_est_cluster(nav_data, image_label_file):

    # Pre-allocate memory for cluster data.
    nav_data['cluster'] = np.zeros(nav_data['length'], dtype=np.int)
    processed = np.zeros(nav_data['length'], dtype=np.bool)

    # Read contents of file.
    with open(image_label_file, 'r') as f:
        contents = f.read().split('\n')

    # Iterate through image label file matching AUV pose IDs to the
    # stereo_pose_est data.
    idx = 0
    for row in contents:

        # Skip header data.
        if row.startswith('%'):
            continue
        else:
            row = row.split()

        # Search for matching pose IDs (assuming ordered).
        while True:
            if row and (idx < nav_data['length']):

                # Line match found, store data.
                if nav_data['pose id'][idx] == int(row[0]):
                    nav_data['cluster'][idx] = int(row[-1])
                    processed[idx] = True
                    break

                # No line match found. Search next line.
                else:
                    idx += 1
            else:
                break

    # Issue warning if no data could be matched.
    if not np.any(processed):
        raise Exception('Could not assign clusters to navigation data.')

    return nav_data, processed


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
