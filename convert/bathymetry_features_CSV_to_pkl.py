#!/usr/bin/python
"""Convert bathymetry feature CSV files to python pickle files.

.. codeauthor:: Asher Bender <a.bender@acfr.usyd.edu.au>

"""
import pickle
from numpy import genfromtxt


if __name__ == '__main__':

    # Location of input CSVs.
    INPUT_PREFIX = '/media/Data/Code/survey_planning/features/ohara_2008_bathymetry'

    # Location of output pickles.
    OUTPUT_PREFIX = '/media/Data/Code/survey_planning/features/ohara_2008_bathymetry'

    # Scales to convert.
    SCALES = [2, 8, 16]

    # Iterate through scales to convert.
    for scale in SCALES:
        print 'Creating scale {0:0>3}:'.format(scale)

        # Create dictionary for current feature scale.
        features = {'neighbours': scale}

        # Iterate through dimension and store in dictionary.
        for field in ['x_bins', 'y_bins', 'resolution']:

            # Create path to dimension data.
            feature_path = INPUT_PREFIX
            feature_path += '_{0}'.format(field)
            feature_path += '.csv'

            # Load dimension data.
            features[field] = genfromtxt(feature_path, delimiter=',')

        # Iterate through features and store in dictionary.
        for field in ['aspect', 'rugosity', 'slope', 'processed']:

            # Create path to feature.
            feature_path = INPUT_PREFIX
            feature_path += '_{0}_{1:0>3}'.format(field, scale)
            feature_path += '.csv'

            # Load data into dictionary.
            print '    loading {0}'.format(field)
            features[field] = genfromtxt(feature_path, delimiter=',')

        # Convert 'processed' into a boolean array.
        features[field] = features[field].astype(bool)

        # Load raster dictionary from pickle file.
        output_file = OUTPUT_PREFIX + '_{0:0>3}.pkl'.format(scale)
        print '    saving to: {0}\n'.format(output_file)
        with open(output_file, 'wb') as f:
            raster = pickle.dump(features, f)

    print 'Done.'
