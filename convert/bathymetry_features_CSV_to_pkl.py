#!/usr/bin/python
"""Convert bathymetry feature CSV files to python pickle files.

.. codeauthor:: Asher Bender <a.bender@acfr.usyd.edu.au>

"""
import os
import time
import pickle
from bz2 import BZ2File
from numpy import genfromtxt

if __name__ == '__main__':

    # Location of input CSVs.
    INPUT_PREFIX = '/media/Data/Code/survey_planning/data/bathymetry_'

    # Location of output pickles.
    OUTPUT_PREFIX = '/media/Data/Code/survey_planning/data/bathymetry/'

    # Scales to convert.
    SCALES = [2, 8, 16]

    # Create bathymetry information.
    t0 = time.time()
    print 'Creating bathymetry:'
    for info in ['index', 'depth', 'resolution', 'x_bins', 'y_bins']:

        info_path = INPUT_PREFIX + '{0}.csv'.format(info)
        data = genfromtxt(info_path, delimiter=',')

        if info == 'index':
            data = data.astype(int)

        elif info == 'x_bins':
            data = data.T

        # Write file to output path.
        output_file = OUTPUT_PREFIX + '{0}.pkl.bz2'.format(info)
        print '    {0}'.format(output_file)
        with BZ2File(output_file, 'w') as f:
            pickle.dump(data, f, protocol=2)

    print '    Elapsed time: {0:1.3f}s\n'.format(time.time() - t0)

    # Iterate through scales to convert.
    for scale in SCALES:
        print 'Creating scale {0:0>3}:'.format(scale)

        # Iterate through features and store in dictionary.
        t0 = time.time()
        for field in ['aspect', 'rugosity', 'slope', 'processed']:

            # Create path to feature.
            feature_path = INPUT_PREFIX
            feature_path += '_{0}_{1:0>3}'.format(field, scale)
            feature_path += '.csv'

            # Load data into dictionary.
            data = genfromtxt(feature_path, delimiter=',')

            #  Create path to output.
            output_path = OUTPUT_PREFIX + '{0:0>3}/'.format(scale)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Write file to output path.
            output_file = output_path + '{0}.pkl.bz2'.format(field)
            print '    {0}'.format(output_file)
            with BZ2File(output_file, 'w') as f:
                pickle.dump(data, f, protocol=2)

        print '    Elapsed time: {0:1.3f}s\n'.format(time.time() - t0)

    print 'Done.'
