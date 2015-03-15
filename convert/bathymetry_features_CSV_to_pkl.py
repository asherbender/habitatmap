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
    INPUT_PREFIX = '/media/Data/Code/survey_planning/data/bathymetry/scott_reef'

    # Location of output pickles.
    OUTPUT_PREFIX = '/media/Data/Code/survey_planning/data/bathymetry/scott_reef/'

    # Scales to convert.
    SCALES = [1, 2, 4, 8, 16]

    # Create bathymetry information.
    t0 = time.time()
    print 'Creating bathymetry:'
    for info in ['index', 'depth', 'resolution', 'x_bins', 'y_bins']:

        info_path = INPUT_PREFIX + '_{0}.csv'.format(info)
        if os.path.exists(info_path):
            data = genfromtxt(info_path, delimiter=',')
        else:
            msg = "The file '{0}' does not exist"
            raise Exception(msg.format(info_path))

        if info == 'index':
            data = data.astype(int)

        # Write file to output path.
        output_file = OUTPUT_PREFIX + '{0}.pkl.bz2'.format(info)
        print '    {0}'.format(output_file)
        with BZ2File(output_file, 'w') as f:
            pickle.dump(data.flatten(), f, protocol=2)

    for info in ['zone', 'lon_bins', 'lat_bins']:

        info_path = INPUT_PREFIX + '_{0}.csv'.format(info)
        if os.path.exists(info_path):

            # Write file to output path.
            if info == 'zone':
                with open(info_path, 'r') as f:
                    data = f.read()
                output_file = OUTPUT_PREFIX + '{0}.txt'.format(info)
                print '    {0}'.format(output_file)
                with open(output_file, 'w') as f:
                    f.write(data.replace(',', ''))

            # Load CSV data as numpy arrays.
            else:
                data = genfromtxt(info_path, delimiter=',')
                output_file = OUTPUT_PREFIX + '{0}.pkl.bz2'.format(info)
                print '    {0}'.format(output_file)
                with BZ2File(output_file, 'w') as f:
                    pickle.dump(data.flatten(), f, protocol=2)

    print '    Elapsed time: {0:1.3f}s\n'.format(time.time() - t0)

    # Iterate through scales to convert.
    for scale in SCALES:
        print 'Creating scale {0:0>3}:'.format(scale)

        # Iterate through features and store in dictionary.
        t0 = time.time()
        for field in ['index', 'aspect', 'rugosity', 'slope']:

            # Create path to feature.
            feature_path = INPUT_PREFIX
            feature_path += '_{0}_{1:0>3}'.format(field, scale)
            feature_path += '.csv'

            # Load data into dictionary.
            if os.path.exists(feature_path):
                data = genfromtxt(feature_path, delimiter=',')
                if field == 'index':
                    data = data.astype(int)
            else:
                msg = "The file '{0}' does not exist"
                raise Exception(msg.format(feature_path))

            #  Create path to output.
            output_path = OUTPUT_PREFIX + '{0:0>3}/'.format(scale)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Write file to output path.
            output_file = output_path + '{0}.pkl.bz2'.format(field)
            print '    {0}'.format(output_file)
            with BZ2File(output_file, 'w') as f:
                pickle.dump(data.flatten(), f, protocol=2)

        print '    Elapsed time: {0:1.3f}s\n'.format(time.time() - t0)

    print 'Done.'
