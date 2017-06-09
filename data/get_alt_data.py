"""Parses time-series data into alternative representations.

Author:         Zander Blasingame
Insitution:     Clarkson University
Lab:            CAMEL
"""

import argparse
import os
import csv
import numpy as np


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(
    'dir',
    type=str,
    help=(
        'Location of the exploit root directory, formatted as:\n'
        'root_dir/\n'
        '    |---subset_n/\n'
        '        |---train_set.csv\n'
        '        |---test_set.csv\n'
        'for all n in {0,...,4}\n'
        'IMPORTANT: All .csv files must be formated as x0,x1,x2\n'
    )
)

parser.add_argument(
    '--out_dir',
    type=str,
    default='.',
    help='Location of the output directory, (default = .)'
)

parser.add_argument(
    '--time_steps',
    type=int,
    default=3,
    help='The number of time steps, (default = 3)'
)

args = parser.parse_args()

files = []

for root, _, _files in os.walk(args.dir):
    for f in _files:
        if 'train' in f or 'test' in f:
            files.append(os.path.join(root, f))

for name in files:
    with open(name, 'r') as f:
        print(name)
        raw_data    = [row for row in csv.reader(f)]
        header      = raw_data[0]
        data        = np.array(raw_data[1:]).astype(np.float32)

    X = data[:, 1:]
    Y = data[:, 0]

    num_samples     = X.shape[0]
    num_features    = int(X.shape[1]/args.time_steps)
    single_header   = header[:1+num_features]

    X_mat       = X.reshape((num_samples, args.time_steps, num_features))
    X_sum       = np.sum(X_mat, axis=1)
    X_single    = X_mat[:, 1]
    Y           = Y.reshape((num_samples, 1))

    types = {
        'full': X,
        'aggregate': X_sum,
        'single_vector': X_single
    }

    for t in types:
        path = '/'.join(name.split('/')[:-1])
        _dir = '/'.join([args.out_dir, t, path])

        if not os.path.exists(_dir):
            os.makedirs(_dir)

        with open('/'.join([args.out_dir, t, name]), 'w') as f:
            writer = csv.writer(f)

            if t == 'full':
                writer.writerows(raw_data)
            else:
                writer.writerow(single_header)
                writer.writerows(np.concatenate((Y, types[t]), axis=1))
