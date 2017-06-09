"""Script to test the performance differences in different feature sets.

Uses the autoencoder model to compare the time series data,
aggregate data, and single vector data.

Author:         Zander Blasingame
Insitution:     Clarkson University
Lab:            CAMEL"""

import numpy as np
import os
import csv
import json
import logging

from models.ae_classifier import Classifier as AE
from models.sda_classifier import Classifier as SDA

# Start logger
logging.basicConfig(filename='status.log', filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')


# Find number of subsets
num_tests = 0
for path, _, files in os.walk('data/features/time_series'):
    if 'subset' in path:
        num_tests += 1

num_tests *= 2

for path, _, files in os.walk('data/features'):
    if 'subset' in path:
        num_tests += 1

test_num = 0


# Test model
def test_data_set(loc, data, model, name):
    global test_num
    for path, _, files in os.walk(loc):
        if 'subset' in path:
            train = '{}/train_set.csv'.format(path)
            test = '{}/test_set.csv'.format(path)

            dirs = path.split('/')
            entry = dict(
                classifier=name,
                feature_set=dirs[2],
                exploit=dirs[3].split('_')[-1],
                subset=dirs[4].split('_')[-1]
            )

            logging.info('-'*40)
            logging.info('Training on {}'.format(train))

            model.train(train, reset_weights=True)

            entry.update(model.test(test))
            data.append(entry)
            test_num += 1

            # log data
            logging.info('Progress: {:03.2f}%'.format(
                100*(test_num/num_tests)
            ))
            logging.info(json.dumps(entry, indent=2))
            logging.info('-'*40)


data = []

# Create model
ae = AE(36, std_param=8, batch_size=100,
        num_epochs=30, normalize=True)
ae_2 = AE(12, std_param=8, batch_size=100,
          num_epochs=30, normalize=True)
sda = SDA(num_input=12, num_units=4, num_steps=3, batch_size=100,
          normalize=True, std_param=18, num_epochs=10, debug=False)

test_data_set('data/features/aggregate', data, ae_2, 'autoencoder')
test_data_set('data/features/single_vector', data, ae_2, 'autoencoder')
test_data_set('data/features/time_series', data, ae, 'autoencoder')
test_data_set('data/features/time_series', data, sda, 'sda')

# Write CSV
header = [
    'classifier',
    'feature_set',
    'exploit',
    'subset',
    'accuracy',
    'tp_num',
    'fp_num',
    'tn_num',
    'fn_num'
]

with open('data/results.csv', 'w') as f:
    writer = csv.DictWriter(f, header)
    writer.writeheader()
    writer.writerows(data)
