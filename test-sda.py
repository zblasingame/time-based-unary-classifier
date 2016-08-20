""" Python script to visualize the effects of the unit layer size on the
    Neural Network.
    Author: Zander Blasingame """

import argparse
import subprocess
import pickle
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--online',
                    action='store_true',
                    help='Flag to publish visualizations online')
parser.add_argument('--gather_stats',
                    action='store_true',
                    help='Flag to gather stats')

args = parser.parse_args()

# Runtime Constants
ONLINE = args.online

# Functions
def make_graph(data, filename, layout=go.Layout()):
    fig = go.Figure(data=data, layout=layout)

    if ONLINE:
        py.iplot(fig, filename=filename)
    else:
        filename = './graphs/{}.png'.format(filename)
        py.image.save_as(fig, filename=filename)


def parse_stats(input_data):
    entry = {}
    parse = False

    lines = input_data.split('\n')
    for line in lines:
        if parse and line:
            name, data = line.split('=')
            entry[name] = data

        if line == 'PARSER_STATS_BEGIN':
            parse = True

    return entry


# Grab the data
dataset_stats = []
storage_filename = '.sda_stats.pickle'

if args.gather_stats:
    num_units = '50'

    for i in range(20):
        dataset = 'keyleak_random' if i % 2 == 0 else 'rootdir_random'
        train_file = 'data/{}/train.csv'.format(dataset)
        test_file = 'data/{}/test.csv'.format(dataset)

        proc = subprocess.Popen(['python', 'main-sda.py', '--train', '--testing',
                                 '--train_file', train_file,
                                 '--test_file', test_file,
                                 '--num_units', num_units, '--parser_stats',
                                 '--normalize'],
                                stdout=subprocess.PIPE)

        entry = parse_stats(proc.stdout.read().decode('utf-8'))
        entry['dataset'] = dataset

        dataset_stats.append(entry)

    with open(storage_filename, 'wb') as f:
        pickle.dump(dataset_stats, f)

else:
    with open(storage_filename, 'rb') as f:
        dataset_stats = pickle.load(f)

# Process the data and generate graphs
x = [float(entry['accuracy']) for entry in dataset_stats]
x_0 = [float(entry['accuracy'])
       for entry in dataset_stats if entry['dataset'] == 'keyleak_random']
x_1 = [float(entry['accuracy'])
       for entry in dataset_stats if entry['dataset'] == 'rootdir_random']

legendStr = '$\\text{{{0:25s}}}\\mu={1:.5f}, \\sigma={2:.5f}$'

data = [go.Box(x=x, name=legendStr.format('both', np.mean(x), np.std(x))),
        go.Box(x=x_0, name=legendStr.format('keyleak',
                                            np.mean(x_0), np.std(x_0))),
        go.Box(x=x_1, name=legendStr.format('rootdir',
                                            np.mean(x_1), np.std(x_1)))]

layout = go.Layout(xaxis=dict(title='Accuracy'),
                   title='Accuracy Distribution for the Stacked AutoEncoder')

make_graph(data=data, layout=layout, filename='sda-acc-box')
