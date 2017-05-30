""" Python script to visualize the effects of the input methodology
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
storage_filename = '.lstm_stats.pickle'

datasets = ['keyleak', 'rootdir']
modes = ['matrix', 'middle', 'aggergate']

if args.gather_stats:
    for dataset in datasets:
        train_file = 'data/{}/train.csv'.format(dataset)
        test_file = 'data/{}/test.csv'.format(dataset)
        for mode in modes:
            for i in range(10):
                proc = subprocess.Popen(['python', 'main.py',
                                         '--train', '--testing',
                                         '--train_file', train_file,
                                         '--test_file', test_file,
                                         '--num_units', '45', '--parser_stats',
                                         '--normalize',
                                         '--mode', mode],
                                        stdout=subprocess.PIPE)

                entry = parse_stats(proc.stdout.read().decode('utf-8'))
                entry['dataset'] = dataset
                entry['mode'] = mode

                dataset_stats.append(entry)

    with open(storage_filename, 'wb') as f:
        pickle.dump(dataset_stats, f)

else:
    with open(storage_filename, 'rb') as f:
        dataset_stats = pickle.load(f)

# Process the data and generate graphs

# LSTM Visualizations
parsed_data = []

for dataset in datasets:
    for mode in modes:
        arr = []
        for entry in dataset_stats:
            if entry['mode'] == mode and entry['dataset'] == dataset:
                arr.append(float(entry['accuracy']))

        parsed_data.append(dict(dataset=dataset, mode=mode, arr=arr,
                                stddev=np.std(arr), mean=np.mean(arr)))

name_str = '$\\mu={:.3f}, \\sigma={:.3f}$'

data = [go.Box(x=entry['arr'],
               whiskerwidth=0.2,
               name='{}, {}: {}'.format(entry['mode'],
                                        entry['dataset'],
                                        name_str.format(entry['mean'],
                                                        entry['stddev'])))
       for entry in parsed_data]

layout = go.Layout(xaxis=dict(title='Accuracy'),
                   yaxis=dict(zeroline=False),
                   title='Accuracy Distribution LSTM-RNN')

make_graph(data=data, layout=layout, filename='lstm-acc-box')

for entry in parsed_data:
    print(entry)
