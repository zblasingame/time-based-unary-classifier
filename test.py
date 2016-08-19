""" Python script to visualize the effects of the unit layer size on the
    Neural Network.
    Author: Zander Blasingame """

import os
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
parser.add_argument('dataset',
                    type=str,
                    help='The dataset to perform training and testing')
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
storage_filename = '.{}_stats.pickle'.format(args.dataset)

if args.gather_stats:
    train_file = 'data/{}/train.csv'.format(args.dataset)
    test_file = 'data/{}/test.csv'.format(args.dataset)

    for i in np.arange(0, 1.05, 0.05):
        for j in range(10):
            proc = subprocess.Popen(['python', 'main-sda.py', '--train', '--testing',
                                     '--train_file', train_file,
                                     '--test_file', test_file,
                                     '--num_units', str(i), '--parser_stats',
                                     '--normalize'],
                                    stdout=subprocess.PIPE)

            entry = parse_stats(proc.stdout.read().decode('utf-8'))
            entry['num_units'] = i

            dataset_stats.append(entry)

    with open(storage_filename, 'wb') as f:
        pickle.dump(dataset_stats, f)

else:
    with open(storage_filename, 'rb') as f:
        dataset_stats = pickle.load(f)

# Process the data and generate graphs

# Box plot of num_unit vs accuracy
num_units = sorted(list(set([entry['num_units'] for entry in dataset_stats])))

colors = ['hsl('+str(h)+',50%, 50%)'
          for h in np.linspace(0, 360, len(num_units))]

data = [go.Box(x=num,
               y=[entry['accuracy']
                  for entry in dataset_stats if entry['num_units'] == num],
               whiskerwidth=0.2,
               marker=dict(color=colors[i]),
               name=str(num))
        for i, num in enumerate(num_units)]

layout = go.Layout(xaxis=dict(type='linear', showgrid=True,
                              range=num_units, dtick=5,
                              title='Noise Parameter'),
                   yaxis=dict(zeroline=False,
                              title='Accuracy'),
                   title='Noise (Stacked Denoising AutoEncoder Vs. Accuracy')

make_graph(data=data, layout=layout, filename='{}-acc-sda-box'.format(args.dataset))
