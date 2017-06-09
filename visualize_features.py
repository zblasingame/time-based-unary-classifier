"""Script to visualize feature comparisions using accuracy

Author:         Zander Blasingame
Insitution:     Clarkson University
Lab:            CAMEL
"""

import csv
import json
import plotly.plotly as py
import plotly.graph_objs as go

dataset_stats = []

with open('./data/feature_test.csv', 'r') as f:
    dataset_stats = [row for row in csv.DictReader(f)]

features = list(set([entry['feature_set'] for entry in dataset_stats]))

accuracies = [[entry['accuracy'] for entry in dataset_stats
               if entry['feature_set'] == feature]
              for feature in features]

data = [go.Box(y=accuracies[i],
               whiskerwidth=0.2,
               name=str(feature))
        for i, feature in enumerate(features)]

layout = go.Layout(title='Averaged Accuracy vs Feature Set',
                   yaxis=dict(title='Accuracy (%)'))

fig = go.Figure(data=data, layout=layout)

plot_url = py.plot(fig, filename=(
    'time-series-paper-feature-set-comparison-exploit-averged'))
