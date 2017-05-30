""" Collection of helpful utilities for deep learning
    Author: Zander Blasingame """

import numpy as np
from functools import reduce

VALID_MODES = ['matrix', 'aggregate', 'middle']

def p(a, b):
    return a + b

parsing_op = {'matrix': lambda x: x,
              'aggregate': lambda x: [list(reduce(lambda a, b: map(p,a,b), x))],
              'middle': lambda x: [x[1]]}

# function parses csv and returns a list of input matrices and output labels
def parse_csv(filename, num_hpc=12, normalize=True, mode='matrix'):
    assert mode in VALID_MODES

    input_matricies = []
    output_labels = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            if line[0] == '#': continue

            line = line[:-1] # remove newline for string
            entries = line.split(',')

            output_labels.append(int(entries[0]))
            entries = entries[1:]

            input_matrix = [list(map(lambda a: float(a),
                                     entries[i*num_hpc:(i+1)*num_hpc]))
                            for i in range(int(len(entries) / num_hpc))]

            input_matrix = parsing_op[mode](input_matrix)

            # normalize matrix
            if normalize:
                input_matricies.append(input_matrix/np.linalg.norm(input_matrix))
            else:
                input_matricies.append(input_matrix)

    return np.array(input_matricies), np.array(output_labels)
