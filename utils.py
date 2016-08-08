""" Collection of helpful utilities for deep learning
    Author: Zander Blasingame """

import numpy as np

# function parses csv and returns a list of input matrices and output labels
def parse_csv(filename, num_hpc=12):
    input_matricies = []
    output_labels = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            if line[0] == '#': continue

            line = line[:-1] # remove newline for string
            entries = line.split(',')

            output_labels.append(int(entries[0]))
            entries = entries[1:]

            input_matricies.append([list(map(lambda a: float(a),
                                        entries[i*num_hpc:(i+1)*num_hpc]))
                                    for i in range(int(len(entries) / num_hpc))])


    return np.array(input_matricies), np.array(output_labels)
