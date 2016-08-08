""" Author: Zander Blasingame
    Location: CAMEL at Clarkson University
    Purpose: LSTM RNN for the purpose of recognizing malicious
            hardware requests.
    Documentation: Enter `python main.py --help` """

import argparse
import sys
import numpy as np
import tensorflow as tf

from utils import parse_cvs

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train',
                    action='store_true',
                    help='Flag to train neural net on dataset')
parser.add_argument('--graphing',
                    action='store_true',
                    help='Flag to turn create visualizations')
parser.add_argument('--testing',
                    action='store_true',
                    help='Flag to turn on testing')
parser.add_argument('--train_file',
                    type=str,
                    help='Location of training file')
parser.add_argument('--test_file',
                    type=str,
                    help='Location of testing file');

args = parser.parse_args()

