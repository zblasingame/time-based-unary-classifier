""" File to create training and testing data sets out of existing files.
    Author: Zander Blasingame """

import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('p_train',
                    type=float,
                    default=0.5,
                    help=('Float between 0 and 1 that describes the percentage '
                          'of training data to be pulled from the dataset'))
parser.add_argument('train_path',
                    type=str,
                    help='Location of output train file')
parser.add_argument('test_path',
                    type=str,
                    help='Location of output test file')
parser.add_argument('input_file',
                    type=str,
                    help='Location of input file')
parser.add_argument('--even_split',
                    action='store_true',
                    help=('Flag to ensure test set has even number of both '
                          'types of labels'))

args = parser.parse_args()

lines = []

with open(args.input_file, 'r') as f:
    for line in f.readlines():
        lines.append(line)


pos_entries = [line for line in lines if line[0] == '1']
neg_entries = [line for line in lines if line[0] == '-']
num_pos_entries = len(pos_entries)
num_neg_entries = len(neg_entries)
total_size = num_pos_entries + num_neg_entries
max_percent = float(num_pos_entries) / total_size
training_percentage = max_percent if args.p_train > max_percent else args.p_train

training_data = []
testing_data = []

if not args.even_split:
    stop_index = int(training_percentage * num_pos_entries)
    training_data.extend(pos_entries[:stop_index])
    testing_data.extend(pos_entries[-stop_index:])
else:
    if num_neg_entries > 0.5 * num_pos_entries:
        print('WARNING: too many negative test cases!')
    
    training_data.extend(pos_entries[:num_neg_entries])
    testing_data.extend(pos_entries[-num_neg_entries:])

testing_data.extend(neg_entries)

# write new files
with open(args.train_path, 'w', encoding='utf-8') as f:
    f.writelines(training_data)

with open(args.test_path, 'w', encoding='utf-8') as f:
    f.writelines(testing_data)
