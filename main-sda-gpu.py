""" Author: Zander Blasingame
    Location: CAMEL at Clarkson University
    Purpose: Stacked Denoising AutoEncoder for recognizing malicious hpc
    Documentation: Enter `python main-sda.py --help` """

import argparse
import sys
import numpy as np
import tensorflow as tf

from utils import parse_csv
from networks import create_network

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
parser.add_argument('--num_units',
                    type=int,
                    default=None,
                    help='Number of units in the LSTM-RNN')
parser.add_argument('--parser_stats',
                    action='store_true',
                    help='Flag to print results in a parser friendly format')
parser.add_argument('--normalize',
                    action='store_true',
                    help='Flag to normalize input data')
parser.add_argument('--mode',
                    type=str,
                    default='matrix',
                    help='Type of input mode')
parser.add_argument('--hpc',
                    type=int,
                    default=12,
                    help='How many perfomance indicators in the dataset')
parser.add_argument('--debug',
                    action='store_true',
                    help='Flag to debug program')

args = parser.parse_args()

mode = args.mode

# helper function
def debug(msg):
    if args.debug:
        print('DEBUG: {}'.format(msg))


normalize = False if not args.normalize else True
if args.train:
    trX, trY = parse_csv(args.train_file, num_hpc=args.hpc,
                         normalize=normalize, mode=mode)

if args.testing:
    teX, teY = parse_csv(args.test_file, num_hpc=args.hpc,
                         normalize=normalize, mode=mode)


# Network parameters
learning_rate = 0.001
reg_param = 0.0
noise_param_value = 1.0
dropout_prob = 1.0
training_epochs = 4
display_step = 1
std_pram = 1.0
num_input = len(trX[0][0]) if args.train else len(teX[0][0])
num_steps = len(trX[0]) if args.train else len(teX[0])
num_units = 15 if args.num_units == None else args.num_units
num_out = num_input
training_size = len(trX) if args.train else None
testing_size = len(teX) if args.testing else None

# Placeholders
X = tf.placeholder('float', [num_input])
Z = tf.placeholder('float', [num_input * num_steps])
Z_auto = tf.placeholder('float', [num_input * num_steps])
Y = tf.placeholder('float')
keep_prob = tf.placeholder('float')
noise_param = tf.placeholder('float')
cost_threshold = tf.Variable([0, 0], dtype=tf.float32)

# Create Networks
model_name = 'Stacked Denoising AutoEncoder'

gpu_id = 0

# Create an Denoising AE for each time step
compression_layer = []
for i in range(num_steps):
    # alternate GPU assignment
    gpu_id = 0 if i % 2 == 0 else 1
    with tf.device('/gpu:{}'.format(gpu_id)):
        network_params = {'keep_prob': keep_prob,
                          'reg_param': reg_param,
                          'noise_param': noise_param,
                          'sizes': [num_input, num_units, num_input],
                          'activations': [tf.nn.sigmoid, tf.identity]}

        network = create_network('DenoisingAutoEncoder', X, X, network_params)

        prediction = network.create_prediction()
        cost = network.create_cost()
        optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate).minimize(cost)

        denoisingAE = dict(prediction=prediction,
                           cost=cost,
                           name='in_layer_denoising_ae_{}'.format(i+1),
                           optimizer=optimizer)

        compression_layer.append(denoisingAE)

with tf.devicei('/gpu:0'):
    network_params = {'keep_prob': keep_prob,
                      'reg_param': reg_param,
                      'noise_param': noise_param,
                      'sizes': [num_steps*num_input, num_units,
                                num_steps*num_input],
                      'activations': [tf.nn.sigmoid, tf.identity]}

    network = create_network('DenoisingAutoEncoder',
                             Z, Z_auto, network_params)

    prediction = network.create_prediction()
    cost = network.create_cost()
    name = 'denoising_ae'
    optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(cost)

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

# config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

with tf.Session(config=config) as sess:
    sess.run(init_op) if args.train else saver.restore(sess, 'model.ckpt')

    if args.train:
        # Train compression layer
        for epoch in range(training_epochs):
            for j in range(num_steps):
                avg_cost = 0
                for i in range(training_size):
                    feed_dict = {X: trX[i][j],
                                 keep_prob: dropout_prob,
                                 noise_param: noise_param_value}

                    _, c = sess.run([compression_layer[j]['optimizer'],
                                     compression_layer[j]['cost']],
                                    feed_dict=feed_dict)

                    debug('cost: {}'.format(c))

                    avg_cost += c / training_size

                if i % display_step == 0:
                    print_str = 'epoch={1:03}\tname={0:30s}\tcost={2:.9f}'
                    print(print_str.format(compression_layer[j]['name'],
                                           epoch+1,
                                           avg_cost))

        costs = []
        for epoch in range(training_epochs):
            avg_cost = 0
            for i in range(training_size):
                feed_dict = {Z: np.array([sess.run(
                                          compression_layer[j]['prediction'],
                                          feed_dict={X: trX[i][j],
                                                     keep_prob: 1.0,
                                                     noise_param: 1.0})
                                 for j in range(num_steps)]).flatten(),
                             Z_auto: trX[i].flatten(),
                             keep_prob: dropout_prob,
                             noise_param: noise_param_value}

                _, c = sess.run([optimizer,
                                 cost],
                                 feed_dict=feed_dict)

                avg_cost += c / training_size

                if epoch == training_epochs - 1:
                    costs.append(c)

            if i % display_step == 0:
                print('General Epoch: {0:03} with cost={1:.9f}'.format(epoch+1,
                                                               avg_cost))


        # calculate cost threshold
        sess.run(cost_threshold.assign([np.mean(costs)-std_pram*np.std(costs),
                                        np.mean(costs)+std_pram*np.std(costs)]))

        print('Optimization Finished')

        # save model
        save_path = saver.save(sess, 'model.ckpt')
        print('Model saved in file: {}'.format(save_path))

    if args.testing:
        accCount = 0
        pos_size = 0
        neg_size = 0
        false_pos_count = 0
        false_neg_count = 0

        avg_pos_cost = 0
        avg_neg_cost = 0

        bounds = cost_threshold.eval()

        for i in range(testing_size):
            feed_dict = {Z: np.array([sess.run(
                                      compression_layer[j]['prediction'],
                                      feed_dict={X: teX[i][j],
                                                 keep_prob: 1.0,
                                                 noise_param: 1.0})
                             for j in range(num_steps)]).flatten(),
                         Z_auto: teX[i].flatten(),
                         keep_prob: 1.0,
                         noise_param: 1.0}

            cost_val = sess.run(cost, feed_dict=feed_dict)

            guess_label = 1 if bounds[1] > float(cost_val) > bounds[0] else -1

            if i % 50 == 0 and not args.parser_stats:
                display_str = 'Prediction: {:.10f}\t\tLabel: {}\t\tGuess: {:2}'
                print(display_str.format(float(cost_val), teY[i], guess_label))

            if teY[i] == 1:
                pos_size += 1
                avg_pos_cost += float(cost_val)
            else:
                neg_size += 1
                avg_neg_cost += float(cost_val)


            if guess_label == teY[i]:
                accCount += 1
            elif teY[i] == 1:
                false_neg_count += 1
            else:
                false_pos_count += 1

        if args.parser_stats:
            print('PARSER_STATS_BEGIN')

        print('model_name={}'.format(model_name))
        print('parsing_mode={}'.format(mode))
        print('accuracy={:.2f}'.format(100 * float(accCount) / testing_size))
        if pos_size != 0:
            false_neg_rate = 100 * float(false_neg_count) / pos_size
            avg_pos_cost /= pos_size
            print('false_neg_rate={:.2f}'.format(false_neg_rate))
            print('avg_pos_cost={}'.format(avg_pos_cost))
        if neg_size != 0:
            false_pos_rate = 100 * float(false_pos_count) / neg_size
            avg_neg_cost /= neg_size
            print('false_pos_rate={:.2f}'.format(false_pos_rate))
            print('avg_neg_cost={}'.format(avg_neg_cost))
        print('upper={}'.format(bounds[1]))
        print('lower={}'.format(bounds[0]))
