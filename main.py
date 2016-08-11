""" Author: Zander Blasingame
    Location: CAMEL at Clarkson University
    Purpose: LSTM rnn for the purpose of recognizing malicious
            hardware requests.
    Documentation: Enter `python main.py --help` """

import argparse
import sys
import numpy as np
import tensorflow as tf

from utils import parse_csv

from MLP import MLP

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

args = parser.parse_args()

# Network parameters
learning_rate = 0.001
reg_param = 0.0
dropout_prob = 1.0 
training_epochs = 4 
display_step = 1
std_pram = 1.0 

if args.train:
    trX, trY = parse_csv(args.train_file, num_hpc=12, normalize=True)

if args.testing:
    teX, teY = parse_csv(args.test_file, num_hpc=12, normalize=True)


num_input = len(trX[0][0]) if args.train else len(teX[0][0])
num_steps = len(trX[0]) if args.train else len(teX[0])
num_units = 15 if args.num_units == None else args.num_units  
num_out = 1

training_size = len(trX) if args.train else None
testing_size = len(teX) if args.testing else None


X = tf.placeholder('float', [num_steps, num_input])
# X = tf.placeholder('float', [num_steps * num_input])
Y = tf.placeholder('float')
keep_prob = tf.placeholder('float')
cost_threshold = tf.Variable([0, 0], dtype=tf.float32)

# define weights
weights = dict(out=tf.Variable(tf.random_normal([num_units, num_out])))
biases = dict(out=tf.Variable(tf.random_normal([num_out])))

# returns an rnn
def rnn(X, weights, biases):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)

    # split input matrix by time steps
    input_list = tf.split(0, num_steps, X)

    outputs, states = tf.nn.rnn(lstm_cell, input_list, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']


# Define AutoEncoder
# ae = MLP([num_input * num_steps, 25, 2, 25, num_input * num_steps],
#          [tf.nn.relu, tf.nn.sigmoid, tf.nn.relu, tf.identity])

#cost = tf.reduce_mean(tf.square(prediction - Y)) + reg_param * mlp.get_l2_loss()
prediction = rnn(X, weights, biases)
cost = tf.reduce_mean(tf.square(prediction - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op) if args.train else saver.restore(sess, 'model.ckpt')

    if args.train:
        costs = []
        for epoch in range(training_epochs):
            avg_cost = 0
            for i in range(training_size):
                if trY[i] == 1:
                    feed_dict = {X: trX[i], Y: trY[i]}
                    _, c, pred = sess.run([optimizer, cost, prediction],
                                          feed_dict=feed_dict)

                    avg_cost += c / training_size

                    if epoch == training_epochs - 1:
                        costs.append(pred)

            if i % display_step == 0:
                print('Epoch: {0:03} with cost={1:.9f}'.format(epoch+1,
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

        for i in range(testing_size):
            feed_x = teX[i]#.flatten()
            pred = sess.run(prediction, feed_dict={X: feed_x,
                                                   keep_prob: 1.0})

            bounds = cost_threshold.eval()

            guess_label = 1 if bounds[1] > float(pred) > bounds[0] else -1

            if i % 50 == 0 and not args.parser_stats:
                display_str = 'Prediction: {:.10f}\t\tLabel: {}\t\tGuess: {:2}'
                print(display_str.format(float(pred), teY[i], guess_label))

            if teY[i] == 1:
                pos_size += 1
                avg_pos_cost += float(pred)
            else:
                neg_size += 1
                avg_neg_cost += float(pred)


            if guess_label == teY[i]:
                accCount += 1
            elif teY[i] == 1:
                false_neg_count += 1
            else:
                false_pos_count += 1

        if args.parser_stats:
            print('PARSER_STATS_BEGIN')

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
