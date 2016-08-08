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

# Network parameters
learning_rate = 0.001
reg_param = 0
training_epochs = 4
display_step = 1

if args.train:
    trX, trY = parse_csv(args.train_file, num_hpc=12)

if args.testing:
    teX, teY = parse_csv(args.test_file, num_hpc=12)


num_input = len(trX[0][0]) if args.train else len(teX[0][0])
num_steps = len(trX[0]) if args.train else len(teX[0])
num_hidden = 128
num_out = 1

training_size = len(trX) if args.train else None
testing_size = len(teX) if args.testing else None


X = tf.placeholder('float', [num_steps, num_input])
Y = tf.placeholder('float')
keep_prob = tf.placeholder('float')

# define weights
# weights = dict(out=tf.Variable(tf.random_normal([num_hidden, num_out])))
# biases = dict(out=tf.Variable(tf.random_normal([num_out])))

# returns an rnn
def rnn(X, weights, biases):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True)

    # split input matrix by time steps
    input_list = tf.split(0, num_steps, X)

    outputs, states = tf.nn.rnn(lstm_cell, input_list, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']


# prediction = rnn(X, weights, biases)
cost = tf.reduce_mean(tf.square(prediction - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op) if args.train else saver.restore(sess, 'model.ckpt')

    if args.train:
        for epoch in range(training_epochs):
            avg_cost = 0
            for i in range(training_size):
                _, c = sess.run([optimizer, cost], feed_dict={X: trX[i],
                                                              Y: trY[i]})

                avg_cost += c / training_size

            if i % display_step == 0:
                print('Epoch: {0:03} with cost={1:.9f}'.format(epoch+1,
                                                               avg_cost))

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

        for i in range(testing_size):
            if teY[i] == 1:
                pos_size += 1
            else:
                neg_size += 1

            pred = sess.run(prediction, feed_dict={X: teX[i], Y: teY[i]})

            guess_label = 1 if pred > 0.5 else -1

            if i % 50 == 0:
                display_str = 'Prediction: {}\t\t\tLabel: {}\t\t\tGuess: {}'
                print(display_str.format(pred, teY[i], guess_label))

            if guess_label == teY[i]:
                accCount += 1
            elif teY[i] == 1:
                false_neg_count += 1
            else:
                false_pos_count += 1

        print('Accuracy: {:.2f}%'.format(100 * float(accCount) / testing_size))
        if pos_size != 0:
            false_neg_rate = 100 * float(false_neg_count) / pos_size
            print('false_neg_rate={}'.format(false_neg_rate))
        if neg_size != 0:
            false_pos_rate = 100 * float(false_pos_count) / neg_size
            print('false_pos_rate={}'.format(false_pos_rate))
