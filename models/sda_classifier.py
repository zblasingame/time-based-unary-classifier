"""Trains and tests a SDA classifier

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import argparse
import csv
import json
import numpy as np
import tensorflow as tf

from models.NeuralNet import NeuralNet


class Classifier:
    """Unary time-series classifier using a SDA model

    Args:
        num_input (int = 4):
            Number of inputs for each time series step.
        num_units (int = 10):
            Number of hiddent units.
        num_steps (int = 3):
            Number of time series steps.
        batch_size (int = 100):
            Size of the mini batch.
        num_epochs (int = 10):
            Number of training epochs
        debug (bool = False):
            Flag to print the output.
        normalize (bool = False):
            Flag to determine if the input data is normalized.
        display_step (int = 1):
            How often to debug epoch data.
        std_param (int = 5):
            Value of the threshold constant for calculating the threshold.
    """
    def __init__(self, num_input=4, num_units=10, num_steps=3,
                 batch_size=100, num_epochs=10, debug=False,
                 normalize=False, display_step=1, std_param=5):
        """Init classifier"""

        ########################################
        # Network Parameters                   #
        ########################################

        self.l_rate             = 0.001
        self.dropout_prob       = 0.5
        self.reg_param          = 0.01
        self.std_param          = std_param
        self.training_epochs    = num_epochs
        self.display_step       = display_step
        self.debug              = debug
        self.normalize          = normalize
        self.num_input          = num_input
        self.num_steps          = num_steps
        self.batch_size         = batch_size

        ########################################
        # TensorFlow Variables                 #
        ########################################

        self.X = tf.placeholder('float', [None, num_input], name='X')
        self.Y = tf.placeholder('int32', [None], name='Y')
        self.Z = tf.placeholder('float',
                                [None, num_steps*num_input], name='Z')
        self.Z_gen = tf.placeholder('float',
                                    [None, num_steps*num_input], name='Z_gen')
        self.keep_prob = tf.placeholder('float')

        # Cost threshold
        self.cost_threshold = tf.Variable(0, dtype=tf.float32)

        # for normalization
        self.feature_min = tf.Variable(np.zeros(num_input*num_steps),
                                       dtype=tf.float32)
        self.feature_max = tf.Variable(np.zeros(num_input*num_steps),
                                       dtype=tf.float32)

        self.compression_layer = []

        for i in range(num_steps):
            with tf.variable_scope('compression_layer-{}'.format(i)):
                net = NeuralNet([num_input, num_units, num_input],
                                [tf.nn.sigmoid, tf.identity])

                prediction = net.create_network(self.X, self.keep_prob)
                cost = tf.reduce_mean(tf.square(prediction - self.X))
                cost += self.reg_param * net.get_l2_loss()
                train_fn = tf.train.AdamOptimizer(learning_rate=self.l_rate)
                opt = train_fn.minimize(cost)

                self.compression_layer.append(dict(
                    prediction=prediction,
                    cost=cost,
                    opt=opt,
                    net=net
                ))

        self.net = NeuralNet(
            [num_input*num_steps, num_units, num_input*num_steps],
            [tf.nn.sigmoid, tf.identity]
        )

        self.prediction = self.net.create_network(self.Z_gen, self.keep_prob)
        self.cost = tf.reduce_mean(tf.square(self.prediction - self.Z))
        self.cost += self.reg_param * self.net.get_l2_loss()
        train_fn = tf.train.AdamOptimizer(learning_rate=self.l_rate)
        self.opt = train_fn.minimize(self.cost)

        # Evaluation meterics
        self.all_costs = tf.reduce_mean(tf.square(self.prediction - self.Z), 1)

        negative_labels = tf.fill(tf.shape(self.Y), -1)
        positive_labels = tf.fill(tf.shape(self.Y), 1)

        self.pred_labels = tf.where(
            tf.less(self.all_costs, tf.fill(
                tf.shape(self.all_costs),
                self.cost_threshold
            )),
            positive_labels,
            negative_labels
        )

        self.accuracy = tf.reduce_mean(
            tf.to_float(tf.equal(self.pred_labels, self.Y))
        )

        self.tp_num = tf.reduce_sum(tf.to_float(tf.logical_and(
            tf.equal(self.pred_labels, negative_labels),
            tf.equal(self.Y, negative_labels)
        )))

        self.tn_num = tf.reduce_sum(tf.to_float(tf.logical_and(
            tf.equal(self.pred_labels, positive_labels),
            tf.equal(self.Y, positive_labels)
        )))

        self.fp_num = tf.reduce_sum(tf.to_float(tf.logical_and(
            tf.equal(self.pred_labels, negative_labels),
            tf.equal(self.Y, positive_labels)
        )))

        self.fn_num = tf.reduce_sum(tf.to_float(tf.logical_and(
            tf.equal(self.pred_labels, positive_labels),
            tf.equal(self.Y, negative_labels)
        )))

        self.tp_rate = tf.div(self.tp_num, self.tp_num + self.fn_num)
        self.tn_rate = tf.div(self.tn_num, self.tn_num + self.fp_num)
        self.fp_rate = tf.div(self.fp_num, self.fp_num + self.tn_num)
        self.fn_rate = tf.div(self.fn_num, self.fn_num + self.tp_num)

        # init and saver
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        # for gpu
        self.config = tf.ConfigProto(log_device_placement=False)
        self.config.gpu_options.allow_growth = False

    def train(self, train_file, reset_weights=False):
        """Trains classifier

        Args:
            train_file (str):
                Location of csv formatted training file.
            reset_weights (bool = False):
                Flag to reset the weights for the entire network
        """

        X, Y = grab_data(train_file)
        training_size = X.shape[0]

        # normalize input data
        if self.normalize:
            _min = X.min(axis=0)
            _max = X.max(axis=0)
            X = normalize(X, _min, _max)

        X_mat = np.reshape(X, (training_size,
                               self.num_steps,
                               self.num_input))

        assert self.batch_size < training_size, (
            'batch size is larger than training size'
        )

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)

            if reset_weights:
                sess.run(self.net.reset_weights())
                for node in self.compression_layer:
                    sess.run(node['net'].reset_weights())

            # Train the compression layer
            for epoch in range(self.training_epochs):
                for j, node in enumerate(self.compression_layer):
                    cost = 0
                    num_costs = 0

                    for i in range(0, training_size, self.batch_size):
                        # for batch training
                        end_batch = i + self.batch_size
                        if end_batch >= training_size:
                            end_batch = training_size - 1

                        feed_dict = {
                            self.X: X_mat[i:end_batch][:, j],
                            self.keep_prob: self.dropout_prob
                        }

                        _, c = sess.run([node['opt'], node['cost']],
                                        feed_dict=feed_dict)

                        cost += c
                        num_costs += 1

                    if epoch % self.display_step == 0:
                        print_str = 'Optimization for Compression Node: {2}\n'
                        print_str += 'Epoch {0:04} with cost {1:.9f}'
                        print_str = print_str.format(
                            epoch+1,
                            cost/num_costs,
                            j
                        )
                        self.print(print_str)

            costs = []

            self.print('Optimizing the final layer...')

            # Train the final layer
            for epoch in range(self.training_epochs):
                cost = 0
                num_costs = 0
                for i in range(0, training_size, self.batch_size):
                    # for batch training
                    end_batch = i + self.batch_size
                    if end_batch >= training_size:
                        end_batch = training_size - 1

                    Z_gen = np.array([sess.run(
                        self.compression_layer[j]['prediction'],
                        feed_dict={
                            self.X: X_mat[i:end_batch][:, j],
                            self.keep_prob: self.dropout_prob
                        }
                    ) for j in range(self.num_steps)])

                    Z_gen = np.swapaxes(Z_gen, 0, 1)
                    Z_gen = np.reshape(
                        Z_gen,
                        (Z_gen.shape[0], self.num_input*self.num_steps)
                    )

                    feed_dict = {
                        self.Z: X[i:end_batch],
                        self.Z_gen: Z_gen,
                        self.keep_prob: self.dropout_prob
                    }

                    _, c = sess.run([self.opt, self.cost],
                                    feed_dict=feed_dict)

                    cost += c
                    num_costs += 1

                    if epoch == self.training_epochs - 1:
                        costs.append(c)

                if epoch % self.display_step == 0:
                    print_str = 'Epoch {0:04} with cost {1:.9f}'
                    print_str = print_str.format(
                        epoch+1,
                        cost/num_costs
                    )
                    self.print(print_str)

            cost_threshold = np.mean(costs) + self.std_param * np.std(costs)
            sess.run(self.cost_threshold.assign(cost_threshold))

            self.print('Threshold: ' + str(cost_threshold))

            # assign normalization values
            if self.normalize:
                sess.run(self.feature_min.assign(_min))
                sess.run(self.feature_max.assign(_max))

            self.print('Optimization Finished')

            # save model
            save_path = self.saver.save(sess, './model.ckpt')
            self.print('Model saved in file: {}'.format(save_path))

    def test(self, test_file):
        """Tests classifier

        Args:
            test_file (str):
                Location of the test file.
        Returns:
            (dict): Dictionary containing the following fields
                accuracy
                false positive rate
                false negative rate
                true positive rate
                true negative rate
        """

        X, Y = grab_data(test_file)

        testing_size = X.shape[0]

        with tf.Session(config=self.config) as sess:
            self.saver.restore(sess, './model.ckpt')

            # normalize data
            if self.normalize:
                _min = self.feature_min.eval()
                _max = self.feature_max.eval()

                X = normalize(X, _min, _max)

            X_mat = np.reshape(X, (testing_size,
                                   self.num_steps,
                                   self.num_input))

            Z_gen = np.array([sess.run(
                self.compression_layer[j]['prediction'],
                feed_dict={
                    self.X: X_mat[:, j],
                    self.keep_prob: 1.0
                }
            ) for j in range(self.num_steps)]).swapaxes(0, 1)

            Z_gen = Z_gen.reshape(
                (Z_gen.shape[0], self.num_input*self.num_steps)
            )

            perf = sess.run([self.tp_rate, self.tn_rate,
                             self.fp_rate, self.fn_rate,
                             self.tp_num, self.tn_num,
                             self.fp_num, self.fn_num,
                             self.accuracy], feed_dict={
                self.Z: X,
                self.Z_gen: Z_gen,
                self.Y: Y,
                self.keep_prob: 1.0
            })

            rtn_dict = {}

            rtn_dict['tp_rate']     = float(perf[0] * 100)
            rtn_dict['tn_rate']     = float(perf[1] * 100)
            rtn_dict['fp_rate']     = float(perf[2] * 100)
            rtn_dict['fn_rate']     = float(perf[3] * 100)
            rtn_dict['tp_num']      = int(perf[4])
            rtn_dict['tn_num']      = int(perf[5])
            rtn_dict['fp_num']      = int(perf[6])
            rtn_dict['fn_num']      = int(perf[7])
            rtn_dict['accuracy']    = float(perf[8] * 100)

            self.print(json.dumps(rtn_dict, indent=4))

        return rtn_dict

    def print(self, msg):
        """Internal function for printing"""

        if self.debug:
            print(msg)


def normalize(data, _min, _max):
    """Function to normalize a dataset of features
    Args:
        data (np.ndarray):
            Feature matrix.
        _min (list):
            List of minimum values per feature.
        _max (list):
            List of maximum values per feature.
    Returns:
        (np.ndarray):
            Normalized features of the same shape as data.
    """

    new_data = (data - _min) / (_max - _min)

    # check if feature is constant, will be nan in new_data
    np.place(new_data, np.isnan(new_data), 1)

    return new_data


def grab_data(filename):
    """Returns the features of a dataset
    Args:
        filename (str):
            Number of time steps in csv file.
    Returns:
        (tuple of np.ndarray): Tuple consisiting of the features, X
            and the labels, Y
    """

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = np.array([row for row in reader]).astype(np.float32)

    X = data[:, 1:]
    Y = data[:, 0]

    return X, Y


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', '-r',
                        type=str,
                        default=None,
                        help='Location of training file')
    parser.add_argument('--test_file', '-t',
                        type=str,
                        default=None,
                        help='Location of testing file')
    parser.add_argument('--batch_size', '-b',
                        type=int,
                        default=100,
                        help='Size of batch for training (mini SGD)')
    parser.add_argument('--epochs', '-e',
                        type=int,
                        default=10,
                        help='Number of training epochs')
    parser.add_argument('--normalize', '-n',
                        action='store_true',
                        help='Flag to normalize features')

    args = parser.parse_args()

    filename = args.train_file if args.train_file else args.test_file

    X, Y = grab_data(filename)
    num_input = X.shape[0]

    classifier = Classifier(num_input, 10, 3, args.batch_size, args.epochs,
                            debug=True, normalize=args.normalize)

    if args.train_file:
        classifier.train(args.train_file)

    if args.test_file:
        classifier.test(args.test_file)
