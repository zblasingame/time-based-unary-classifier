"""Class for the training and testing of an autoencoder
based classifier.

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
    """Unary classifier to detect anomalous behavior.

    Args:
        num_input (int):
            Number of input for classifier.
        batch_size (int = 100):
            Batch size.
        num_epochs (int = 10):
            Number of training epochs.
        debug (bool = False):
            Flag to print output.
        blacklist (list = []):
            List of features to ignore.
            Cannot be used if whitelist is being used.
        whitelist (list = []):
            List of features to use.
            Cannot be used if blacklist is being used.
        normalize (bool = False):
            Flag to determine if data is normalized.
        display_step (int = 1):
            How often to debug epoch data during training.
        std_param (int = 5):
            Value of the threshold constant for calculating the threshold.
    """

    def __init__(self, num_input, batch_size=100, num_epochs=10,
                 debug=False, blacklist=[], whitelist=[],
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
        self.batch_size         = batch_size
        self.debug              = debug
        self.normalize          = normalize
        self.blacklist          = blacklist
        self.whitelist          = whitelist

        assert not (self.blacklist and self.whitelist), (
            'Both whitelist and blacklist are defined'
        )

        ########################################
        # TensorFlow Variables                 #
        ########################################

        self.X = tf.placeholder('float', [None, num_input], name='X')
        self.Y = tf.placeholder('int32', [None], name='Y')
        self.keep_prob = tf.placeholder('float')

        # Cost threshold for anomaly detection
        self.cost_threshold = tf.Variable(0, dtype=tf.float32)

        # for normalization
        self.feature_min = tf.Variable(np.zeros(num_input), dtype=tf.float32)
        self.feature_max = tf.Variable(np.zeros(num_input), dtype=tf.float32)

        # Create Network
        network_sizes = [num_input, 25, 2, 25, num_input]
        activations = [tf.nn.relu, tf.nn.sigmoid, tf.nn.relu, tf.nn.sigmoid]

        self.neural_net = NeuralNet(network_sizes, activations)

        prediction = self.neural_net.create_network(self.X, self.keep_prob)

        self.cost = tf.reduce_mean(tf.square(prediction - self.X))
        self.cost += self.reg_param * self.neural_net.get_l2_loss()
        train_fn = tf.train.AdamOptimizer(learning_rate=self.l_rate)
        self.opt = train_fn.minimize(self.cost)

        # Evaluation meterics
        self.all_costs = tf.reduce_mean(tf.square(prediction - self.X), 1)

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

        # Variable ops
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        # for gpu
        self.config = tf.ConfigProto(log_device_placement=False)
        self.config.gpu_options.allow_growth = True

    def train(self, train_file='', reset_weights=False):
        """Trains classifier

        Args:
            train_file (str = ''):
                Training file location csv formatted,
                must consist of only regular behavior.
            reset_weights (bool = False):
                Flag to reset weights.
        """

        X, Y = load_data(train_file, self.blacklist, self.whitelist)
        training_size = X.shape[0]

        # normalize X
        if self.normalize:
            _min = X.min(axis=0)
            _max = X.max(axis=0)
            X = normalize(X, _min, _max)

        assert self.batch_size < training_size, (
            'batch size is larger than training_size'
        )

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)

            if reset_weights:
                sess.run(self.neural_net.reset_weights())

            costs = []

            for epoch in range(self.training_epochs):
                cost = 0
                num_costs = 0
                for batch_x, in gen_batches([X], self.batch_size):
                    _, c = sess.run([self.opt, self.cost], feed_dict={
                        self.X: batch_x,
                        self.keep_prob: self.dropout_prob
                    })

                    cost += c
                    num_costs += 1

                    # calculate average cost on last epoch for threshold
                    if epoch == self.training_epochs - 1:
                        costs.append(c)

                if epoch % self.display_step == 0:
                    display_str = 'Epoch {0:04} with cost={1:.9f}'
                    display_str = display_str.format(epoch+1, cost/num_costs)
                    self.print(display_str)

            # assign cost threshold
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

    def test(self, test_file=''):
        """Tests classifier

        Args:
            test_file (str = ''):
                Testing file location csv formatted.

        Returns:
            (dict):
                Dictionary containing the following fields:
                    accuracy, tp_rate, fp_rate, fn_rate, and tn_rate.
        """

        X, Y = load_data(test_file, self.blacklist, self.whitelist)

        rtn_dict = {
            'num_acc': 0,
            'num_fp': 0,
            'num_tn': 0,
            'num_fn': 0,
            'num_tp': 0
        }

        with tf.Session(config=self.config) as sess:
            self.saver.restore(sess, './model.ckpt')

            # normalize data
            if self.normalize:
                _min = self.feature_min.eval()
                _max = self.feature_max.eval()

                X = normalize(X, _min, _max)

            perf = sess.run([self.tp_rate, self.tn_rate,
                             self.fp_rate, self.fn_rate,
                             self.tp_num, self.tn_num,
                             self.fp_num, self.fn_num,
                             self.accuracy], feed_dict={
                self.X: X,
                self.Y: Y,
                self.keep_prob: 1.0
            })

            rtn_dict = {}
            # rtn_dict['tp_rate']     = float(perf[0] * 100)
            # rtn_dict['tn_rate']     = float(perf[1] * 100)
            # rtn_dict['fp_rate']     = float(perf[2] * 100)
            # rtn_dict['fn_rate']     = float(perf[3] * 100)
            rtn_dict['tp_num']      = int(perf[4])
            rtn_dict['tn_num']      = int(perf[5])
            rtn_dict['fp_num']      = int(perf[6])
            rtn_dict['fn_num']      = int(perf[7])
            rtn_dict['accuracy']    = float(perf[8] * 100)

            self.print(json.dumps(rtn_dict, indent=4))

            return rtn_dict

    def print(self, val):
        """Internal function for printing"""

        if self.debug:
            print(val)


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
        (np.ndarray): Normalized features of the same shape as data
    """

    new_data = (data - _min) / (_max - _min)

    # check if feature is constant, will be nan in new_data
    np.place(new_data, np.isnan(new_data), 1)

    return new_data


def load_data(filename, blacklist=[], whitelist=[]):
    """Returns the features of a dataset

    Args:
        filename (str):
            File location (csv formatted).
        blacklist (list = []):
            List of features to ignore,
            cannot be used if whitelist is being used.
        whitelist (list = []):
            List of features to use,
            cannot be used if blacklist is being used.

    Returns:
        (tuple of np.ndarray):
            Tuple consisiting of the features, X, and the labels, Y.
    """

    assert not (blacklist and whitelist), (
        'Both whitelist and blacklist are defined'
    )

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = np.array([row for row in reader
                         if '#' not in row[0]]).astype(np.float32)

    X = data[:, 1:]
    Y = data[:, 0]

    return X, Y


def gen_batches(data, batch_size=100):
    """Creates a generator to yield batches of batch_size.
    When batch is too large to fit remaining data the batch
    is clipped.

    Args:
        data (List of np.ndarray):
            List of data elements to be batched. The first dimension
            must be the batch size and the same for all data elements.
        batch_size (int = 100):
            Size of the mini batches.
    Yields:
        The next mini_batch in the dataset.
    """

    batch_start = 0
    batch_end   = batch_size

    while batch_end < data[0].shape[0]:
        yield [el[batch_start:batch_end] for el in data]

        batch_start = batch_end
        batch_end   += batch_size

    yield [el[batch_start:] for el in data]


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
    parser.add_argument('--whitelist', '-w',
                        type=str,
                        help='Location of the whitelist file (csv formatted)')
    parser.add_argument('--normalize', '-n',
                        action='store_true',
                        help='Flag to normalize features')

    args = parser.parse_args()

    filename = args.train_file if args.train_file else args.test_file

    whitelist = []
    if args.whitelist:
        with open(args.whitelist, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                whitelist = row

    X, Y = load_data(filename, whitelist=whitelist)
    num_input = len(X[0])

    classifier = Classifier(num_input, args.batch_size, args.epochs,
                            debug=True, whitelist=whitelist,
                            normalize=args.normalize)

    if args.train_file:
        classifier.train(args.train_file)

    if args.test_file:
        classifier.test(args.test_file)
