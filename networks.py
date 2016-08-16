""" File to create different neural nets
    Author: Zander Blasingame """

from Network import NeuralNet
from functools import reduce
import tensorflow as tf

class LSTM_RNN(NeuralNet):
    def __init__(self, X, Y, network_params):
        self.X = X
        self.params = network_params
        self.prediction = self.__rnn()
        self.cost = tf.reduce_mean(tf.square(self.prediction - Y))

    def __rnn(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.params['num_units'],
                                                 state_is_tuple=True)

        # split input matrix by time steps
        input_list = tf.split(0, self.params['num_steps'], self.X)

        outputs, states = tf.nn.rnn(lstm_cell, input_list, dtype=tf.float32)

        weights = tf.Variable(tf.random_normal([self.params['num_units'],
                                                self.params['num_out']]))
        biases = tf.Variable(tf.random_normal([self.params['num_out']]))

        return tf.matmul(outputs[-1], weights) + biases

    def create_prediction(self):
        return self.prediction

    def create_cost(self):
        return self.cost


class MLP(NeuralNet):
    def __init__(self, X, Y, network_params):
        self.model = self.__gen_model(network_params)
        self.prediction = self.__mlp(X, network_params)

        l2_loss = self.__gen_l2_loss()
        reg_param = network_params['reg_param']
        self.cost = tf.reduce_mean(tf.square(self.prediction - Y)) + reg_param * l2_loss

    def __gen_model(self,network_params):
        sizes = network_params['sizes']
        activations = network_params['activations']

        def create_weights(shape, name):
            return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)

        return [{'weights': create_weights([sizes[i], sizes[i+1]],
                                            'w' + str(i)),
                 'biases': create_weights([sizes[i+1]], 'b' + str(i)),
                 'activation': activations[i]}
                for i in range(len(sizes) - 1)]

    def __mlp(self, X, network_params):
        keep_prob = network_params['keep_prob']

        def compose_func(a, x, w, b):
            return a(tf.matmul(x, w) + b)

        prev_value = tf.expand_dims(X, 0)
        for i, entry in enumerate(self.model):
            prev_value = compose_func(entry['activation'],
                                      prev_value,
                                      entry['weights'],
                                      entry['biases'])

            if i != len(self.model) - 1:
                prev_value = tf.nn.dropout(prev_value, keep_prob)

        return prev_value

    def __gen_l2_loss(self):
        weights = [entry['weights'] for entry in self.model]
        weights += [entry['biases'] for entry in self.model]

        return reduce(lambda a, b: a + tf.nn.l2_loss(b), weights, 0)

    def create_prediction(self):
        return self.prediction

    def create_cost(self):
        return self.cost



networks_dict = {'LSTM_RNN': LSTM_RNN,
                 'MLP': MLP}

def create_network(type, X, Y, network_params):
    return networks_dict[type](X, Y, network_params)
