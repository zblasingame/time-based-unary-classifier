""" A class to create a Multilayer Perceptron.
    Author: Zander Blasingame """

import tensorflow as tf
from functools import reduce

class MLP:
    def __init__(self, sizes, activations):
        def create_weights(shape, name):
            return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)

        self.network = [{'weights': create_weights([sizes[i], sizes[i+1]],
                                                   'w' + str(i)),
                         'biases': create_weights([sizes[i+1]], 'b' + str(i)),
                         'activation': activations[i]}
                        for i in range(len(sizes) - 1)]


    # Method that creates network
    def create_network(self, X, keep_prob):
        def compose_func(a, x, w, b):
            return a(tf.matmul(x, w) + b) 

        prev_value = tf.expand_dims(X, 0)
        for i, entry in enumerate(self.network):
            prev_value = compose_func(entry['activation'],
                                      prev_value,
                                      entry['weights'],
                                      entry['biases'])

            if i != len(self.network) - 1:
                prev_value = tf.nn.dropout(prev_value, keep_prob)

        return prev_value


    # Returns L2 loss
    def get_l2_loss(self):
        weights = [entry['weights'] for entry in self.network]
        weights += [entry['biases'] for entry in self.network]

        return reduce(lambda a, b: a + tf.nn.l2_loss(b), weights, 0) 
