"""A class to create a basic neural net in Python

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import tensorflow as tf
from functools import reduce


class NeuralNet:
    """A basic neural net implementation

    Args:
        sizes (list of int): List describing the size of each layer
        activations (list of function): List of TensorFlow activation functions
            must be one less element than the number of elements in
            the parameter sizes

    Attributes:
        network (list of dict): List of dictionaries outlining the weights,
            biases, and activation functions at each layer
    """

    def __init__(self, sizes, activations):
        """Initializes NeuralNet class"""
        assert len(sizes) == len(activations) + 1, (
            'sizes and activations have a missmatched number of elements'
        )

        def create_weights(shape, name):
            return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)

        self.network = [{'weights': create_weights([sizes[i], sizes[i+1]],
                                                   'w' + str(i)),
                         'biases': create_weights([sizes[i+1]], 'b' + str(i)),
                         'activation': activations[i]}
                        for i in range(len(sizes) - 1)]

    def create_network(self, X, keep_prob):
        """Method to construct the network

        Args:
            X (tf.Tensor): Placeholder Tensor with dimenions of the
                training Tensor
            keep_prob (tf.Tensor): Placeholder Tensor of rank one
                of the probability for the dropout technique

        Returns:
            (tf.Tensor): A tensor to be evaulated containing the predicted
                output of the neural net
        """

        def compose_func(a, x, w, b):
            return a(tf.matmul(x, w) + b)

        prev_value = X
        for i, entry in enumerate(self.network):
            prev_value = compose_func(entry['activation'],
                                      prev_value,
                                      entry['weights'],
                                      entry['biases'])

            if i != len(self.network) - 1:
                prev_value = tf.nn.dropout(prev_value, keep_prob)

        return prev_value

    def reset_weights(self):
        """Resets TensorFlow weights so the model can be used again

        Returns:
            (list of tf.Operation) List of operations to reassign weights,
                run using Session.run()
        """

        weights = [entry['weights'] for entry in self.network]
        weights.extend([entry['biases'] for entry in self.network])

        return [weight.assign(tf.random_normal(weight.get_shape(), stddev=0.1))
                for weight in weights]

    def get_l2_loss(self):
        """Method to return the L2 loss for L2 regularization techniques

        Returns:
            (tf.Tensor): A tensor to be evaulated containing the
                L2 loss of the network
        """

        weights = [entry['weights'] for entry in self.network]
        weights.extend([entry['biases'] for entry in self.network])

        return reduce(lambda a, b: a + tf.nn.l2_loss(b), weights, 0)
