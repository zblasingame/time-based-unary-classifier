""" Abstract Class for Neural Network Model
    Author: Zander Blasingame """

from abc import ABC, abstractmethod

class NeuralNet(ABC):
    @abstractmethod
    def create_prediction(self):
        pass

    @abstractmethod
    def create_cost(self):
        pass
