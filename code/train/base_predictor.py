from copy import copy

class BasePredictor:
    def __init__(self, parameters):
        self.parameters = copy(parameters)

    def train(self, X, y):
        pass

    def predict(self, X):
        pass