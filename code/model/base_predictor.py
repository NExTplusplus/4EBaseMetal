from copy import copy

class BasePredictor:
    def __init__(self, parameters):
        self.pars = copy(parameters)

    def train(self, X_tr, y_tr, X_val, y_val, tune_para=False):
        pass

    def predict(self, X):
        pass