import numpy as np

from base_predictor import BasePredictor

class LSTM(BasePredictor):
    def __init__(self, parameters):
        BasePredictor.__init__(parameters)