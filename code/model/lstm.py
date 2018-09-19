import numpy as np
import os
import random
from sklearn.utils import shuffle
import tensorflow as tf
from time import time

from base_predictor import BasePredictor

class LSTM(BasePredictor):
    def __init__(self, parameters, epochs=50, batch_size=1024, ):
        BasePredictor.__init__(parameters)