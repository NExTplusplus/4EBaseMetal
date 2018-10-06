import numpy as np
import pandas as import pd

def flatten(np_array):
  return np_array.reshape(np_array.shape[0],np_array.shape[1]*np_array.shape[2])