import numpy as np
import pandas as pd

def flatten(np_array):
  '''
  single level flattening
  '''
  return np_array.reshape(np_array.shape[0],np_array.shape[1]*np_array.shape[2])