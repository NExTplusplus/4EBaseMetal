import numpy as np
import pandas as pd

def construct(time_series, ground_truth, start_ind, end_ind, T):
    num = 0
    for ind in range(start_ind + 1, end_ind + 1):
        if not time_series.iloc[ind - T: ind].isnull().values.any():
            num += 1
    X = np.zeros([num, T, time_series.shape[1]], dtype=np.float32)
    y = np.zeros([num, 1], dtype=np.float32)

    sample_ind = 0
    for ind in range(start_ind + 1, end_ind + 1):
        if not time_series.iloc[ind - T: ind].isnull().values.any():
            X[sample_ind] = time_series.values[ind - T: ind, :]
            y[sample_ind, 0] = ground_truth.values[ind - 1]
            sample_ind += 1

    return X,y