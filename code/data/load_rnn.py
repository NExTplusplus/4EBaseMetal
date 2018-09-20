import numpy as np
import pandas as pd


'''
parameters:
fname_columns (dict): the files going to be read, with the selected columns, a 
    sample, {'fname1': [col1, col2, ...], 'fname2': [...]}.
ground_truth (str): the way to construct ground truth.
norm_method (str): the way to normalize data.
split_dates ([datetime]): start date for training, validation, and testing.
T (int): the lag size/sequence length.
S (int): the number of days towards the expected day of the selected time 
    horizon, e.g., S = 3 means the time horizon of 3-day, default value is 1.

returns:
X_tr (3d numpy array): N_t * T * D, N_t is the number of training samples, D is
    the dimension of temporal features.
y_tr (2d numpy array): 
X_val (3d numpy array):
y_val (2d numpy array):
X_tes (3d numpy array):
y_tes (2d numpy array):
'''
def load_pure_lstm(fname_columns, ground_truth, norm_method, split_dates, T, S=1):
    # read data from files
    time_series = None
    for fname in fname_columns:
        print('read columns:', fname_columns[fname], 'from:', fname)
        if time_series is None:
            time_series = read_single_csv(fname, fname_columns[fname])
        else:
            time_series = merge_data_frame(
                time_series, read_single_csv(fname, fname_columns[fname])
            )
    print('data shape:')

    # pre-process for missing values, e.g., NA
    time_series = process_missing_value(time_series)

    # normalize data
    norm_time_series = norm_method(time_series)

    # construct the training, validation, and testing
    X_tr = np.zeros([N, T, D], dtype=float32)
    pass
    return