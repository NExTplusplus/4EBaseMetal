import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

from utils.read_data import read_single_csv, merge_data_frame, \
    process_missing_value
from utils.normalize_feature import log_1d_return

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
def load_pure_lstm(fname_columns, gt_column, norm_method, split_dates, T, S=1):
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

    # # pre-process for missing values, e.g., NA
    # time_series, sta_ind = process_missing_value(time_series)
    # data = time_series.values
    #
    # # normalize data
    # if norm_method == 'log_1d_return':
    #     norm_data = log_1d_return(data)
    #     sta_ind += 1
    # else:
    #     norm_data = data
    #
    # tra_ind = time_series.index.get_loc(split_dates[0])
    # if tra_ind - T < sta_ind:
    #     tra_ind = sta_ind + T
    # val_ind = time_series.index.get_loc(split_dates[1])
    # assert val_ind >= sta_ind + T, 'without training data'
    # tes_ind = time_series.index.get_loc(split_dates[2])
    #
    # # construct the training, validation, and testing
    # X_tr = np.zeros([val_ind - tra_ind, T, norm_data.shape[1]], dtype=np.float32)
    # y_tr = np.zeros([val_ind - tra_ind, 1], dtype=np.float32)
    # for ind in range(tra_ind, val_ind):
    #     X_tr[ind - tra_ind] = norm_data[]
    # pass
    # return

    # construct ground truth
    ground_truth = time_series[gt_column]
    for ind in range(time_series.shape[1] - S):
        if ground_truth.iloc[ind + S] - ground_truth.iloc[ind] > 0:
            ground_truth.iloc[ind] = 1
        else:
            ground_truth.iloc[ind] = 0

    # normalize data
    if norm_method == 'log_1d_return':
        norm_data = log_1d_return(time_series)
    else:
        norm_data = time_series

    tra_ind = time_series.index.get_loc(split_dates[0])
    if tra_ind < T - 1:
        tra_ind = T - 1
    val_ind = time_series.index.get_loc(split_dates[1])
    assert val_ind >= T - 1, 'without training data'
    tes_ind = time_series.index.get_loc(split_dates[2])

    # construct the training
    tra_num = 0
    for ind in range(tra_ind, val_ind):
        if not norm_data.iloc[ind - T + 1: ind].isnull().values.any():
            tra_num += 1
    X_tr = np.zeros([tra_num, T, norm_data.shape[1]], dtype=np.float32)
    y_tr = np.zeros([tra_num, 1], dtype=np.float32)

    for ind in range(tra_ind, val_ind):
        X_tr[ind - tra_ind] = norm_data[]
    pass
    return