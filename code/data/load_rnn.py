from copy import copy
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

from utils.read_data import read_single_csv, merge_data_frame, \
    process_missing_value
from utils.normalize_feature import log_1d_return
from utils.transform_data import flatten
from utils.construct_data import construct

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
    print('data shape:', time_series.shape)

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
    ground_truth = copy(time_series[gt_column])
    for ind in range(time_series.shape[0] - S):
        if ground_truth.iloc[ind + S] - ground_truth.iloc[ind] > 0:
            ground_truth.iloc[ind] = 1
        else:
            ground_truth.iloc[ind] = 0

    # normalize data
    if norm_method == 'log_1d_return' or norm_method == 'log_nd_return':
        norm_data = copy(log_1d_return(time_series))
    else:
        norm_data = copy(time_series)

    tra_ind = norm_data.index.get_loc(split_dates[0])
    if tra_ind < T - 1:
        tra_ind = T - 1
    val_ind = norm_data.index.get_loc(split_dates[1])
    assert val_ind >= T - 1, 'without training data'
    tes_ind = norm_data.index.get_loc(split_dates[2])

    X_tr = None
    X_va = None
    X_te = None
    y_tr = None
    y_va = None
    y_te = None

    if norm_method == 'log_1d_return':
        # construct the training
        X_tr,y_tr = construct(norm_data, ground_truth, tra_ind, val_ind, T)

        # construct the validation
        X_va,y_va = construct(norm_data, ground_truth, val_ind, tes_ind, T)
        
        # construct the testing
        X_te,y_te = construct(norm_data, ground_truth, tes_ind, norm_data.shape[0]-S-1, T)

    elif norm_method == 'log_nd_return':
        for i in range(1,T+1):
            
        

    return X_tr, y_tr, X_va, y_va, X_te, y_te

def load_pure_log_reg(fname_columns, gt_column, norm_method, split_dates, T, S=1):
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_pure_lstm(fname_columns, gt_column, norm_method, split_dates, T, S=1)
    X_tr = flatten(X_tr)
    X_va = flatten(X_va)
    X_te = flatten(X_te)

    return X_tr, y_tr, X_va, y_va, X_te, y_te
