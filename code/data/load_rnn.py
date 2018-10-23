from copy import copy
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

from utils.read_data import read_single_csv, merge_data_frame, \
    process_missing_value_v3
from utils.normalize_feature import log_1d_return, normalize_volume, normalize_3mspot_spread, \
    normalize_OI, normalize_3mspot_spread_ex
from utils.transform_data import flatten
from utils.construct_data import construct

'''
parameters:
fname_columns (dict): the files going to be read, with the selected columns, a 
    sample, {'fname1': [col1, col2, ...], 'fname2': [...]}.
ground_truth (str): the way to construct ground truth.
norm_method ([str]): the way to normalize data for each column.
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
def load_pure_lstm(fname_columns, gt_column, norm_method, split_dates, T, S=1, OI_name = None, len_ma = None, len_update = None, lme_col = None, shfe_col = None,\
                    exchange = None, Volume_norm_v ="v1", spread_ex_v = "v1", spread_v = "v1"):
    # read data from files
    time_series = None


    for fname in fname_columns:
        # print('read columns:', fname_columns[fname], 'from:', fname)
        if time_series is None:
            time_series = read_single_csv(fname, fname_columns[fname])
        else:
            time_series = merge_data_frame(
                time_series, read_single_csv(fname, fname_columns[fname])
            )
    
    process_missing_value_v3(time_series,10)
    org_columns = time_series.columns.values.tolist()    

    if OI_name is not None:
        # print(1)
        temp = normalize_OI(time_series,OI_name, OI_norm_v)
        if temp is not None:
            # print(2)
            time_series = temp
        temp = normalize_volume(time_series,OI_name,len_ma, Volume_norm_v)
        if temp is not None:
            # print(3)
            time_series = temp
        
    if lme_col is not None and shfe_col is not None and exchange is not None:
        # print(4)
        temp = normalize_3mspot_spread_ex (time_series,lme_col,shfe_col,exchange,len_update, spread_ex_v)
        if temp != None:
            # print(5)
            time_series = temp
 
    if "Close.Price" in org_columns:
        # print(6)
        temp = normalize_3mspot_spread(time_series,gt_column,len_update, spread_v)
        if temp is not None:
            # print(7)
            time_series = temp
    
    

        
    ground_truth = copy(time_series[gt_column])
    # print(time_series.columns.values.tolist())
    time_series = time_series.drop(["Volume",OI_name,shfe_col,exchange],axis = 1, errors = "ignore")
    
    for i in ["Volume",OI_name,shfe_col,exchange]:
        if i in org_columns:
            org_columns.remove(i)

    # print(org_columns)
    # print(time_series.columns.values.tolist())

    for ind in range(time_series.shape[0] - S):
        #print(S)
        if ground_truth.iloc[ind + S] - ground_truth.iloc[ind] > 0:
            ground_truth.iloc[ind] = 1
        else:
            ground_truth.iloc[ind] = 0

    # normalize data
    if norm_method == 'log_1d_return' or norm_method == 'log_nd_return':
        norm_data = copy(log_1d_return(time_series,org_columns))
    else:
        norm_data = copy(time_series)

    tra_ind = norm_data.index.get_loc(split_dates[0])
    if tra_ind < T - 1:
        tra_ind = T - 1
    val_ind = norm_data.index.get_loc(split_dates[1])
    assert val_ind >= T - 1, 'without training data'
    tes_ind = norm_data.index.get_loc(split_dates[2])

    # construct the training
    X_tr,y_tr = construct(norm_data, ground_truth, tra_ind, val_ind, T, norm_method)

    # construct the validation
    X_va,y_va = construct(norm_data, ground_truth, val_ind, tes_ind, T, norm_method)

    # construct the testing
    X_te,y_te = construct(norm_data, ground_truth, tes_ind, norm_data.shape[0]-S-1, T, norm_method)

        

            
    return X_tr, y_tr, X_va, y_va, X_te, y_te

def load_pure_log_reg(fname_columns, gt_column, norm_method, split_dates, T, S=1, OI_name = None, len_ma = None, len_update = None, lme_col = None, shfe_col = None,\
                    exchange = None, Volume_norm_v ="v1", spread_ex_v = "v1", spread_v = "v1"):
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_pure_lstm(fname_columns, gt_column, norm_method, split_dates, T, S)
    neg_y_tr = y_tr - 1
    neg_y_va = y_va - 1
    neg_y_te = y_te - 1
    y_tr = y_tr + neg_y_tr
    y_va = y_va + neg_y_va
    y_te = y_te + neg_y_te
    X_tr = flatten(X_tr)
    X_va = flatten(X_va)
    X_te = flatten(X_te)

    return X_tr, y_tr, X_va, y_va, X_te, y_te
