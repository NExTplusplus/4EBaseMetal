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
from utils.construct_data import construct,normalize,technical_indication

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
def load_pure_lstm(fname_columns, gt_column, norm_method, split_dates, T, S=1, 
                    vol_norm ="v1", ex_spread_norm = "v1", spot_spread_norm = "v1",inc = True):
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
    time_series = process_missing_value_v3(time_series,10)
    org_cols = time_series.columns.values.tolist()
    print("Normalizing")
    norm_params = normalize(time_series,vol_norm = vol_norm,spot_spread_norm=spot_spread_norm,ex_spread_norm=ex_spread_norm)
    time_series = copy(norm_params["val"])
    del norm_params["val"]
    time_series = technical_indication(time_series)
    cols = time_series.columns.values.tolist()
    for col in cols:
        if "_Volume" in col or "_OI" in col or "CNYUSD" in col:
            time_series.drop(col,errors = "ignore")
            org_cols.remove(col)
 
    ground_truth = copy(time_series[gt_column])

    for ind in range(time_series.shape[0] - S):
        #print(S)
        if ground_truth.iloc[ind + S] - ground_truth.iloc[ind] > 0:
            ground_truth.iloc[ind] = 1
        else:
            ground_truth.iloc[ind] = 0
            
    norm_data = copy(log_1d_return(time_series,org_cols))
    
    norm_data = process_missing_value_v3(norm_data,10)

    # normalize data
    # if norm_method == 'log_1d_return' or norm_method == 'log_nd_return':
    #     norm_data = copy(log_1d_return(time_series,org_columns))
    # else:
    #     norm_data = copy(time_series)
    tra_ind = 0
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

        

            
    return X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params

def load_pure_log_reg(fname_columns, gt_column, norm_method, split_dates, T, S=1, OI_name = None, len_ma = None, 
                        len_update = None, lme_col = None, shfe_col = None, exchange = None, vol_norm ="v1", 
                        ex_spread_norm = "v1", spot_spread_norm = "v1", inc = True
                        ):
    X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params = load_pure_lstm(fname_columns, gt_column, norm_method, split_dates, T, S = S,
                                                        vol_norm = vol_norm, ex_spread_norm = ex_spread_norm,
                                                        spot_spread_norm = spot_spread_norm, inc = True
                                                        )
    neg_y_tr = y_tr - 1
    neg_y_va = y_va - 1
    neg_y_te = y_te - 1
    y_tr = y_tr + neg_y_tr
    y_va = y_va + neg_y_va
    y_te = y_te + neg_y_te
    X_tr = flatten(X_tr)
    X_va = flatten(X_va)
    X_te = flatten(X_te)

    return X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params
