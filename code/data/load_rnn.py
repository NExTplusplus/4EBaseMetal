from copy import copy
import numpy as np
import os
import sys
import pandas as pd
from utils.read_data import read_single_csv, merge_data_frame, \
    process_missing_value_v3
from utils.normalize_feature import log_1d_return, normalize_volume, normalize_3mspot_spread, \
    normalize_OI, normalize_3mspot_spread_ex
from utils.transform_data import flatten
from utils.construct_data import construct,normalize,technical_indication,construct_keras_data, rescale

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
using_frame (str): a flag to distinguish data process for keras model
returns:
X_tr (3d numpy array): N_t * T * D, N_t is the number of training samples, D is
    the dimension of temporal features.
y_tr (2d numpy array): 
X_val (3d numpy array):
y_val (2d numpy array):
X_tes (3d numpy array):
y_tes (2d numpy array):
'''
def save_data(fname,time_series,columns, ground_truth = None):
    col_name = ""
    for col in columns:
        col_name = col_name + " " + col
    with open("../../"+fname+".csv","w") as out:
        out.write(col_name.replace(" ",","))
        out.write(",\n")
        for i in range(len(time_series)):
            row = time_series.iloc[i]
            out.write(time_series.index[i]+",")
            for v in row:
                out.write(str(v)+",")
            if ground_truth is not None:
                out.write(str(ground_truth[i]))
            out.write("\n")

def load_pure_lstm(fname_columns, norm_method, split_dates, T, gt_column = None, S = 1,
    vol_norm ="v1", ex_spread_norm = "v1", spot_spread_norm = "v1",
    len_ma = 5, len_update = 30, version = 1, norm_strength = 0.01, norm_both = 0, tech_strength = 0.01, tech_both = 0
    ):
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

    columns = time_series.columns

    # save_data("i1",time_series,columns)
    time_series = process_missing_value_v3(time_series,10)
    # save_data("i2",time_series,columns)
    train_end = time_series.index.get_loc(split_dates[1])
    org_cols = time_series.columns.values.tolist()
    print("Normalizing")
    norm_params = normalize(time_series,vol_norm = vol_norm, vol_len = len_ma,
                            spot_spread_norm = spot_spread_norm, ex_spread_norm = ex_spread_norm,
                            spot_spread_len= len_update, ex_spread_len = len_update, train_end = train_end,
                            strength = norm_strength, both = norm_both
                            )
    time_series = copy(norm_params["val"])

    # save_data("i3",time_series,time_series.columns.values.tolist())
    
    del norm_params["val"]
    time_series = technical_indication(time_series, train_end, strength = tech_strength, both = tech_both)
    
    # save_data("i4",time_series,time_series.columns.values.tolist())
    
    for col in copy(time_series.columns):
        if "_Volume" in col or "_OI" in col or "CNYUSD" in col or "_PVT" in col:
            time_series = time_series.drop(col,axis = 1)
            if col in org_cols:
                org_cols.remove(col)
    
    # save_data("i5_"+str(vol_norm),time_series,time_series.columns.values.tolist())
    
    # if using_frame == "keras":
    #     for col in cols:
    #         if "Spot" in col:
    #             ground_truth_index = cols.index(col)
    #     X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_daybefore_val, Y_daybefore_tes, unnormalized_bases_val, unnormalized_bases_tes, window_size = construct_keras_data(time_series, ground_truth_index, T+1) 
    #     return X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_daybefore_val, Y_daybefore_tes, unnormalized_bases_val, unnormalized_bases_tes, window_size
            
    norm_data = copy(log_1d_return(time_series,org_cols))
    norm_data = process_missing_value_v3(norm_data,10)
    norm_data = rescale(norm_data)
    cols = norm_data.columns.values.tolist()
    if gt_column is None:
        all_metals = []
        gt_column = 'self'
        for col in cols:
            if "_Spot" in col:
                temp = copy(norm_data)
                temp['self'] = copy(temp[col])
                temp.insert(0,'self',temp.pop('self'),allow_duplicates = True)
                all_metals.append(temp)
        norm_data = all_metals
    else:
        norm_data.insert(0,gt_column,norm_data.pop(gt_column),allow_duplicates = True)
        norm_data = [norm_data]
    
    if version == 1 or version == 2:
        for i in range(len(norm_data)):
            norm_data[i] = pd.DataFrame(norm_data[i][gt_column])
    # print(norm_data[0].columns)
    
    ground_truth = []
    for data_set in norm_data:
        to_be_predicted = copy(data_set[gt_column])
        if S > 1:
            for i in range(S-1):
                to_be_predicted = to_be_predicted + data_set[gt_column].shift(-i-1)
        ground_truth.append((to_be_predicted > 0).shift(-1))

    save_data("i6_"+str(vol_norm),pd.concat(norm_data),norm_data[0].columns.values.tolist(),np.concatenate(ground_truth))

    tra_ind = 0
    if tra_ind < T - 1:
        tra_ind = T - 1
    val_ind = norm_data[0].index.get_loc(split_dates[1])
    assert val_ind >= T - 1, 'without training data'
    tes_ind = norm_data[0].index.get_loc(split_dates[2])

    X_tr = []
    y_tr = []
    X_va = []
    y_va = []
    X_te = []
    y_te = []
    # print(len(norm_data))
    for ind in range(len(norm_data)):
        # construct the training
        temp = construct(norm_data[ind], ground_truth[ind], tra_ind, val_ind, T, S, norm_method)
        X_tr.append(temp[0])
        y_tr.append(temp[1])

        # construct the validation
        temp = construct(norm_data[ind], ground_truth[ind], val_ind, tes_ind, T, S, norm_method)
        X_va.append(temp[0])
        y_va.append(temp[1])

        # construct the testing
        if tes_ind < norm_data[ind].shape[0]-S-1:
            temp = construct(norm_data[ind], ground_truth[ind], tes_ind, norm_data[ind].shape[0]-S-1, T, S, norm_method)
            X_te.append(temp[0])
            y_te.append(temp[1])
        else:
            X_te = None
            y_te = None
    
    
    
    return X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params

def load_pure_log_reg(fname_columns, norm_method, split_dates, T, gt_column = None, S=1, 
                        vol_norm ="v1", ex_spread_norm = "v1", spot_spread_norm = "v1",
                        len_ma = 5, len_update = 30, version = 1,
                        norm_strength = 0.01, norm_both = 0, tech_strength = 0.01, tech_both = 0
                        ):
    X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params = load_pure_lstm(fname_columns, norm_method, split_dates, T, gt_column = gt_column, S = S,
                                                        vol_norm = vol_norm, ex_spread_norm = ex_spread_norm, spot_spread_norm = spot_spread_norm,
                                                        len_ma = len_ma, len_update = len_update, version = version,
                                                        norm_strength = norm_strength, norm_both = norm_both, tech_strength = tech_strength, tech_both = tech_both
                                                        )
    for ind in range(len(X_tr)):
        neg_y_tr = y_tr[ind] - 1
        y_tr[ind] = y_tr[ind] + neg_y_tr
        X_tr[ind] = flatten(X_tr[ind])
        
        
        neg_y_va = y_va[ind] - 1
        y_va[ind] = y_va[ind] + neg_y_va
        X_va[ind] = flatten(X_va[ind])
        
        if X_te is not None:
            neg_y_te = y_te[ind] - 1
            y_te[ind] = y_te[ind] + neg_y_te
            X_te[ind] = flatten(X_te[ind])
    
    # print(y_te[:-1])

    return X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params
