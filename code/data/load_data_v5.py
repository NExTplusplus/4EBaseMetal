from copy import copy
import numpy as np
import os
import sys
import pandas as pd
import json

from utils.read_data import  process_missing_value_v3
from utils.normalize_feature import log_1d_return, normalize_volume, normalize_3mspot_spread, normalize_OI, normalize_3mspot_spread_ex
from utils.transform_data import flatten
from utils.construct_data import construct,normalize_without_1d_return,technical_indication,construct_keras_data,scaling,labelling,deal_with_abnormal_value

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
def load_data_v5(config, horizon, ground_truth_columns, lags, source, split_dates, norm_params, tech_params):
    """
    input: config: A file to define which file we load and which column we use.
           split_dates: define the time that we use to define the range of the data.
    output:data_list: A list contains the data from the file or Database.
           LME_dates: A list contains the LME's date.
    """
    if source =="NExT":
        from utils.read_data import read_data_NExT
        data_list, LME_dates = read_data_NExT(config, split_dates[0])
        time_series = pd.concat(data_list, axis = 1, sort = True)
    elif source == "4E":
        from utils.read_data import read_data_v5_4E
        time_series, LME_dates = read_data_v5_4E(split_dates[0])

    '''
    deal with the abnormal data which we found in the data. 
    '''
    time_series = deal_with_abnormal_value(time_series)
    '''
    Extract the rows with dates where LME has trading operations
    and generate labels
    '''
    time_series = time_series.loc[LME_dates]
    labels = labelling(time_series, horizon, ground_truth_columns)

    '''
    Normalize, create technical indicators, handle outliers and rescale data
    '''
    org_cols = time_series.columns.values.tolist()
    time_series, norm_params = normalize_without_1d_return(time_series, split_dates[1], params = norm_params)
    time_series = technical_indication(time_series, split_dates[1], params = tech_params)
    for col in copy(time_series.columns):
        if "_Volume" in col or "_OI" in col or "CNYUSD" in col or "_PVT" in col:
            time_series = time_series.drop(col,axis = 1)
            if col in org_cols:
                org_cols.remove(col)
    time_series = log_1d_return(time_series,org_cols)
    time_series = process_missing_value_v3(time_series,1)
    time_series = scaling(time_series,time_series.index.get_loc(split_dates[1]))
    complete_time_series = []
    time_series = time_series[sorted(time_series.columns)]
    
    all_cols = []
    if len(ground_truth_columns) > 1:
        for ground_truth in ground_truth_columns:
            temp = copy(time_series)
            temp['self'] = copy(temp[ground_truth])
            temp.insert(0,'self',temp.pop('self'),allow_duplicates = True)
            complete_time_series.append(temp)
            all_cols.append(temp.columns)
        time_series = complete_time_series
    else:
        time_series.insert(0,ground_truth_columns[0],time_series.pop(ground_truth_columns[0]),allow_duplicates = True)
        time_series = [time_series]
        all_cols.append(time_series[0].columns)
    
    

    '''
    Merge labels with time series dataframe
    '''
    for ind in range(len(time_series)):
        time_series[ind] = pd.concat([time_series[ind], labels[ind]], axis = 1)
        time_series[ind] = process_missing_value_v3(time_series[ind],1)
    #   save_data("i6",time_series[0],time_series[0].columns.values.tolist())

    '''
    create 3d array with dimensions (n_samples, lags, n_features)
    '''

    tra_ind = 0
    if tra_ind < lags - 1:
        tra_ind = lags - 1
    # closest_val = time_series[0].loc[time_series[0].index >= split_dates[1]].index[0]
    val_ind = time_series[0].index.get_loc(split_dates[1])
    assert val_ind >= lags - 1, 'without training data'
    # closest_tes = time_series[0].loc[time_series[0].index <= split_dates[2]].index[-1]
    tes_ind = time_series[0].index.get_loc(split_dates[2])

    X_tr = []
    y_tr = []
    X_va = []
    y_va = []
    X_te = []
    y_te = []
    # print(len(time_series))
    for ind in range(len(time_series)):
        # construct the training
        temp = construct(time_series[ind][all_cols[ind]], time_series[ind]["Label"], tra_ind, val_ind, lags, horizon, 'log_1d_return')
        X_tr.append(temp[0])
        y_tr.append(temp[1])

        # construct the validation
        temp = construct(time_series[ind][all_cols[ind]], time_series[ind]["Label"], val_ind, tes_ind, lags, horizon, 'log_1d_return')
        X_va.append(temp[0])
        y_va.append(temp[1])

        # construct the testing
        if tes_ind < time_series[ind].shape[0]-horizon-1:
            temp = construct(time_series[ind][all_cols[ind]], time_series[ind]["Label"], tes_ind, time_series[ind].shape[0]-horizon-1, lags, horizon, 'log_1d_return')
            X_te.append(temp[0])
            y_te.append(temp[1])
        else:
            X_te = None
            y_te = None
    
    
    
    return X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params


