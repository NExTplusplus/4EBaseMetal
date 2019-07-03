from copy import copy
import numpy as np
import os
import sys
import pandas as pd
import json
sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],"..")))
from utils.read_data import read_data, process_missing_value_v3
from utils.normalize_feature import log_1d_return, normalize_volume, normalize_3mspot_spread, normalize_OI, normalize_3mspot_spread_ex
from utils.transform_data import flatten
from utils.construct_data import construct,normalize,technical_indication,construct_keras_data,rescale,labelling,deal_with_outlier


def load_data_v5(config, horizon, ground_truth_columns, lags, source, split_dates, norm_params, tech_params):
    
    '''
    Load data into individual dataframes and merge them into a single dataframe
    
    the read_data function has two inputs and two outputs:
    Input
    config(dict) 	: A dictionary with keys of filepaths relative to 4EBaseMetal folder
                      and values of a list of columns to be read from a stated path
    source(str)		: An identifier of the source of data, takes in only two values ["4E","NExT"]
                      Based on the source, the function will read data differently
    Output
    data_list(list)	: A list of datasets, each representing a file and its respective columns (if source is NExT)
    LME_dates(list)	: A list of dates(string) where LME has trading operations
    '''
    data_list, LME_dates = read_data(config, source,split_dates[0])
    time_series = pd.concat(data_list, axis = 1, sort = True)

    
    '''
    Handle data errors in the data
    '''

    '''
    Handle NA values that belong to Class 3 (missing data)
    '''
    time_series = deal_with_outlier(time_series)
    # time_series = process_missing_value_v3(time_series,0)
    '''
    Extract the rows with dates where LME has trading operations
    and generate labels
    '''
    time_series = time_series.loc[LME_dates]
    ground_truth = labelling(time_series,ground_truth_columns)

    '''
    Normalize, create technical indicators, handle outliers and rescale data
    '''
    org_cols = time_series.columns.values.tolist()
    time_series, norm_params = normalize(time_series, split_dates[1], params = norm_params)
    time_series = technical_indication(time_series, split_dates[1], params = tech_params)
    for col in copy(time_series.columns):
        if "_Volume" in col or "_OI" in col or "CNYUSD" in col or "_PVT" in col:
            time_series = time_series.drop(col,axis = 1)
            if col in org_cols:
                org_cols.remove(col)
    time_series = log_1d_return(time_series,org_cols)
    time_series = rescale(time_series)
    complete_time_series = []
    for ground_truth in ground_truth_columns:
        temp = copy(time_series)
        temp['self'] = copy(temp[ground_truth])
        temp.insert(0,'self',temp.pop('self'),allow_duplicates = True)
        complete_time_series.append(temp)
        time_series = complete_time_series
    else:
        time_series.insert(0,gt_column,time_series.pop(gt_column),allow_duplicates = True)
        time_series = [time_series]

    all_cols = time_series.columns

    '''
    Merge labels with time series dataframe
    '''
    for ind in range(length(time_series)):
        time_series[ind] = pd.concat([time_series[ind], labels[ind]], axis = 1)
        time_series[ind] = process_missing_value_v3(time_series,0)[ind]
    

    '''
    create 3d array with dimensions (n_samples, lags, n_features)
    '''

    tra_ind = 0
    if tra_ind < T - 1:
        tra_ind = T - 1
    val_ind = time_series[0].index.get_loc(split_dates[1])
    assert val_ind >= T - 1, 'without training data'
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
        temp = construct(time_series[ind][all_cols], time_series[ind]["Label"], tra_ind, val_ind, T, S, norm_method)
        X_tr.append(temp[0])
        y_tr.append(temp[1])

        # construct the validation
        temp = construct(time_series[ind][all_cols], time_series[ind]["Label"], val_ind, tes_ind, T, S, norm_method)
        X_va.append(temp[0])
        y_va.append(temp[1])

        # construct the testing
        if tes_ind < time_series[ind].shape[0]-S-1:
            temp = construct(time_series[ind][all_cols], time_series[ind]["Label"], tes_ind, time_series[ind].shape[0]-S-1, T, S, norm_method)
            X_te.append(temp[0])
            y_te.append(temp[1])
        else:
            X_te = None
            y_te = None
    
    
    
    return X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params

with open(os.path.join(sys.path[0],"exp","3d","Co","logistic_regression","v5","LMCADY_v5.conf")) as fin:
    fname_columns = json.load(fin)

tra_date = '2004-11-12'
val_date = '2016-06-01'
tes_date = '2016-12-16'
split_dates = [tra_date, val_date, tes_date]
norm_params = {'vol_norm': "v2", "ex_spread_norm":"v2", "spot_spread_norm":"v1","len_ma":5, "len_update": 30, "both" : 3, "strength" : 0.01}
tech_params = {'strength':0.01,'both':3}

for f in fname_columns:
    X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params= load_data_v5(
                                    f, 3, ["LME_Co_Spot"],5,"NExT", split_dates, norm_params, tech_params
                                )
