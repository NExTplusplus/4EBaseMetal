from copy import copy
import numpy as np
import os
import sys
import pandas as pd
import json

from utils.read_data import  process_missing_value_v3
from utils.normalize_feature import log_1d_return, normalize_volume, normalize_3mspot_spread, normalize_OI, normalize_3mspot_spread_ex
from utils.transform_data import flatten
from utils.construct_data import construct,normalize_without_1d_return,technical_indication_v5,construct_keras_data,scaling,labelling,deal_with_abnormal_value

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
def load_data_v5(time_series, horizon, ground_truth_columns, lags, LME_dates, split_dates, norm_params, tech_params):
    """
    input: time_series: A dataframe that holds the data.
           split_dates: define the time that we use to define the range of the data.
           horizon: (int) The time horizon.
           ground_truth_columns: (str)The column name that we want to predict.
           LME_dates: (int)list of dates of which LME has trading operations.
           source: (str)An identifier of the source of data, takes in only two values ["4E", "NExT"]. Based on the source, the function will read data differently
           norm_params: (dictionary)contains the param we need to normalize OI, Volume ,and Spread.
                        'vol_norm': Version of volume normalization
                        'len_ma': length of period to compute moving average
                        'len_update': length of period to update moving average
                        'spot_spread_norm': Version of 3 months to spot spread normalization
                        'strength': Strength of thresholding for OI and Volume
                        'both': Sides of thresholding (0 for no thresholding, 1 for left, 2 for right, 3 for both sides) for OI and Volume
                        'ex_spread_norm': Version of the cross exchange spread normalization
           tech_params: (dictionary)contains the param we need to create technical indicators.
                        'strength': Strength of thresholding for divPVT 
                        'both': Sides of thresholding (0 for no thresholding, 1 for left, 2 for right, 3 for both sides)
    output:
           data: (array)An array contains the data that we use to feed into the model
           norm_check: (dictionary)we use this to check whether any column specific normalization is triggered. It is a dictionary with 3 key-value pairs
                        nVol (boolean) : check True if volume is normalized
                        spread(boolean): check True if Spread is produced
                        nEx(boolean): check True if Cross Exchange Spread is produced
           
    """


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
    time_series, norm_params = normalize_without_1d_return(time_series, time_series.index.get_loc(split_dates[1]), params = norm_params)
    time_series = technical_indication_v5(time_series, time_series.index.get_loc(split_dates[1]), params = tech_params)
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
        print("mistake")
        # for ground_truth in ground_truth_columns:
        #     temp = copy(time_series)
        #     temp['self'] = copy(temp[ground_truth])
        #     temp.insert(0,'self',temp.pop('self'),allow_duplicates = True)
        #     complete_time_series.append(temp)
        #     all_cols.append(temp.columns)
        # time_series = complete_time_series
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
            temp = construct(time_series[ind][all_cols[ind]], time_series[ind]["Label"], tes_ind, time_series[ind].shape[0]-1, lags, horizon, 'log_1d_return')
            X_te.append(temp[0])
            y_te.append(temp[1])
        else:
            X_te = None
            y_te = None
    
    
    
    return X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params


