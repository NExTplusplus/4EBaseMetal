from copy import copy
import numpy as np
import os
import sys
import pandas as pd
import json

from utils.version_control_functions import *
from utils.read_data import process_missing_value_v3

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
def load_data(time_series, LME_dates, horizon, ground_truth_columns, lags,  split_dates, norm_params, tech_params, version_params):
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
    parameters = {'time_series':time_series, 'LME_dates': LME_dates, 'horizon': horizon, 
                    'ground_truth_columns': ground_truth_columns, 'lags': lags, 'split_dates':split_dates,
                    'norm_params':norm_params, 'tech_params': tech_params}
    parameters['norm_params'] = generate_norm_params(version_params['generate_norm_params'], 1 if norm_params['xgboost'] else 0)
    parameters['tech_params'] = generate_tech_params(version_params['generate_tech_params'])
    parameters['strat_params'],parameters['activation_params'] = generate_strat_params(ground_truth_columns[0], horizon, version_params['generate_strat_params'])
    parameters['SD_params'] = generate_SD_params(version_params['generate_SD_params'])
    original_test_date = split_dates[2]
    '''
    deal with the abnormal data which we found in the data. 
    '''
    
    parameters['time_series'] = deal_with_abnormal_value(parameters,version_params["deal_with_abnormal_value"])
    '''
    Extract the rows with dates where LME has trading operations
    and generate labels and strategy signals
    '''
    LME_dates = sorted(set(LME_dates).intersection(parameters['time_series'].index.values.tolist()))
    parameters['time_series'] = parameters['time_series'].loc[LME_dates]
    if version_params['labelling']=='v2':
        parameters['spot_price'] = spot_price_normalization(parameters)
    parameters['labels'] = labelling(parameters, version_params['labelling'])
    parameters['time_series'] = process_missing_value(parameters,version_params['process_missing_value'])
    parameters['org_cols'] = time_series.columns.values.tolist()

    # we construct the signal strategy of LME
    parameters['time_series'] = strategy_signal(parameters,version_params['strategy_signal'])
    #print(parameters['time_series'].columns.values.tolist())
    # save_data("i3",parameters['time_series'],parameters['time_series'].columns.values.tolist())
    # reset the split date before dealing with the abnormal value
    split_dates = reset_split_dates(parameters['time_series'],split_dates)


    '''
    Normalize, create technical indicators, handle outliers and rescale data
    '''
    parameters['cat_cols'] = []
    parameters['time_series'], parameters['norm_check'] = normalize_without_1d_return(parameters, version_params['normalize_without_1d_return'])
    # construct the techincal indicator of COMEX and LME.Because we use the LME dates so we will lose some comex's information
    parameters['time_series'] = technical_indication(parameters, version_params['technical_indication'])
    # save_data("i4",parameters['time_series'],parameters['time_series'].columns.values.tolist())
    if parameters['norm_params']['xgboost']:
        print("xgboost")
        parameters['cat_cols'] = ['day','month']
        parameters['time_series'] = insert_date_into_feature(parameters)
    # remove the unused feature includes the unused column and the original useless feature
    parameters['time_series'] = supply_and_demand(parameters,version_params['supply_and_demand'])
    print("origin features",parameters['time_series'].columns.values.tolist())
    parameters['time_series'], parameters['org_cols'] = remove_unused_columns(parameters, version_params['remove_unused_columns'],ground_truth_columns)
    # save_data("i5",parameters['time_series'],parameters['time_series'].columns.values.tolist())
    #print("values",parameters['time_series'].columns.values.tolist())
    parameters['time_series'] = price_normalization(parameters,version_params['price_normalization'])
    parameters['time_series'] = process_missing_value(parameters, version_params['process_missing_value'])
    split_dates = reset_split_dates(parameters['time_series'],split_dates)
    print("features",parameters['time_series'].columns.values.tolist())
    for col in parameters['time_series'].columns.values.tolist():
        if len(parameters['time_series'][col].unique().tolist()) <= 3:
            parameters['cat_cols'].append(col)
    parameters['time_series'] = scaling(parameters,version_params['scaling'])
    complete_time_series = []
    parameters['time_series'] = parameters['time_series'][sorted(parameters['time_series'].columns)]
    
    parameters['all_cols'] = []
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
        # parameters['time_series'].insert(0,ground_truth_columns[0],parameters['time_series'].pop(ground_truth_columns[0]),allow_duplicates = True)
        parameters['time_series'] = [parameters['time_series']]
        parameters['all_cols'].append(parameters['time_series'][0].columns.tolist())
    if version_params['labelling']=='v2': 
        parameters['all_cols'][0].append('Spot_price')
    
    

    '''
    Merge labels with time series dataframe
    '''
    for ind in range(len(parameters['time_series'])):
        if version_params['labelling']=='v2': 
            parameters['time_series'][ind] = pd.concat([parameters['time_series'][ind], parameters['spot_price'][ind]],sort = True, axis = 1)
        parameters['time_series'][ind] = pd.concat([parameters['time_series'][ind], parameters['labels'][ind]],sort = True, axis = 1)
        if "live" in tech_params.keys():
            parameters['time_series'][ind]["Label"][-horizon:] = [0]*horizon
        
        parameters['time_series'][ind] = process_missing_value_v3(parameters['time_series'][ind])
        split_dates = reset_split_dates(parameters['time_series'][ind],split_dates)
        # save_data(ground_truth_columns[0]+"i6",parameters['time_series'][0],parameters['time_series'][0].columns.values.tolist())

    '''
    create 3d array with dimensions (n_samples, lags, n_features)
    '''



    tra_ind = 0
    if tra_ind < lags - 1:
        tra_ind = lags - 1
    val_ind = parameters['time_series'][0].index.get_loc(split_dates[1])
    assert val_ind >= lags - 1, 'without training data'
    if split_dates[2] == original_test_date and "live" not in tech_params.keys():
        tes_ind = parameters['time_series'][0].index.get_loc(split_dates[2])
    else:
        tes_ind = parameters['time_series'][0].index.get_loc(split_dates[2])+1

    X_tr = []
    y_tr = []
    X_va = []
    y_va = []
    X_te = []
    y_te = []
    # print(len(time_series))
    for ind in range(len(parameters['time_series'])):
        # construct the training
        parameters['start_ind'] = tra_ind
        parameters['end_ind'] = val_ind
        temp = construct(ind, parameters,version_params['construct'])
        X_tr.append(temp[0])
        y_tr.append(temp[1])

        # construct the validation
        parameters['start_ind'] = val_ind
        parameters['end_ind'] = tes_ind
        temp = construct(ind, parameters,version_params['construct'])
        X_va.append(temp[0])
        y_va.append(temp[1])

        # construct the testing
        parameters['start_ind'] = tes_ind
        parameters['end_ind'] = parameters['time_series'][ind].shape[0]-1
        if tes_ind < parameters['time_series'][ind].shape[0]-horizon-1:
            temp = construct(ind, parameters,version_params['construct'])
            X_te.append(temp[0])
            y_te.append(temp[1])
        else:
            X_te = None
            y_te = None
    
    
    if "live" not in tech_params.keys():
        return X_tr, y_tr, X_va, y_va, X_te, y_te,parameters['norm_check'], parameters['all_cols']
    else:
        return X_tr, y_tr, X_va, y_va, X_te, y_te,parameters['norm_check'], parameters['all_cols'], parameters["time_series"][0].index.values.tolist()[val_ind:tes_ind]


