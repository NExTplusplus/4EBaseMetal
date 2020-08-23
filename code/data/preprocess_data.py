from copy import copy
import os
import sys
import pandas as pd

from utils.data_preprocess_version_control import *
from utils.data_preprocess_functions import reset_split_dates, process_missing_value as pmv

def preprocess_data(time_series, LME_dates, horizon, ground_truth_columns, lags,  split_dates, norm_params, tech_params, version_params):
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
            column_list: (array)An array that holds the name of the columns as sorted in the dataframe (inaccessible from data)
            prediction_dates: (array) An array that holds the dates that are predicted in a live testing format
           
    """
    parameters = {'time_series':time_series, 'LME_dates': LME_dates, 'horizon': horizon, 
                    'ground_truth_columns': ground_truth_columns, 'lags': lags, 'split_dates':split_dates,
                    'norm_params':norm_params, 'tech_params': tech_params}
    parameters['norm_params'] = generate_norm_params(version_params['generate_norm_params'], 1 if norm_params['date'] else 0)
    parameters['tech_params'] = generate_tech_params(version_params['generate_tech_params'])
    parameters['strat_params'],parameters['activation_params'] = generate_strat_params(ground_truth_columns[0], horizon, version_params['generate_strat_params'])
    parameters['SD_params'] = generate_SD_params(version_params['generate_SD_params'])
    original_test_date = split_dates[2]
    
    
    #deal with the abnormal data such as outliers and missing data
    parameters['time_series'] = deal_with_abnormal_value(parameters,version_params["deal_with_abnormal_value"])

    #Extract the rows with dates where LME has trading operations
    LME_dates = sorted(set(LME_dates).intersection(parameters['time_series'].index.values.tolist()))

    #generate labels and strategy signals
    parameters['time_series'] = parameters['time_series'].loc[LME_dates]
    if version_params['labelling']=='v2':
        parameters['spot_price'] = spot_price_normalization(parameters)
    parameters['labels'] = labelling(parameters, version_params['labelling'])
    parameters['time_series'] = process_missing_value(parameters,version_params['process_missing_value'])
    parameters['org_cols'] = time_series.columns.values.tolist()

    # we construct the signal strategy of LME
    parameters['time_series'] = strategy_signal(parameters,version_params['strategy_signal'])

    # reset the split date before dealing with the abnormal value
    split_dates = reset_split_dates(parameters['time_series'],split_dates)


    '''
    Normalize, create technical indicators, handle outliers and rescale data
    '''
    #holder for special columns that are not to be scale
    parameters['cat_cols'] = []

    # normalize OI, volume and spread
    parameters['time_series'], parameters['norm_check'] = normalize_without_1d_return(parameters, version_params['normalize_without_1d_return'])

    # construct the techincal indicator of COMEX and LME.Because we use the LME dates so we will lose some comex's information
    parameters['time_series'] = technical_indication(parameters, version_params['technical_indication'])

    if parameters['norm_params']['date']:
        print("date")
        parameters['cat_cols'] = ['day','month']
        parameters['time_series'] = insert_date_into_feature(parameters)

    # generate supply and demand indicators
    parameters['time_series'] = supply_and_demand(parameters,version_params['supply_and_demand'])

    # remove the data columns that are not required in the final result
    print("origin features",parameters['time_series'].columns.values.tolist())
    parameters['time_series'], parameters['org_cols'] = remove_unused_columns(parameters, version_params['remove_unused_columns'])

    # normalize the prices into returns when toggled
    parameters['time_series'] = price_normalization(parameters,version_params['price_normalization'])

    # remove missing values in data
    parameters['time_series'] = process_missing_value(parameters, version_params['process_missing_value'])
    split_dates = reset_split_dates(parameters['time_series'],split_dates)
    print("features",sorted(parameters['time_series'].columns.values.tolist()))

    # identify columns that are of form -1,0,1
    for col in parameters['time_series'].columns.values.tolist():
        if len(parameters['time_series'][col].unique().tolist()) <= 3:
            parameters['cat_cols'].append(col)

    # normalize across data columns with scaling each column
    parameters['time_series'] = scaling(parameters,version_params['scaling'])
    complete_time_series = []

    # sort data columns according to alphabetical order 
    # when dealing with even versions, alphabetical order ensures that the data columns are similarly sorted
    parameters['time_series'] = parameters['time_series'][sorted(parameters['time_series'].columns)]

    parameters['all_cols'] = []
    if len(ground_truth_columns) > 1:
        print("mistake")
    else:
        parameters['time_series'] = [parameters['time_series']]
        parameters['all_cols'].append(parameters['time_series'][0].columns.tolist())
    if version_params['labelling']=='v2': 
        parameters['all_cols'][0].append('Spot_price')
    
    
    # Merge labels with time series dataframe to be passed into construct
    for ind in range(len(parameters['time_series'])):
        
        if version_params['labelling']=='v2': 
            # for labelling of v2, spot price is required to be stored in results
            parameters['time_series'][ind] = pd.concat([parameters['time_series'][ind], parameters['spot_price'][ind]],sort = True, axis = 1)
        
        parameters['time_series'][ind] = pd.concat([parameters['time_series'][ind], parameters['labels'][ind]],sort = True, axis = 1)

        if "live" in tech_params.keys():
            # if live testing is toggled, then we fill the results that we cannot acquire as 0
            parameters['time_series'][ind]["Label"][-horizon:] = [0]*horizon
            if "Regression Label" in parameters['labels'][0].columns:
                parameters['time_series'][ind]["Regression Label"][-horizon:] = [0]*horizon
        
        parameters['time_series'][ind] = pmv(parameters['time_series'][ind])
        split_dates = reset_split_dates(parameters['time_series'][ind],split_dates)

    
    #create 3d array with dimensions (n_samples, lags, n_features)

    #get training index
    tra_ind = parameters['time_series'][0].index.get_loc(split_dates[0]) - horizon + 1
    if tra_ind < lags - 1:
        tra_ind = lags - 1

    #get validation starting index
    val_start_ind = parameters['time_series'][0].index.get_loc(split_dates[1])
    assert val_start_ind >= lags - 1, 'without training data'

    #get validation ending index
    if split_dates[2] == original_test_date and "live" not in tech_params.keys():
        val_end_ind = parameters['time_series'][0].index.get_loc(split_dates[2])
    else:
        val_end_ind = parameters['time_series'][0].index.get_loc(split_dates[2])+1

    X_tr = []
    y_tr = []
    X_va = []
    y_va = []

    for ind in range(len(parameters['time_series'])):
        # construct the training
        parameters['start_ind'] = tra_ind
        parameters['end_ind'] = val_start_ind - horizon + 1
        temp = construct(ind, parameters,version_params['construct'])
        X_tr.append(temp[0])
        y_tr.append(temp[1])

        # construct the validation
        parameters['start_ind'] = val_start_ind
        parameters['end_ind'] = val_end_ind
        temp = construct(ind, parameters,version_params['construct'])
        X_va.append(temp[0])
        y_va.append(temp[1])

    if "live" not in tech_params.keys():
        return X_tr, y_tr, X_va, y_va,parameters['norm_check'], parameters['all_cols']
    else:
        return X_tr, y_tr, X_va, y_va,parameters['norm_check'], parameters['all_cols'], parameters["time_series"][0].index.values.tolist()[val_start_ind:val_end_ind]


