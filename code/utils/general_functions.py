import os
import json
from data.preprocess_data import preprocess_data
import pandas as pd
import numpy as np
from copy import copy

#creates the split dates array which rolls forward every 6 months
#[[2009-01-01,2009-07-01,2014-07-01,2015-01-01,2015-07-01],...]
def rolling_half_year(start_date,end_date,length):
    '''
        Input   :   start_date(str) : date to start creating split_dates
                    end_date(str)   : date to stop creating split_dates
                    length(int)     : number of years for training set
        Output  :   split_dates(list)   : list of list. Each list holds 5 dates, data start,train start, val start, and val end, data end.
    '''
    start_year = start_date.split("-")[0]
    end_year = end_date.split("-")[0]
    split_dates = []

    for year in range(int(start_year),int(end_year)+1):
        split_dates.append([str(year-1)+"-07-01",str(year)+"-01-01",str(int(year)+length)+"-01-01",str(int(year)+length)+"-07-01",str(int(year)+length+1)+"-01-01"])
        split_dates.append([str(year)+"-01-01",str(year)+"-07-01",str(int(year)+length)+"-07-01",str(int(year)+length+1)+"-01-01",str(int(year)+length+1)+"-07-01"])
    
    while split_dates[0][1] < start_date:
        del split_dates[0]
    
    while split_dates[-1][-2] > end_date:
        del split_dates[-1]
    
    return split_dates

# check if version is even numbered
def even_version(version):
    '''
        input:  version:  string of version
        output: even: boolean stating whether the version number is even or otherwise 
    '''
    version_num = int(version[1:])
    return (version_num % 2 == 0)

#generate the configuration file path holding the data columns that are required for a particular version
def generate_config_path(version):
    '''
        input:  version:  string which hold the version
        output: config_path: string which holds the path to configuration file
    '''
    if version in ["v5","v7"]:
        #requires global data
        return "exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf"
    elif version in ["v3","v23","v24","v28","v30","v37","v39","v41","v43"]:
        return "exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf"
    elif version in ["v9","v10","v12","v16","v26"]:
        return "exp/online_v10.conf"
    elif version in ["v31","v32"]:
        return "exp/supply and demand.conf"
    elif version in ["v33","v35"]:
        return "exp/TP_v1.conf"
    elif version in ['r2']:
        return 'exp/regression_r2.conf'
    else:
        print("Version out of bounds!")
        os.exit()


#generate relevant dates for tuning,training and testing
def get_relevant_dates(date, length, version):
    '''
        input:  date:     string containing the time point's year, month and day
                length:   length of training period required
                version:  tuning,training or testing
    '''
    #ensure date is of correct format
    assert date[4] == '-' and date[7] == '-' and int(date[5:7]) <=12 and int(date[5:7]) >=1

    month = int(str(date).split("-")[1])
    year = int(str(date).split("-")[0])
    if version == "tune":
        
        #requires enough data for 5 rolling half years and their respective training periods of length years before stated date
        if month <= 6:
            start_year = year - 3 - length
            start_time = str(start_year)+"-07-01"
            end_time = str(year) +"-01-01"
        else:
            start_year = year - 3 - length + 1
            end_year = year
            start_time = str(start_year)+"-01-01"
            end_time = str(end_year)+"-07-01"
        
        return start_time,end_time

    elif version == "train" or version == "test":
        #only requires the preceding years for training
        if month <=6:
            start_year = year - length - 1
            start_time = str(start_year)+"-07-01"
            train_time = str(start_year+1) +"-01-01"
            evalidate_year = int(year)
            evalidate_date = str(evalidate_year)+"-01-01"
            end_time = str(date)
        else:
            start_year = year - length
            start_time = str(start_year)+"-01-01"
            train_time = str(start_year) +"-07-01"
            evalidate_date = str(year)+"-07-01"
            end_time = str(date)
        
        print(start_time,evalidate_date)
        return start_time,train_time,evalidate_date
    else:
        print("Version out of bounds!")
        os.exit()

#read data from either 4E or NExT database with path to generate configuration, to ensure that the data is consistent across both databases
def read_data_with_specified_columns(source,path,start_date):
    '''
        input:  source:     4E/NExT
                path:       path to configuration file, which specifies the columns to be loaded in
                start_date: date to begin collecting of data
        output: time_series:  pandas dataframe that holds all available information from start date with configuration as stated in path
                LME_dates:    a list of dates of which LME is trading
    '''
    with open(os.path.join(os.getcwd(),path)) as fin:
        fname_columns = json.load(fin)

    if source == "NExT":
        from utils.read_data import read_data_NExT
        time_series, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
    elif source=='4E':
        from utils.read_data import read_data_4E
        time_series, LME_dates = read_data_4E("2003-11-12")
        os.chdir("NEXT/4EBaseMetal")
    return time_series,LME_dates,len(fname_columns[0])

#prepare data based on configuration as stated in parameters
def prepare_data(time_series,LME_dates,horizon,ground_truth_list,lag,split_date,version_params,date = False,live = False,metal_id_bool = False, reshape = True):
    '''
        
        input:  time_series:        pandas dataframe that holds all information
                LME_dates:          a list of dates of which LME has trading operations
                horizon:            amount of days ahead that we need to predict
                ground_truth_list:  list of ground truths
                lag:                number of lags for load data to build
                split_date:         list of dates with 1st being beginning of training, 2nd being beginning of validation, and 3rd being end of validation
                version_params:     dictionary holding the trigger for version control within load_data
                date:               boolean which controls inclusion of date as data column
                live:               boolean which controls inclusion of live
                metal_id_bool:      boolean which controls inclusion of metal_id
        output: final_X_tr:         training samples
                final_y_tr:         training labels
                final_X_va:         validation samples
                final_y_va:         validation labels
                column_lag_list:    list of columns with lag taken into account
    '''

    #params that can be manually added
    norm_params = {"date":date}
    tech_params = {}
    if live:
        tech_params["live"] = None

    #initialize values
    final_X_tr = []
    final_y_tr = []
    final_X_va = []
    final_y_va = []
    i = 0

    #preprocess data
    for ground_truth in ground_truth_list:
        print(norm_params)
        if live:
            X_tr, y_tr, X_va, y_va, norm_check,column_list,val_dates = preprocess_data(time_series,LME_dates,horizon,[ground_truth],lag,copy(split_date),copy(norm_params),copy(tech_params),version_params)
        else:
            val_dates = None
            X_tr, y_tr, X_va, y_va, norm_check,column_list = preprocess_data(time_series,LME_dates,horizon,[ground_truth],lag,copy(split_date),copy(norm_params),copy(tech_params),version_params)
        X_tr = np.concatenate(X_tr)
        y_tr = np.concatenate(y_tr)
        X_va = np.concatenate(X_va)
        y_va = np.concatenate(y_va)
        if reshape:
            X_tr = X_tr.reshape(len(X_tr),lag*len(column_list[0]))
            X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))

        
        #if version is even, add metal id to X
        if metal_id_bool and not live:
            metal_id = [0,0,0,0,0,0]
            metal_id[i] = 1
            X_tr = np.append(X_tr,[metal_id]*len(X_tr),axis = 1)
            X_va = np.append(X_va,[metal_id]*len(X_va),axis = 1)
            final_X_tr.append(X_tr)
            final_y_tr.append(y_tr)
            final_X_va.append(X_va)
            final_y_va.append(y_va)
            i+=1
        else:
            final_X_tr = X_tr
            final_y_tr = y_tr
            final_X_va = X_va
            final_y_va = y_va
    
    #sort metal by timestamp then metal, not by metal then timestamp
    if metal_id_bool and not live:
        final_X_tr = [np.transpose(arr) for arr in np.dstack(final_X_tr)]
        final_y_tr = [np.transpose(arr) for arr in np.dstack(final_y_tr)]
        final_X_va = [np.transpose(arr) for arr in np.dstack(final_X_va)]
        final_y_va = [np.transpose(arr) for arr in np.dstack(final_y_va)]
        final_X_tr = np.reshape(final_X_tr,[np.shape(final_X_tr)[0]*np.shape(final_X_tr)[1],np.shape(final_X_tr)[2]])
        final_y_tr = np.reshape(final_y_tr,[np.shape(final_y_tr)[0]*np.shape(final_y_tr)[1],np.shape(final_y_tr)[2]])
        final_X_va = np.reshape(final_X_va,[np.shape(final_X_va)[0]*np.shape(final_X_va)[1],np.shape(final_X_va)[2]])
        final_y_va = np.reshape(final_y_va,[np.shape(final_y_va)[0]*np.shape(final_y_va)[1],np.shape(final_y_va)[2]])
    
    #if live version needs metal_id
    if metal_id_bool and live:
        metal_id = [0,0,0,0,0,0]
        if ground_truth == "LME_Co_Spot":
            metal_id[0] = 1
        elif ground_truth == "LME_Al_Spot":
            metal_id[1] = 1
        elif ground_truth == "LME_Ni_Spot":
            metal_id[2] = 1
        elif ground_truth == "LME_Ti_Spot":
            metal_id[3] = 1
        elif ground_truth == "LME_Zi_Spot":
            metal_id[4] = 1
        elif ground_truth == "LME_Le_Spot":
            metal_id[5] = 1
        
        final_X_tr = np.append(X_tr,[metal_id]*len(X_tr),axis = 1)
        final_X_va = np.append(X_va,[metal_id]*len(X_va),axis = 1)
                
    #generate column list with lags
    column_lag_list = []
    column_name = []
    for i in range(lag):
        for item in column_list[0]:
            if metal_id_bool:
                new_item = item.replace(ground_truth[:6],"LME_Le")+"_"+str(lag-i)
            else:
                new_item = item+"_"+str(lag-i)
            column_lag_list.append(new_item)

    #add metal id to column lag list
    if metal_id_bool:
        column_lag_list.append("Co")
        column_lag_list.append("Al")
        column_lag_list.append("Ni")
        column_lag_list.append("Ti")
        column_lag_list.append("Zi")
        column_lag_list.append("Le")
    
    return final_X_tr,final_y_tr,final_X_va,final_y_va,val_dates,column_lag_list

