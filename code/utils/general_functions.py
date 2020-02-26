import os
import json
from utils.read_data import read_data_NExT
from data.load_data import load_data
import pandas as pd
import numpy as np
from copy import copy

def even_version(version):
  '''
    check if version is odd numbered or even numbered
    input:
      version:  string of version
    output:
      even: boolean stating whether the version number is even or otherwise 
  '''
  version_num = int(version[1:])
  return (version_num % 2 == 0)

def assert_version(version,path):
  '''
    confirm that the version and path are as predetermined
    input:
      version:  string which hold the version
      path:     string which holds the path to the configuration file
    There is not output, however if the path and version do not coincide then the assertions will fail,
    resulting in the termination of the program
  '''
  if version in ["v5","v7"]:
    #requires global data
    assert path == "exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf"
  elif version in ["v3","v23","v24","v28","v30","v37"]:
    assert path == "exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf"
  elif version in ["v9","v10","v12","v16","v26"]:
    assert path == "exp/online_v10.conf"
  elif version in ["v31","v32"]:
    assert path == "exp/supply and demand.conf"
  elif version in ["v33","v35"]:
    assert path == "exp/TP_v1.conf"
  else:
    print("Version out of bounds!")
    os.exit()



def get_relevant_dates(date, length, version):
  '''
    generate relevant dates for tuning,training and testing
    input:
      date:     string containing the time point's year, month and day
      length:   length of training period required
      version:  tuning,training or testing
    output:
  '''
  #ensure date is of correct format
  assert date[4] == '-' and date[7] == '-' and int(date[5:7]) <=12 and int(date[5:7]) >=1

  month = int(str(date).split("-")[1])
  year = int(str(date).split("-")[0])
  if version == "tune":
    '''
      requires enough data for 5 rolling half years and their respective training periods of length years before stated date
    '''
    if month <= 6:
      start_year = year - 3 - length
      start_time = str(start_year)+"-07-01"
      end_time = str(year) +"-07-01"
    else:
      start_year = year - 2 - length
      end_year = year + 1
      start_time = str(start_year)+"-01-01"
      end_time = str(end_year)+"-01-01"
    
    return start_time,end_time

  elif version == "train" or version == "test":
    '''
      only requires the preceding years for training
      check for labels is relegated to later
    '''
    if month <=6:
      start_year = year - length
      start_time = str(start_year)+"-01-01"
      evalidate_year = int(year)
      evalidate_date = str(evalidate_year)+"-01-01"
      end_time = str(date)
    else:
      start_year = year - length
      start_time = str(start_year)+"-07-01"
      evalidate_date = str(year)+"-07-01"
      end_time = str(date)
    return start_time,evalidate_date
  else:
    print("Version out of bounds!")
    os.exit()

def assert_labels(dates,split_date,horizon):
  '''
    ensure that there are enough days to generate the correct labels for training period based on horizon
  '''
  df = pd.DataFrame(index =dates)
  index_diff = df.index.get_loc(split_date[2],method = "bfill") - df.index.get_loc(split_date[1],method = "bfill") + 1
  assert index_diff >= horizon


def read_data_with_specified_columns(source,path,start_date):
  '''
    read data from either 4E or NExT database with path to generate configuration, to ensure that the data is consistent across both databases
    input:
      source:     4E/NExT
      path:       path to configuration file, which specifies the columns to be loaded in
      start_date: date to begin collecting of data
    output:
      time_series:  pandas dataframe that holds all available information from start date with configuration as stated in path
      LME_dates:    a list of dates of which LME is trading
  '''
  with open(os.path.join(os.getcwd(),path)) as fin:
    fname_columns = json.load(fin)
  time_series, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
  time_series = pd.concat(time_series, axis = 1, sort = True)
  columns_to_be_stored = time_series.columns.values.tolist()

  if source=='4E':
    from utils.read_data import read_data_v31_4E
    time_series, LME_dates = read_data_v31_4E("2003-11-12")
    time_series = time_series[columns_to_be_stored]
    os.chdir("NEXT/4EBaseMetal")
  return time_series,LME_dates,len(fname_columns[0])

def prepare_data(ts,LME_dates,horizon,ground_truth_list,lag,split_date,version_params,xgboost = False,live = False,metal_id_bool = False):
  '''
    prepare data based on configuration as stated in parameters
    input:
      time_series:        pandas dataframe that holds all information
      LME_dates:          a list of dates of which LME has trading operations
      horizon:            amount of days ahead that we need to predict
      ground_truth_list:  list of ground truths
      lag:                number of lags for load data to build
      split_date:         list of dates with 1st being beginning of training, 2nd being beginning of validation, and 3rd being end of validation
      version_params:     dictionary holding the trigger for version control within load_data
      xgboost:            boolean which controls inclusion of date
      live:               boolean which controls inclusion of live
      metal_id_bool:      boolean which controls inclusion of metal_id
    output:
      final_X_tr:         training samples
      final_y_tr:         training labels
      final_X_va:         validation samples
      final_y_va:         validation labels
      column_lag_list:    list of columns with lag taken into account

  '''

  #ensure that there is no mislabelling of training period due to lack of data
  assert_labels(LME_dates,split_date,horizon)

  #params that can be manually added
  norm_params = {"xgboost":xgboost}
  tech_params = {}
  if live:
    tech_params["live"] = None

  #initialize values
  final_X_tr = []
  final_y_tr = []
  final_X_va = []
  final_y_va = []
  i = 0

  #load data
  for ground_truth in ground_truth_list:
    print(norm_params)
    if live:
      X_tr, y_tr, X_va, y_va, X_te, y_te, norm_check,column_list,val_dates = load_data(ts,LME_dates,horizon,[ground_truth],lag,copy(split_date),copy(norm_params),copy(tech_params),version_params)
    else:
      val_dates = None
      X_tr, y_tr, X_va, y_va, X_te, y_te, norm_check,column_list = load_data(ts,LME_dates,horizon,[ground_truth],lag,copy(split_date),copy(norm_params),copy(tech_params),version_params)
    X_tr = np.concatenate(X_tr)
    X_tr = X_tr.reshape(len(X_tr),lag*len(column_list[0]))
    y_tr = np.concatenate(y_tr)
    X_va = np.concatenate(X_va)
    X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
    y_va = np.concatenate(y_va)

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

