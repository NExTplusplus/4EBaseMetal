import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
from data.load_data import load_data
from model.logistic_regression import LogReg
from utils.transform_data import flatten
from utils.construct_data import rolling_half_year
from utils.log_reg_functions import objective_function, loss_function
from utils.read_data import read_data_NExT
import warnings
import xgboost as xgb
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn import metrics
from sklearn.model_selection import KFold
from utils.version_control_functions import generate_version_params
from sklearn.externals import joblib
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '4EBaseMetal')))


class XGBoost_online():
  """
  lag: the window size of the data feature
  horizon: the time horizon of the predict target
  version: the version of the feature
  gt: the ground_truth metal name
  date: the last date of the prediction
  source: the data source
  """
  def __init__(self,
        lag,
        horizon,
        version,
        gt,
        date,
        source,
        path):
    self.lag = lag
    self.horizon = horizon
    self.version = version
    self.gt = gt
    self.date = date
    self.source = source
    self.path = path
  """
  this function is used to choose the parameter
  """
  def choose_parameter(self):
    print("begin to choose the parameter")

    #assert that the configuration path is correct
    if self.version in ['v5','v7']:
      assert (self.path == "exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf")
    elif self.version in ["v3","v23"]:
      assert (self.path == "exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf")
    elif self.version in ["v9","v10","v12"]:
      assert (self.path == "exp/online_v10.conf")

    #retrieve column list based on configuration path
    with open(os.path.join(os.getcwd(),self.path)) as fin:
      fname_columns = json.load(fin)
    temporary_df, not_used = read_data_NExT(fname_columns[0], "2003-11-12")
    temporary_df = pd.concat(temporary_df, axis = 1, sort = True)
    columns_to_be_stored = temporary_df.columns.values.tolist()

    """
    read the data from the 4E or NExT database
    """
    if self.source=='4E':
      from utils.read_data import read_data_v5_4E
      time_series, LME_dates = read_data_v5_4E("2003-11-12")

    elif self.source=='NExT':
      with open(os.path.join(os.getcwd(),self.path)) as fin:
        fname_columns = json.load(fin)
      data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
      time_series = pd.concat(data_list, axis = 1, sort = True)
    time_series = time_series[columns_to_be_stored]

    """
    begin to split the train data
    """
    today = self.date

    if LME_dates[-self.horizon] < today:
      print("the data is not enough for training")
      os.exit(0)

    month = str(today).split("-")[1]
    year = str(today).split("-")[0]
    if month <="06":
      start_year = int(year)-8
      start_time = str(start_year)+"-07-01"
      end_time = str(year) +"-07-01"
    else:
      year = str(today).split("-")[0]
      start_year = int(year)-7
      end_year = int(year)+1
      start_time = str(start_year)+"-01-01"
      end_time = str(end_year)+"-01-01"
    length = 5
    split_dates = rolling_half_year(start_time,end_time,length)
    split_dates  =  split_dates[:]
    print(split_dates)
    print(sys.path[0])
    """
    generate the version
    """			
    version_params=generate_version_params(self.version)
    ans = {"C":[]}
    for s, split_date in enumerate(split_dates[:-1]):
      print("the train date is {}".format(split_date[0]))
      print("the test date is {}".format(split_date[1]))
      norm_volume = "v1"
      norm_3m_spread = "v1"
      norm_ex = "v1"
      len_ma = 5
      len_update = 30
      tol = 1e-7
      norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
              'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
      tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                      'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
      ts = copy(time_series.loc[split_date[0]:split_dates[s+1][2]])
      
      final_X_tr = []
      final_y_tr = []
      final_X_va = []
      final_y_va = []
      """
      load_data
      """
      
      if self.version in ['v3','v5','v7','v9','v23']:
        X_tr, y_tr, X_va, y_va, X_te, y_te, norm_params,column_list = load_data(ts,LME_dates,self.horizon,self.gt,self.lag,split_date,norm_params,tech_params,version_params)
        column_lag_list = []
        column_name = []
        for i in range(lag):
          for item in column_list[0]:
            new_item = item+"_"+str(lag-i)
            column_lag_list.append(new_item)
        X_tr = np.concatenate(X_tr)
        final_X_tr = X_tr.reshape(len(X_tr),lag*len(column_list[0]))
        final_y_tr = np.concatenate(y_tr)
        X_va = np.concatenate(X_va)
        final_y_va = np.concatenate(y_va)
        final_X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
        
      else:
        i = 0
        #iterate over different ground truths for data loading
        for ground_truth in ['LME_Co_Spot','LME_Al_Spot','LME_Ni_Spot','LME_Ti_Spot','LME_Zi_Spot','LME_Le_Spot']:
          print(ground_truth)
          metal_id = [0,0,0,0,0,0]
          metal_id[i] = 1
          
          #load data
          X_tr, y_tr, X_va, y_va, X_te, y_te, norm_check,column_list = load_data(copy(ts),LME_dates,horizon,[ground_truth],lag,copy(split_date),norm_params,tech_params,version_params)
          
          #post load process and metal id extension
          X_tr = np.concatenate(X_tr)
          X_tr = X_tr.reshape(len(X_tr),lag*len(column_list[0]))
          X_tr = np.append(X_tr,[metal_id]*len(X_tr),axis = 1)
          y_tr = np.concatenate(y_tr)
          X_va = np.concatenate(X_va)
          X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
          X_va = np.append(X_va,[metal_id]*len(X_va),axis = 1)
          y_va = np.concatenate(y_va)
          final_X_tr.append(X_tr)
          final_y_tr.append(y_tr)
          final_X_va.append(X_va)
          final_y_va.append(y_va)
          i+=1
        
        #sort by time, not by metal
        final_X_tr = [np.transpose(arr) for arr in np.dstack(final_X_tr)]
        final_y_tr = [np.transpose(arr) for arr in np.dstack(final_y_tr)]
        final_X_va = [np.transpose(arr) for arr in np.dstack(final_X_va)]
        final_y_va = [np.transpose(arr) for arr in np.dstack(final_y_va)]
        final_X_tr = np.reshape(final_X_tr,[np.shape(final_X_tr)[0]*np.shape(final_X_tr)[1],np.shape(final_X_tr)[2]])
        final_y_tr = np.reshape(final_y_tr,[np.shape(final_y_tr)[0]*np.shape(final_y_tr)[1],np.shape(final_y_tr)[2]])
        final_X_va = np.reshape(final_X_va,[np.shape(final_X_va)[0]*np.shape(final_X_va)[1],np.shape(final_X_va)[2]])
        final_y_va = np.reshape(final_y_va,[np.shape(final_y_va)[0]*np.shape(final_y_va)[1],np.shape(final_y_va)[2]])
        
        column_lag_list = []
        column_name = []
        for i in range(lag):
          for item in column_list[0]:
            new_item = item+"_"+str(lag-i)
            column_lag_list.append(new_item)
        column_lag_list.append("Co")
        column_lag_list.append("Al")
        column_lag_list.append("Ni")
        column_lag_list.append("Ti")
        column_lag_list.append("Zi")
        column_lag_list.append("Le")
      train_dataframe = pd.DataFrame(final_X_tr,columns=column_lag_list)
      train_X = train_dataframe.loc[:,column_lag_list]
      train_y = pd.DataFrame(final_y_tr,columns=['result'])
      
      test_dataframe = pd.DataFrame(final_X_va,columns=column_lag_list)
      test_X = test_dataframe.loc[:,column_lag_list] 
      n_splits=args.k_folds
      """
      tune xgboost hyper parameter
      """
      for max_depth in [4,5,6]:
        for learning_rate in [0.7,0.8,0.9]:
          for gamma in [0.7,0.8,0.9]:
            for min_child_weight in [3,4,5]:
              for subsample in [0.7,0.85,0.9]:
                from sklearn.metrics import accuracy_score
                  model = xgb.XGBClassifier(max_depth=max_depth,
                              learning_rate = learning_rate,
                              n_estimators=500,
                              silent=True,
                              nthread=10,
                              gamma=gamma,
                              min_child_weight=min_child_weight,
                              subsample=subsample,
                              colsample_bytree=0.7,
                              colsample_bylevel=1,
                              reg_alpha=0.0001,
                              reg_lambda=1,
                              scale_pos_weight=1,
                              seed=1440,
                              missing=None)
                  folds = KFold(n_splits=n_splits)
                  scores = []
                  prediction = np.zeros((len(X_va), 1))
                  folder_index = []
                  
                  #generate k fold and train xgboost model
                  for fold_n, (train_index, valid_index) in enumerate(folds.split(train_X)):
                    #print("the train_index is {}".format(train_index))
                    #print("the test_index is {}".format(valid_index))
                    X_train, X_valid = train_X[column_lag_list].iloc[train_index], train_X[column_lag_list].iloc[valid_index]
                    y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]
                    model.fit(X_train, y_train,eval_metric='error',verbose=True,eval_set=[(X_valid,y_valid)],early_stopping_rounds=5)
                    y_pred_valid = model.predict(X_valid)
                    y_pred = model.predict_proba(test_X, ntree_limit=model.best_ntree_limit)[:, 1]
                    y_pred = y_pred.reshape(-1, 1)
                    #prediction_each_folder = y_pred
                    if fold_n == 0:
                      folder_1=y_pred
                      folder_1=folder_1.reshape(len(folder_1),1)
                    elif fold_n == 1:
                        
                      folder_2=y_pred
                      folder_2=folder_2.reshape(len(folder_2),1)
                    elif fold_n==2:
                        
                      folder_3 = y_pred
                      folder_3=folder_3.reshape(len(folder_3),1)
                    elif fold_n==3:
                        
                      folder_4 = y_pred
                      folder_4=folder_4.reshape(len(folder_4),1)
                    elif fold_n==4:
                        
                      folder_5=y_pred
                      folder_5=folder_5.reshape(len(folder_5),1)
                    elif fold_n==5:
                        
                      folder_6=y_pred
                      folder_6=folder_6.reshape(len(folder_6),1)
                    elif fold_n==6:
                        
                      folder_7=y_pred
                      folder_7=folder_7.reshape(len(folder_7),1)
                    elif fold_n==7:
                        
                      folder_8=y_pred
                      folder_8=folder_8.reshape(len(folder_8),1)
                    elif fold_n==8:
                        
                      folder_9=y_pred
                      folder_9=folder_9.reshape(len(folder_9),1)
                    elif fold_n==9:
                      folder_10=y_pred
                      folder_10=folder_10.reshape(len(folder_10),1) 
                  
                  #calculate the all folder voting
                  result = np.concatenate((folder_1,folder_2,folder_3,folder_4,folder_5,folder_6,folder_7,folder_8,folder_9,folder_10),axis=1)
                  final_list = []
                  for j in range(len(result)):
                    count_1=0
                    count_0=0
                    for item in result[j]:
                      if item > 0.5:
                        count_1+=1
                      else:
                        count_0+=1
                    if count_1>count_0:
                      final_list.append(1)
                    else:
                      final_list.append(0)
                  #print("the lag is {}".format(lag))
                  print("the all folder voting precision is {}".format(metrics.accuracy_score(final_y_va, final_list)))
                  
                  #calculate the near folder voting
                  result = np.concatenate((folder_6,folder_7,folder_8,folder_9,folder_10),axis=1)
                  final_list = []
                  for j in range(len(result)):
                    count_1=0
                    count_0=0
                    for item in result[j]:
                      if item > 0.5:
                        count_1+=1
                      else:
                        count_0+=1
                    if count_1>count_0:
                      final_list.append(1)
                    else:
                      final_list.append(0)
                  print("the near precision is {}".format(metrics.accuracy_score(final_y_va, final_list)))
                  
                  #calculate the far folder voting
                  result = np.concatenate((folder_1,folder_2,folder_3,folder_4,folder_5),axis=1)
                  final_list = []
                  for j in range(len(result)):
                    count_1=0
                    count_0=0
                    for item in result[j]:
                      if item > 0.5:
                        count_1+=1
                      else:
                        count_0+=1
                    if count_1>count_0:
                      final_list.append(1)
                    else:
                      final_list.append(0)
                  print("the far precision is {}".format(metrics.accuracy_score(final_y_va, final_list)))
                  #calculate the same folder voting
                  if split_date[1].split("-")[1]=='01':
                    result = np.concatenate((folder_1,folder_3,folder_5,folder_7,folder_9),axis=1)
                    final_list = []
                    for j in range(len(result)):
                      count_1=0
                      count_0=0
                      for item in result[j]:
                        if item > 0.5:
                          count_1+=1
                        else:
                          count_0+=1
                      if count_1>count_0:
                        final_list.append(1)
                      else:
                        final_list.append(0)
                    #print("the lag is {}".format(lag))
                    print("the same precision is {}".format(metrics.accuracy_score(final_y_va, final_list)))
                    
                    #calculate the reverse folder voting
                    result = np.concatenate((folder_2,folder_4,folder_6,folder_8,folder_10),axis=1)
                    final_list = []
                    for j in range(len(result)):
                      count_1=0
                      count_0=0
                      for item in result[j]:
                        if item > 0.5:
                          count_1+=1
                        else:
                          count_0+=1
                      if count_1>count_0:
                        final_list.append(1)
                      else:
                        final_list.append(0)
                    #print("the lag is {}".format(lag))
                    print("the reverse precision is {}".format(metrics.accuracy_score(final_y_va, final_list)))
                    print("the max_depth is {}".format(max_depth))
                    print("the learning_rate is {}".format(learning_rate))
                    print("the gamma is {}".format(gamma))
                    print("the min_child_weight is {}".format(min_child_weight))
                    print("the subsample is {}".format(subsample))
                  else:
                      
                    #calculate the same folder voting
                    result = np.concatenate((folder_2,folder_4,folder_6,folder_8,folder_10),axis=1)
                    final_list = []
                    for j in range(len(result)):
                      count_1=0
                      count_0=0
                      for item in result[j]:
                        if item > 0.5:
                          count_1+=1
                        else:
                          count_0+=1
                      if count_1>count_0:
                        final_list.append(1)
                      else:
                        final_list.append(0)
                    #print("the lag is {}".format(lag))
                    print("the same precision is {}".format(metrics.accuracy_score(final_y_va, final_list)))
                    
                    #calculate the reverse folder voting
                    result = np.concatenate((folder_1,folder_3,folder_5,folder_7,folder_9),axis=1)
                    final_list = []
                    for j in range(len(result)):
                      count_1=0
                      count_0=0
                      for item in result[j]:
                        if item > 0.5:
                          count_1+=1
                        else:
                          count_0+=1
                      if count_1>count_0:
                        final_list.append(1)
                      else:
                        final_list.append(0)
                    #print("the lag is {}".format(lag))
                    print("the reverse precision is {}".format(metrics.accuracy_score(final_y_va, final_list)))
                    print("the max_depth is {}".format(max_depth))
                    print("the learning_rate is {}".format(learning_rate))
                    print("the gamma is {}".format(gamma))
                    print("the min_child_weight is {}".format(min_child_weight))
                    print("the subsample is {}".format(subsample))
      print("the lag is {}".format(lag))
      print("the train date is {}".format(split_date[0]))
      print("the test date is {}".format(split_date[1]))


  #-------------------------------------------------------------------------------------------------------------------------------------#
  """
  this function is used to train the model and save it
  """
  def train(self,C=0.01,tol=1e-7,max_iter=100):
    print("begin to train")
    #assert that the configuration path is correct
    if self.version in ['v5','v7']:
      assert (self.path == "exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf")
    elif self.version in ["v3","v23"]:
      assert (self.path == "exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf")
    elif self.version in ["v9","v10","v12"]:
      assert (self.path == "exp/online_v10.conf")

    #retrieve column list based on configuration path
    with open(os.path.join(os.getcwd(),self.path)) as fin:
      fname_columns = json.load(fin)
    temporary_df, not_used = read_data_NExT(fname_columns[0], "2003-11-12")
    temporary_df = pd.concat(temporary_df, axis = 1, sort = True)
    columns_to_be_stored = temporary_df.columns.values.tolist()

    """
    read the data from the 4E or NExT database
    """
    if self.source=='4E':
      from utils.read_data import read_data_v5_4E
      time_series, LME_dates = read_data_v5_4E("2003-11-12")

    elif self.source=='NExT':
      with open(os.path.join(os.getcwd(),self.path)) as fin:
        fname_columns = json.load(fin)
      data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
      time_series = pd.concat(data_list, axis = 1, sort = True)
    time_series = time_series[columns_to_be_stored]

    """
    begin to split the train data
    """
    today = self.date
    month = str(today).split("-")[1]
    year = str(today).split("-")[0]
    if month <="06":
      if month=="01":
        day = int(str(today).split("-")[2])
        if day <=self.horizon:
          print("the data is not enough for training")
          os.exit(0)
        else:
          start_year = int(year)-5
          start_time = str(start_year)+"-01-01"
          evalidate_year = int(year)
          evalidate_date = str(evalidate_year)+"-01-01"
          end_time = str(today)
      else:
        start_year = int(year)-5
        start_time = str(start_year)+"-01-01"
        evalidate_year = int(year)
        evalidate_date = str(evalidate_year)+"-01-01"
        end_time = str(today)
    else:
      if month=="07":
        day = int(str(today).split("-")[2])
        if day<=self.horizon:
          print("the data is not enough for training")
          os.exit(0)
        else:
          year = str(today).split("-")[0]
          start_year = int(year)-5
          end_year = int(year)
          start_time = str(start_year)+"-07-01"
          evalidate_year = int(year)
          evalidate_date = str(evalidate_year)+"-07-01"
          end_time = str(today)
      else:
        year = str(today).split("-")[0]
        start_year = int(year)-5
        end_year = int(year)
        start_time = str(start_year)+"-07-01"
        evalidate_year = int(year)
        evalidate_date = str(evalidate_year)+"-07-01"
        end_time = str(today)
    split_dates  =  [start_time,evalidate_date,str(today)]
    """
    generate the version
    """			
    version_params=generate_version_params(self.version)
    ans = {"C":[]}
    print("the train date is {}".format(split_dates[0]))
    print("the test date is {}".format(split_dates[1]))
    norm_volume = "v1"
    norm_3m_spread = "v1"
    norm_ex = "v1"
    len_ma = 5
    len_update = 30
    tol = 1e-7
    pure_LogReg = LogReg(parameters={})
    norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
            'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
    tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                    'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
    ts = copy(time_series.loc[split_dates[0]:split_dates[2]])
    final_X_tr = []
    final_y_tr = []
    final_X_va = []
    final_y_va = []			
    """
    load_data
    """
    
    if self.version in ['v3','v5','v7','v9','v23']:
      X_tr, y_tr, X_va, y_va, X_te, y_te, norm_params,column_list = load_data(ts,LME_dates,self.horizon,self.gt,self.lag,split_date,norm_params,tech_params,version_params)
      column_lag_list = []
      column_name = []
      for i in range(lag):
        for item in column_list[0]:
          new_item = item+"_"+str(lag-i)
          column_lag_list.append(new_item)
      X_tr = np.concatenate(X_tr)
      final_X_tr = X_tr.reshape(len(X_tr),lag*len(column_list[0]))
      final_y_tr = np.concatenate(y_tr)
      X_va = np.concatenate(X_va)
      final_y_va = np.concatenate(y_va)
      final_X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
      
    else:
      i = 0
      #iterate over different ground truths for data loading
      for ground_truth in ['LME_Co_Spot','LME_Al_Spot','LME_Ni_Spot','LME_Ti_Spot','LME_Zi_Spot','LME_Le_Spot']:
        print(ground_truth)
        metal_id = [0,0,0,0,0,0]
        metal_id[i] = 1
        
        #load data
        X_tr, y_tr, X_va, y_va, X_te, y_te, norm_check,column_list = load_data(copy(ts),LME_dates,horizon,[ground_truth],lag,copy(split_date),norm_params,tech_params,version_params)
        
        #post load process and metal id extension
        X_tr = np.concatenate(X_tr)
        X_tr = X_tr.reshape(len(X_tr),lag*len(column_list[0]))
        X_tr = np.append(X_tr,[metal_id]*len(X_tr),axis = 1)
        y_tr = np.concatenate(y_tr)
        X_va = np.concatenate(X_va)
        X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
        X_va = np.append(X_va,[metal_id]*len(X_va),axis = 1)
        y_va = np.concatenate(y_va)
        final_X_tr.append(X_tr)
        final_y_tr.append(y_tr)
        final_X_va.append(X_va)
        final_y_va.append(y_va)
        i+=1
      
      #sort by time, not by metal
      final_X_tr = [np.transpose(arr) for arr in np.dstack(final_X_tr)]
      final_y_tr = [np.transpose(arr) for arr in np.dstack(final_y_tr)]
      final_X_va = [np.transpose(arr) for arr in np.dstack(final_X_va)]
      final_y_va = [np.transpose(arr) for arr in np.dstack(final_y_va)]
      final_X_tr = np.reshape(final_X_tr,[np.shape(final_X_tr)[0]*np.shape(final_X_tr)[1],np.shape(final_X_tr)[2]])
      final_y_tr = np.reshape(final_y_tr,[np.shape(final_y_tr)[0]*np.shape(final_y_tr)[1],np.shape(final_y_tr)[2]])
      final_X_va = np.reshape(final_X_va,[np.shape(final_X_va)[0]*np.shape(final_X_va)[1],np.shape(final_X_va)[2]])
      final_y_va = np.reshape(final_y_va,[np.shape(final_y_va)[0]*np.shape(final_y_va)[1],np.shape(final_y_va)[2]])
      
      column_lag_list = []
      column_name = []
      for i in range(lag):
        for item in column_list[0]:
          new_item = item+"_"+str(lag-i)
          column_lag_list.append(new_item)
      column_lag_list.append("Co")
      column_lag_list.append("Al")
      column_lag_list.append("Ni")
      column_lag_list.append("Ti")
      column_lag_list.append("Zi")
      column_lag_list.append("Le")
    train_dataframe = pd.DataFrame(final_X_tr,columns=column_lag_list)
    train_X = train_dataframe.loc[:,column_lag_list]
    train_y = pd.DataFrame(final_y_tr,columns=['result'])
    
    test_dataframe = pd.DataFrame(final_X_va,columns=column_lag_list)
    test_X = test_dataframe.loc[:,column_lag_list] 
    n_splits=args.k_folds
    from sklearn.metrics import accuracy_score
    model = xgb.XGBClassifier(max_depth=max_depth,
          learning_rate = learning_rate,
          n_estimators=500,
          silent=True,
          nthread=10,
          gamma=gamma,
          min_child_weight=min_child_weight,
          subsample=subsample,
          colsample_bytree=0.7,
          colsample_bylevel=1,
          reg_alpha=0.0001,
          reg_lambda=1,
          scale_pos_weight=1,
          seed=1440,
          missing=None)
    folds = KFold(n_splits=n_splits)
    scores = []
    prediction = np.zeros((len(X_va), 1))
    folder_index = []
    """
    save the model
    """
    for fold_n, (train_index, valid_index) in enumerate(folds.split(train_X)):
      X_train, X_valid = train_X[column_lag_list].iloc[train_index], train_X[column_lag_list].iloc[valid_index]
      y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]
      model.fit(X_train, y_train,eval_metric='error',verbose=True,eval_set=[(X_valid,y_valid)],early_stopping_rounds=5)
      y_pred_valid = model.predict(X_valid)
      pickle.dump(model, open(split_date+"_"+self.gt+"_"+str(self.horizon)+"_"+str(lag)+"_"+str(fold_n)+"_"+'xgb.model', "wb"))
      #bst.save_model(split_date+"_"+self.gt+"_"+str(self.horizon)+"_"+str(lag)+"_"+'xgb.model')


  #-------------------------------------------------------------------------------------------------------------------------------------#
  """
  this function is used to predict the date
  """
  def test(self):
    """
    split the date
    """
    #os.chdir(os.path.abspath(sys.path[0]))
    print("begin to test")
    pure_LogReg = LogReg(parameters={})

    #assert that the configuration path is correct
    if self.version in ['v5','v7']:
      assert (self.path == "exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf")
    elif self.version in ["v3","v23"]:
      assert (self.path == "exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf")
    elif self.version in ["v9","v10","v12"]:
      assert (self.path == "exp/online_v10.conf")

    #retrieve column list based on configuration path
    with open(os.path.join(os.getcwd(),self.path)) as fin:
      fname_columns = json.load(fin)
    temporary_df, not_used = read_data_NExT(fname_columns[0], "2003-11-12")
    temporary_df = pd.concat(temporary_df, axis = 1, sort = True)
    columns_to_be_stored = temporary_df.columns.values.tolist()

    """
    read the data from the 4E or NExT database
    """
    if self.source=='4E':
      from utils.read_data import read_data_v5_4E
      time_series, LME_dates = read_data_v5_4E("2003-11-12")

    elif self.source=='NExT':
      with open(os.path.join(os.getcwd(),self.path)) as fin:
        fname_columns = json.load(fin)
      data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
      time_series = pd.concat(data_list, axis = 1, sort = True)
    time_series = time_series[columns_to_be_stored]
    """
    begin to split the train data
    """
    today = self.date
    month = str(today).split("-")[1]
    year = str(today).split("-")[0]
    if month <="06":
      if month=="01":
        day = int(str(today).split("-")[2])
        if day <=self.horizon:
          print("the data is not enough for training")
          os.exit(0)
        else:
          start_year = int(year)-5
          start_time = str(start_year)+"-01-01"
          evalidate_year = int(year)
          evalidate_date = str(evalidate_year)+"-01-01"
          end_time = str(today)
      else:
        start_year = int(year)-5
        start_time = str(start_year)+"-01-01"
        evalidate_year = int(year)
        evalidate_date = str(evalidate_year)+"-01-01"
        end_time = str(today)
    else:
      if month=="07":
        day = int(str(today).split("-")[2])
        if day<=self.horizon:
          print("the data is not enough for training")
          os.exit(0)
        else:
          year = str(today).split("-")[0]
          start_year = int(year)-5
          end_year = int(year)
          start_time = str(start_year)+"-07-01"
          evalidate_year = int(year)
          evalidate_date = str(evalidate_year)+"-07-01"
          end_time = str(today)
      else:
        year = str(today).split("-")[0]
        start_year = int(year)-5
        end_year = int(year)
        start_time = str(start_year)+"-07-01"
        evalidate_year = int(year)
        evalidate_date = str(evalidate_year)+"-07-01"
        end_time = str(today)
    split_dates  =  [start_time,evalidate_date,str(today)]
    
    model = pure_LogReg.load(self.version, self.gt, self.horizon, self.lag,evalidate_date)

    """
    generate the version
    """
    version_params=generate_version_params(self.version)

    norm_volume = "v1"
    norm_3m_spread = "v1"
    norm_ex = "v1"
    len_ma = 5
    len_update = 30
    tol = 1e-7
    norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
            'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
    tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                    'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2,"live":None}
    ts = copy(time_series.loc[split_dates[0]:split_dates[2]])
    date_list = time_series.loc[split_dates[1]:split_dates[2]].index.values.tolist()
    X_tr, y_tr, X_va, y_va, X_te, y_te, norm_params,column_list,val_dates = load_data(ts,LME_dates,self.horizon,[self.gt],self.lag,copy(split_dates),norm_params,tech_params,version_params)
    column_lag_list = []
    column_name = []
    for i in range(lag):
      for item in column_list[0]:
        new_item = item+"_"+str(lag-i)
        column_lag_list.append(new_item)
    if self.version not in ['v3','v5','v7','v9','v23']:
      metal_id = [0,0,0,0,0,0]
      j = 0
      for ground_truth in ['LME_Co_Spot','LME_Al_Spot','LME_Ni_Spot','LME_Ti_Spot','LME_Zi_Spot','LME_Le_Spot']:
        if self.gt == ground_truth:
          metal_id[j] = 1
        j+=1
      column_lag_list.append("Co")
      column_lag_list.append("Al")
      column_lag_list.append("Ni")
      column_lag_list.append("Ti")
      column_lag_list.append("Zi")
      column_lag_list.append("Le")

    X_tr = np.concatenate(X_tr)
    X_tr = X_tr.reshape(len(X_tr),lag*len(column_list[0]))
    train_dataframe = pd.DataFrame(X_tr,columns=column_lag_list)
    train_X = train_dataframe.loc[:,column_lag_list]
    y_tr = np.concatenate(y_tr)
    train_y = pd.DataFrame(y_tr,columns=['result'])
    X_va = np.concatenate(X_va)
    y_va = np.concatenate(y_va)
    X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
    test_dataframe = pd.DataFrame(X_va,columns=column_lag_list)
    test_X = test_dataframe.loc[:,column_lag_list] 
    n_splits=args.k_folds
    from sklearn.metrics import accuracy_score
    model = xgb.XGBClassifier(max_depth=max_depth,
          learning_rate = learning_rate,
          n_estimators=500,
          silent=True,
          nthread=10,
          gamma=gamma,
          min_child_weight=min_child_weight,
          subsample=subsample,
          colsample_bytree=0.7,
          colsample_bylevel=1,
          reg_alpha=0.0001,
          reg_lambda=1,
          scale_pos_weight=1,
          seed=1440,
          missing=None)
    folds = KFold(n_splits=n_splits)
    scores = []
    prediction = np.zeros((len(X_va), 1))
    folder_index = []
    """
    save the model
    """
    for fold_n, (train_index, valid_index) in enumerate(folds.split(train_X)):
      pickle.load(model, open(split_date+"_"+self.gt+"_"+str(self.horizon)+"_"+str(lag)+"_"+str(fold_n)+"_"+'xgb.model', "rb"))
      y_pred = model.predict_proba(test_X, ntree_limit=model.best_ntree_limit)[:, 1]
      y_pred = y_pred.reshape(-1, 1)
      if fold_n == 0:
        folder_1=y_pred
        folder_1=folder_1.reshape(len(folder_1),1)
      elif fold_n == 1:    
        folder_2=y_pred
        folder_2=folder_2.reshape(len(folder_2),1)
      elif fold_n==2:    
        folder_3 = y_pred
        folder_3=folder_3.reshape(len(folder_3),1)
      elif fold_n==3:
        folder_4 = y_pred
        folder_4=folder_4.reshape(len(folder_4),1)
      elif fold_n==4:
        folder_5=y_pred
        folder_5=folder_5.reshape(len(folder_5),1)
      elif fold_n==5:
        folder_6=y_pred
        folder_6=folder_6.reshape(len(folder_6),1)
      elif fold_n==6:
        folder_7=y_pred
        folder_7=folder_7.reshape(len(folder_7),1)
      elif fold_n==7:
        folder_8=y_pred
        folder_8=folder_8.reshape(len(folder_8),1)
      elif fold_n==8:
        folder_9=y_pred
        folder_9=folder_9.reshape(len(folder_9),1)
      elif fold_n==9:
        folder_10=y_pred
        folder_10=folder_10.reshape(len(folder_10),1) 
        #calculate the all folder voting
    result = np.concatenate((folder_1,folder_2,folder_3,folder_4,folder_5,folder_6,folder_7,folder_8,folder_9,folder_10),axis=1)
    final_list = []
    for j in range(len(result)):
      count_1=0
      count_0=0
      for item in result[j]:
        if item > 0.5:
          count_1+=1
        else:
          count_0+=1
      if count_1>count_0:
        final_list.append(1)
      else:
        final_list.append(0)
    #print("the all folder voting precision is {}".format(metrics.accuracy_score(y_va, final_list)))
    return final_list
    result = np.concatenate((folder_6,folder_7,folder_8,folder_9,folder_10),axis=1)
    final_list = []
    for j in range(len(result)):
      count_1=0
      count_0=0
      for item in result[j]:
        if item > 0.5:
          count_1+=1
        else:
          count_0+=1
      if count_1>count_0:
        final_list.append(1)
      else:
        final_list.append(0)
    #print("the near precision is {}".format(metrics.accuracy_score(y_va, final_list)))
    return final_list
    result = np.concatenate((folder_1,folder_2,folder_3,folder_4,folder_5),axis=1)
    final_list = []
    for j in range(len(result)):
      count_1=0
      count_0=0
      for item in result[j]:
        if item > 0.5:
          count_1+=1
        else:
          count_0+=1
      if count_1>count_0:
        final_list.append(1)
      else:
        final_list.append(0)
    #print("the far precision is {}".format(metrics.accuracy_score(y_va, final_list)))
    return final_list
    if split_date[1].split("-")[1]=='01':
      result = np.concatenate((folder_1,folder_3,folder_5,folder_7,folder_9),axis=1)
      final_list = []
      for j in range(len(result)):
        count_1=0
        count_0=0
        for item in result[j]:
          if item > 0.5:
            count_1+=1
          else:
            count_0+=1
        if count_1>count_0:
          final_list.append(1)
        else:
          final_list.append(0)
            #print("the lag is {}".format(lag))
      #print("the same precision is {}".format(metrics.accuracy_score(y_va, final_list)))
      return final_list
      result = np.concatenate((folder_2,folder_4,folder_6,folder_8,folder_10),axis=1)
      final_list = []
      for j in range(len(result)):
        count_1=0
        count_0=0
        for item in result[j]:
          if item > 0.5:
            count_1+=1
          else:
            count_0+=1
        if count_1>count_0:
          final_list.append(1)
        else:
          final_list.append(0)
      #print("the reverse precision is {}".format(metrics.accuracy_score(y_va, final_list)))
      return final_list
    else:
      result = np.concatenate((folder_2,folder_4,folder_6,folder_8,folder_10),axis=1)
      final_list = []
      for j in range(len(result)):
        count_1=0
        count_0=0
        for item in result[j]:
          if item > 0.5:
            count_1+=1
          else:
            count_0+=1
        if count_1>count_0:
          final_list.append(1)
        else:
          final_list.append(0)
      #print("the same precision is {}".format(metrics.accuracy_score(y_va, final_list)))
      return final_list
      result = np.concatenate((folder_1,folder_3,folder_5,folder_7,folder_9),axis=1)
      final_list = []
      for j in range(len(result)):
        count_1=0
        count_0=0
        for item in result[j]:
          if item > 0.5:
            count_1+=1
          else:
            count_0+=1
        if count_1>count_0:
          final_list.append(1)
        else:
          final_list.append(0)
      #print("the lag is {}".format(lag))
      #print("the reverse precision is {}".format(metrics.accuracy_score(y_va, final_list)))
      #bst.save_model(split_date+"_"+self.gt+"_"+str(self.horizon)+"_"+str(lag)+"_"+'xgb.model')
      return final_list







