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
from utils.general_functions import *
import warnings
import xgboost as xgb
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn import metrics
from sklearn.model_selection import KFold
from utils.version_control_functions import generate_version_params
from sklearn.externals import joblib
import pickle
import datetime
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
    assert_version(self.version,self.path)

    #read the data from the 4E or NExT database
    time_series,LME_dates,config_length = read_data_with_specified_columns(self.source,self.path,"2003-11-12")

    #generate list of list of dates to be used to roll over 5 half years
    today = self.date
    length = 5
    if even_version(self.version) and self.horizon > 5:
      length = 4
    start_time,end_time = get_relevant_dates(today,length,"tune")
    split_dates = rolling_half_year(start_time,end_time,length)
    split_dates  =  split_dates[:]
    
    #generate the version
    version_params=generate_version_params(self.version)


    for s, split_date in enumerate(split_dates[:-1]):

      print("the train date is {}".format(split_date[0]))
      print("the test date is {}".format(split_date[1]))

      #toggle metal id
      metal_id = False
      ground_truth_list = [self.gt]
      if even_version(self.version):
        metal_id = True
        ground_truth_list = ["LME_Co_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot","LME_Le_Spot"]

      #extract copy of data to process
      ts = copy(time_series.loc[split_date[0]:split_dates[s+1][2]])

      #load data for use
      final_X_tr, final_y_tr, final_X_va, final_y_va, val_dates, column_lag_list = prepare_data(ts,LME_dates,self.horizon,ground_truth_list,self.lag,copy(split_date),version_params,metal_id_bool = metal_id)
        
      train_dataframe = pd.DataFrame(final_X_tr,columns=column_lag_list)
      train_X = train_dataframe.loc[:,column_lag_list]
      train_y = pd.DataFrame(final_y_tr,columns=['result'])
      
      test_dataframe = pd.DataFrame(final_X_va,columns=column_lag_list)
      test_X = test_dataframe.loc[:,column_lag_list] 
      n_splits=10
      """
      tune xgboost hyper parameter
      """

      for max_depth in [3,4,5]:
        for learning_rate in [0.6,0.7,0.8,0.9]:
          for gamma in [0.6,0.7,0.8,0.9]:
            for min_child_weight in [3,4,5,6]:
              for subsample in [0.6,0.7,0.85,0.9]:
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
                prediction = np.zeros((len(final_X_va), 1))
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
      print("the lag is {}".format(self.lag))
      print("the train date is {}".format(split_date[0]))
      print("the test date is {}".format(split_date[1]))


  #-------------------------------------------------------------------------------------------------------------------------------------#
  """
  this function is used to train the model and save it
  """
  def train(self,max_depth,learning_rate,gamma,min_child_weight,subsample):
    print("begin to train")

    #assert that the configuration path is correct
    assert_version(self.version,self.path)

    #read the data from the 4E or NExT database
    time_series,LME_dates,config_length = read_data_with_specified_columns(self.source,self.path,"2003-11-12")

    #generate list of dates for today's model training period
    today = self.date
    length = 5
    if even_version(self.version) and self.horizon > 5:
      length = 4
    start_time,evalidate_date = get_relevant_dates(today,length,"train")
    split_dates  =  [start_time,evalidate_date,str(today)]

    #generate the version
    version_params=generate_version_params(self.version)

    print("the train date is {}".format(split_dates[0]))
    print("the test date is {}".format(split_dates[1]))

    #toggle metal id
    metal_id = False
    ground_truth_list = [self.gt]
    if even_version(self.version):
      metal_id = True
      ground_truth_list = ["LME_Co_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot","LME_Le_Spot"]

  
    time_series = time_series.loc[split_dates[0]:split_dates[2]]

    assert_labels(LME_dates,split_dates,self.horizon)

    #load data for use
    final_X_tr, final_y_tr, final_X_va, final_y_va, val_dates, column_lag_list = prepare_data(time_series,LME_dates,self.horizon,ground_truth_list,self.lag,copy(split_dates),version_params,metal_id_bool = metal_id)

    train_dataframe = pd.DataFrame(final_X_tr,columns=column_lag_list)
    train_X = train_dataframe.loc[:,column_lag_list]
    train_y = pd.DataFrame(final_y_tr,columns=['result'])
    
    test_dataframe = pd.DataFrame(final_X_va,columns=column_lag_list)
    test_X = test_dataframe.loc[:,column_lag_list] 
    n_splits=10
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
    prediction = np.zeros((len(final_X_va), 1))
    folder_index = []
    """
    save the model
    """
    for fold_n, (train_index, valid_index) in enumerate(folds.split(train_X)):
      X_train, X_valid = train_X[column_lag_list].iloc[train_index], train_X[column_lag_list].iloc[valid_index]
      y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]
      model.fit(X_train, y_train,eval_metric='error',verbose=True,eval_set=[(X_valid,y_valid)],early_stopping_rounds=5)
      y_pred_valid = model.predict(X_valid)
      pickle.dump(model, open(os.path.join('result','model','xgboost',split_dates[1]+"_"+self.gt+"_"+str(self.horizon)+"_"+str(self.lag)+"_"+str(fold_n)+"_"+self.version+"_"+'xgb.model'), "wb"))
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

    #assert that the configuration path is correct
    assert_version(self.version,self.path)

    #read the data from the 4E or NExT database
    time_series,LME_dates,config_length = read_data_with_specified_columns(self.source,self.path,"2003-11-12")

    #generate list of dates for today's model training period
    today = self.date
    length = 5
    if even_version(self.version) and self.horizon > 5:
      length = 4
    start_time,evalidate_date = get_relevant_dates(today,length,"test")
    split_dates  =  [start_time,evalidate_date,str(today)]
    

    #generate the version
    version_params=generate_version_params(self.version)

    metal_id = False
    if even_version(self.version):
      metal_id = True

    time_series = time_series.loc[split_dates[0]:split_dates[2]]

    assert_labels(LME_dates,split_dates,self.horizon)

    #load data for use
    final_X_tr, final_y_tr, final_X_va, final_y_va,val_dates, column_lag_list = prepare_data(time_series,LME_dates,self.horizon,[self.gt],self.lag,copy(split_dates),version_params,metal_id_bool = metal_id,live = True)

    train_dataframe = pd.DataFrame(final_X_tr,columns=column_lag_list)
    train_X = train_dataframe.loc[:,column_lag_list]
    train_y = pd.DataFrame(final_y_tr,columns=['result'])
    test_dataframe = pd.DataFrame(final_X_va,columns=column_lag_list)
    test_X = test_dataframe.loc[:,column_lag_list] 
    n_splits=10
    from sklearn.metrics import accuracy_score
    model = xgb.XGBClassifier(
          n_estimators=500,
          silent=True,
          nthread=10,
          colsample_bytree=0.7,
          colsample_bylevel=1,
          reg_alpha=0.0001,
          reg_lambda=1,
          scale_pos_weight=1,
          seed=1440,
          missing=None)
    folds = KFold(n_splits=n_splits)
    scores = []
    prediction = np.zeros((len(final_X_va), 1))
    folder_index = []
    """
    save the model
    """
    for fold_n, (train_index, valid_index) in enumerate(folds.split(train_X)):
      model = pickle.load(open(os.path.join("result","model","xgboost",split_dates[1]+"_"+self.gt+"_"+str(self.horizon)+"_"+str(self.lag)+"_"+str(fold_n)+"_"+self.version+"_"+'xgb.model'), "rb"))
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
    np.savetxt(os.path.join("result","probability","xgboost",self.gt+"_h"+str(self.horizon)+"_"+self.date+"_xgboost"+self.version+".txt"),result)
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
    print("the all folder voting precision is {}".format(metrics.accuracy_score(final_y_va, final_list)))
    final_list = pd.DataFrame(final_list,index = val_dates, columns = ["Prediction"])
    print(final_list)
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
    if split_dates[1].split("-")[1]=='01':
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







