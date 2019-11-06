import os
import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from data.load_data import load_data
from model.logistic_regression import LogReg
from utils.transform_data import flatten
from utils.construct_data import rolling_half_year
from utils.log_reg_functions import objective_function, loss_function
import warnings
import xgboost as xgb
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn import metrics
from sklearn.model_selection import KFold
from utils.version_control_functions import generate_version_params

"""
there are two functions below,
the test_one is for the v5 v7 v9 whose features are for the specific metal
the test_two is for the v10 v12 whose six metals features are reunited together 
"""
def test_one():
    action = "train"
    os.chdir(os.path.abspath(sys.path[0]))
    # read data configure file
    data_configure_file = 'exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
    with open(os.path.join(sys.path[0],data_configure_file)) as fin:
        fname_columns = json.load(fin)
    if action == 'train':
        comparison = None
        n = 0
        for f in fname_columns:
            # we only check the specific lag 5 feature
            lag = 5
            """
            we check the different version
            """
            for version in ["v5","v7","v9"]:
                from utils.read_data import read_data_NExT
                data_list, LME_dates = read_data_NExT(f, "2003-11-12")
                time_series = pd.concat(data_list, axis = 1, sort = True)
                length = 5
                split_dates = rolling_half_year("2009-07-01","2017-07-01",length)
                split_dates  =  split_dates[:]
                importance_list = []
                version_params=generate_version_params(version)
                """
                the test are for the different metal
                """
                for ground_truth in ["LME_Ti_Spot","LME_Co_Spot","LME_Al_Spot","LME_Zi_Spot","LME_Le_Spot","LME_Ni_Spot"]:
                    for horizon in [1,3,5]:
                        """
                        we load the accurate feature from the file
                        """
                        for s, split_date in enumerate(split_dates[:-1]):
                            if version=="v5":
                                true_data_x_tr = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_X_tr_"+ground_truth+str(horizon)+"_l5_"+"v5_"+split_date[1])
                                true_data_y_tr = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_y_tr_"+ground_truth+str(horizon)+"_l5_"+"v5_"+split_date[1])
                                true_data_x_va = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_X_va_"+ground_truth+str(horizon)+"_l5_"+"v5_"+split_date[1])
                                true_data_y_va = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_y_va_"+ground_truth+str(horizon)+"_l5_"+"v5_"+split_date[1])
                            elif version=="v7":
                                true_data_x_tr = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_X_tr_"+ground_truth+str(horizon)+"_l5_"+"v7_"+split_date[1])
                                true_data_y_tr = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_y_tr_"+ground_truth+str(horizon)+"_l5_"+"v7_"+split_date[1])
                                true_data_x_va = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_X_va_"+ground_truth+str(horizon)+"_l5_"+"v7_"+split_date[1])
                                true_data_y_va = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_y_va_"+ground_truth+str(horizon)+"_l5_"+"v7_"+split_date[1])
                            elif version=="v9":
                                true_data_x_tr = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_X_tr_"+ground_truth+str(horizon)+"_l5_"+"v9_"+split_date[1])
                                true_data_y_tr = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_y_tr_"+ground_truth+str(horizon)+"_l5_"+"v9_"+split_date[1])
                                true_data_x_va = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_X_va_"+ground_truth+str(horizon)+"_l5_"+"v9_"+split_date[1])
                                true_data_y_va = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_y_va_"+ground_truth+str(horizon)+"_l5_"+"v9_"+split_date[1])
                            
                            norm_volume = "v1"
                            norm_3m_spread = "v1"
                            norm_ex = "v1"
                            len_ma = 5
                            len_update = 30
                            tol = 1e-7
                            
                            if version=="v7":
                                norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                                            'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':True}
                            else:
                                norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                                            'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
                            
                            tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                                                            'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
                            
                            ts = copy(time_series.loc[split_date[0]:split_dates[s+1][2]])
                            
                            X_tr, y_tr, X_va, y_va, X_te, y_te, norm_params,column_list = load_data(ts,LME_dates,horizon,[ground_truth],lag,copy(split_date),norm_params,tech_params,version_params)
                           
                            X_tr = np.concatenate(X_tr)
                            X_tr = X_tr.reshape(len(X_tr),lag*len(column_list[0]))
                            y_tr = np.concatenate(y_tr)
                            y_tr = y_tr.reshape(1,len(X_tr))
                            #print(y_tr)
                            X_va = np.concatenate(X_va)
                            X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
                            y_va = np.concatenate(y_va)
                            y_va = y_va.reshape(1,len(X_va))
                            #X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
                            """
                            after loading the feature, we now check whether the accurate feature is same with the feature from the load data
                            """
                            print("version is {}".format(version))
                            print("the split date is {}".format(split_date))
                            print("the horizon is {}".format(horizon))
                            print("the metal is {}".format(ground_truth))
                            #print(X_tr == true_data_x_tr)
                            assert ((X_tr == true_data_x_tr).all())
                            assert ((y_tr == true_data_y_tr).all())
                            assert ((X_va == true_data_x_va).all())
                            assert ((y_va == true_data_y_va).all())

def test_two():
    action = "train"
    os.chdir(os.path.abspath(sys.path[0]))
    # read data configure file
    data_configure_file = 'exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
    with open(os.path.join(sys.path[0],data_configure_file)) as fin:
        fname_columns = json.load(fin)
    if action == 'train':
        comparison = None
        n = 0
        for f in fname_columns:
            # we only check the specific lag 5 feature
            lag = 5
            """
            we check the different version
            """
            for version in ["v10","v12"]:
                from utils.read_data import read_data_NExT
                data_list, LME_dates = read_data_NExT(f, "2003-11-12")
                time_series = pd.concat(data_list, axis = 1, sort = True)
                length = 5
                split_dates = rolling_half_year("2009-07-01","2017-07-01",length)
                split_dates  =  split_dates[:]
                importance_list = []
                version_params=generate_version_params(version)
                for horizon in [1,3,5]:
                        for s, split_date in enumerate(split_dates[:-1]):
                            """
                            we load the accurate feature from the file
                            """
                            if version=="v10":
                                true_data_x_tr = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_X_tr_"+"ALL"+str(horizon)+"_l5_"+"v10_"+split_date[1])
                                true_data_y_tr = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_y_tr_"+"ALL"+str(horizon)+"_l5_"+"v10_"+split_date[1])
                                true_data_x_va = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_X_va_"+"ALL"+str(horizon)+"_l5_"+"v10_"+split_date[1])
                                true_data_y_va = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_y_va_"+"ALL"+str(horizon)+"_l5_"+"v10_"+split_date[1])
                            elif version=="v12":
                                true_data_x_tr = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_X_tr_"+"ALL"+str(horizon)+"_l5_"+"v12_"+split_date[1])
                                true_data_y_tr = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_y_tr_"+"ALL"+str(horizon)+"_l5_"+"v12_"+split_date[1])
                                true_data_x_va = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_X_va_"+"ALL"+str(horizon)+"_l5_"+"v12_"+split_date[1])
                                true_data_y_va = np.loadtxt(os.sys.path[0]+"/code/feature_pytest/lagged_data/lagged_data_y_va_"+"ALL"+str(horizon)+"_l5_"+"v12_"+split_date[1])
                            norm_volume = "v1"
                            norm_3m_spread = "v1"
                            norm_ex = "v1"
                            len_ma = 5
                            len_update = 30
                            tol = 1e-7
                            norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                                        'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
                            final_X_tr = []
                            final_y_tr = []
                            final_X_va = []
                            final_y_va = []
                            final_X_te = []
                            final_y_te = [] 
                            tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                                                            'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
                            ts = copy(time_series.loc[split_date[0]:split_dates[s+1][2]])
                            i = 0
                            
                            #iterate over ground truths
                            for ground_truth in ['LME_Co_Spot','LME_Al_Spot','LME_Ni_Spot','LME_Ti_Spot','LME_Zi_Spot','LME_Le_Spot']:
                                print(ground_truth)
                                metal_id = [0,0,0,0,0,0]
                                metal_id[i] = 1
                                
                                #load data
                                X_tr, y_tr, X_va, y_va, X_te, y_te, norm_check,column_list = load_data(copy(ts),LME_dates,horizon,[ground_truth],lag,copy(split_date),norm_params,tech_params,version_params)
                                
                                #post load processing and metal id extension
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
                            """
                            after loading the feature, we now check whether the accurate feature is same with the feature from the load data
                            """
                            print("version is {}".format(version))
                            print("the split date is {}".format(split_date))
                            print("the horizon is {}".format(horizon))

                            assert ((final_X_tr == true_data_x_tr).all())
                            assert ((final_y_tr == true_data_y_tr).all())
                            assert ((final_X_va == true_data_x_va).all())
                            assert ((final_y_va == true_data_y_va).all())




