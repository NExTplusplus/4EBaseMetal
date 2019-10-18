import os
import sys
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

if __name__ == '__main__':
    desc = 'the logistic regression model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--data_configure_file', '-c', type=str,
        help='configure file of the features to be read',
        default='exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
    )
    parser.add_argument('-s','--steps',type=int,default=5,
                        help='steps in the future to be predicted')
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                        type=str, default="LME_Co_Spot")
    parser.add_argument('-max_iter','--max_iter',type=int,default=100,
                        help='max number of iterations')
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "NExT"
    )
    parser.add_argument(
        '-mout', '--model_save_path', type=str, help='path to save model',
        default='../../exp/log_reg/model'
    )
    parser.add_argument(
        '-l','--lag', type=int, default = 5, help='lag'
    )
    parser.add_argument(
        '-k','--k_folds', type=int, default = 10, help='number of folds to conduct cross validation'
    )
    parser.add_argument(
        '-v','--version', help='version', type = str, default = 'v10'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    parser.add_argument('-xgb','--xgboost',type = int,help='if you want to train the xgboost you need to inform us of that',default=0)
    args = parser.parse_args()
    if args.ground_truth =='None':
        args.ground_truth = None
    os.chdir(os.path.abspath(sys.path[0]))
    # read data configure file
    with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)
    args.ground_truth = args.ground_truth.split(",")
    #print("args.ground_truth is {}".format(args.ground_truth))
    #import os
    #os._exit(0)
    '''
    if args.lag==5 and args.ground_truth[0]=='LME_Ti_Spot':
        gamma=0.8
        learning_rate=0.9
        max_depth=4
        subsample=0.9
    elif args.lag==5 and args.ground_truth[0]=='LME_Co_Spot':
        gamma=0.9
        learning_rate=0.7
        max_depth=5
        subsample=0.85
    elif args.lag==10 and args.ground_truth[0]=='LME_Ti_Spot':
        gamma=0.9
        learning_rate=0.9
        max_depth=4
        subsample=0.7
    elif args.lag==10 and args.ground_truth[0]=='LME_Co_Spot':
        gamma=0.8
        learning_rate=0.8
        max_depth=6
        subsample=0.9
    elif args.lag==20 and args.ground_truth[0]=='LME_Ti_Spot':
        gamma=0.7
        learning_rate=0.8
        max_depth=4
        subsample=0.7
    elif args.lag==20 and args.ground_truth[0]=='LME_Co_Spot':
        gamma=0.8
        learning_rate=0.7
        max_depth=4
        subsample=0.9
    elif args.lag==30 and args.ground_truth[0]=='LME_Ti_Spot':
        gamma=0.7
        learning_rate=0.8
        max_depth=4
        subsample=0.7
    elif args.lag==30 and args.ground_truth[0]=='LME_Co_Spot':
        gamma=0.7
        learning_rate=0.7
        max_depth=4
        subsample=0.9
    '''
    if args.action == 'train':
        comparison = None
        n = 0
        #iterate over list of configurations
        for f in fname_columns:
            lag = args.lag
            #read data
            if args.source == "NExT":
                from utils.read_data import read_data_NExT
                data_list, LME_dates = read_data_NExT(f, "2003-11-12")
                time_series = pd.concat(data_list, axis = 1, sort = True)
            elif args.source == "4E":
                from utils.read_data import read_data_v5_4E
                time_series, LME_dates = read_data_v5_4E("2003-11-12")
            #generate parameters for load data
            length = 5
            split_dates = rolling_half_year("2009-07-01","2017-01-01",length)
            split_dates  =  split_dates[:]
            importance_list = []
            version_params=generate_version_params(args.version)
            for split_date in split_dates:
                #print("the train date is {}".format(split_date[0]))
                #print("the test date is {}".format(split_date[1]))
                horizon = args.steps
                norm_volume = "v1"
                norm_3m_spread = "v1"
                norm_ex = "v1"
                len_ma = 5
                len_update = 30
                tol = 1e-7
                if args.xgboost==1:
                    print(args.xgboost)
                    norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                                'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':True}
                else:
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
                ts = copy(time_series.loc[split_date[0]:split_date[2]])
                i = 0
                #iterate over different ground truths
                for ground_truth in ['LME_Co_Spot','LME_Al_Spot','LME_Ni_Spot','LME_Ti_Spot','LME_Zi_Spot','LME_Le_Spot']:
                    print(ground_truth)
                    metal_id = [0,0,0,0,0,0]
                    metal_id[i] = 1
                    #load data
                    X_tr, y_tr, X_va, y_va, X_te, y_te, norm_check,column_list = load_data(copy(ts),LME_dates,horizon,[ground_truth],lag,split_date,norm_params,tech_params,version_params)
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
                
                #tune hyperparameter for xgboost
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
