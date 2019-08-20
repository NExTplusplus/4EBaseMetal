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
    parser.add_argument('-s','--steps',type=int,default=3,
                        help='steps in the future to be predicted')
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                        type=str, default="LME_Ti_Spot")
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
        '-v','--version', help='version', type = str, default = 'v7'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    parser.add_argument('-xgb','--xgboost',type = int,help='if you want to train the xgboost you need to inform us of that',default=1)
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
        subsample=0.9'''
    if args.action == 'train':
        comparison = None
        n = 0
        for f in fname_columns:
            lag = args.lag
            if args.source == "NExT":
                from utils.read_data import read_data_NExT
                data_list, LME_dates = read_data_NExT(f, "2003-11-12")
                time_series = pd.concat(data_list, axis = 1, sort = True)
            elif args.source == "4E":
                from utils.read_data import read_data_v5_4E
                time_series, LME_dates = read_data_v5_4E("2003-11-12")
            length = 5
            split_dates = rolling_half_year("2009-07-01","2017-01-01",length)
            split_dates  =  split_dates[-args.k_folds:]
            importance_list = []
            version_params=generate_version_params(args.version)
            for max_depth in [4,5,6]:
                    for learning_rate in [0.7,0.8,0.9]:
                        for gamma in [0.7,0.8,0.9]:
                            for min_child_weight in [3,4,5]:
                                for subsample in [0.7,0.85,0.9]:
                                    for split_date in split_dates:
                                        print("the train date is {}".format(split_date[0]))
                                        print("the test date is {}".format(split_date[1]))
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
                                        tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                                                                        'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
                                        ts = copy(time_series.loc[split_date[0]:split_date[2]])
                                        X_tr, y_tr, X_va, y_va, X_te, y_te, norm_params,column_list = load_data(ts,LME_dates,horizon,args.ground_truth,lag,split_date,norm_params,tech_params,version_params)
                                        column_lag_list = []
                                        column_name = []
                                        for i in range(lag):
                                            for item in column_list[0]:
                                                new_item = item+"_"+str(lag-i)
                                                column_lag_list.append(new_item)
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
                                        for fold_n, (train_index, valid_index) in enumerate(folds.split(train_X)):
                                            print("the train_index is {}".format(train_index))
                                            print("the valid_index is {}".format(valid_index))
                                            X_train, X_valid = train_X[column_lag_list].iloc[train_index], train_X[column_lag_list].iloc[valid_index]
                                            y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]
                                            model.fit(X_train, y_train,eval_metric='error',verbose=True,eval_set=[(X_valid,y_valid)],early_stopping_rounds=5)
                                            y_pred_valid = model.predict(X_valid)
                                            y_pred = model.predict_proba(test_X, ntree_limit=model.best_ntree_limit)[:, 1]
                                            y_pred = y_pred.reshape(-1, 1)
                                            prediction_each_folder = y_pred
                                            print("the max_depth is {}".format(max_depth))
                                            print("the learning_rate is {}".format(learning_rate))
                                            print("the gamma is {}".format(gamma))
                                            print("the min_child_weight is {}".format(min_child_weight))
                                            print("the subsample is {}".format(subsample))
                                            np.savetxt(str(lag)+"_"+str(split_date[1])+"_"+args.ground_truth[0]+"_"+str(fold_n)+"_"+str(max_depth)+"_"+str(learning_rate)+"_"+str(gamma)+"_"+str(min_child_weight)+"_"+str(subsample)+".txt", prediction_each_folder)




