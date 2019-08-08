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
        '-k','--k_folds', type=int, default = 5, help='number of folds to conduct cross validation'
    )
    parser.add_argument(
        '-v','--version', help='version', type = str, default = 'v5'
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
            split_dates = rolling_half_year("2003-01-01","2017-01-01",length)
            split_dates  =  split_dates[-args.k_folds:]
            importance_list = []
            version_params=generate_version_params(args.version)
            for split_date in split_dates:
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
                #print("the date in data is {}".format(ts.index))
                X_tr, y_tr, X_va, y_va, X_te, y_te, norm_params,column_list = load_data(ts,LME_dates,horizon,args.ground_truth,lag,split_date,norm_params,tech_params,version_params)
                #print(column_list)
                #break
                #print("the X_tr is {}".format(len(X_tr)))
                #print("the column list is {}".format(column_list))
                #print("the length of column_list is {}".format(len(column_list[0])))
                print("the column_list length is {}".format(len(column_list[0])))
                column_lag_list = []
                column_name = []
                for i in range(lag):
                    for item in column_list[0]:
                        new_item = item+"_"+str(lag-i)
                        column_lag_list.append(new_item)
                #useful_column_list = []
                #for i in range(len(column_lag_list)):
                #	if 'LME' in column_lag_list[i]:
                #		useful_column_list.append(i)
                        #column_name.append(column_lag_list[i])
                        #print(column_lag_list[i])
                #	elif 'Co' in column_lag_list[i]:
                #		useful_column_list.append(i)
                        #column_name.append(column_lag_list[i])
                        #print(column_lag_list[i])
                #print("the length of the column_lag_list is {}".format(column_lag_list))
                #with open("column.txt","w") as f:
                    #for column in column_list[0]:
                        #f.write(column)
                        #f.write("\n")
                #print("the result is {}")
                        #column_lag_list.append(new_item)
                #column_dict = {}
                #for i in range(len(column_list)):
                    #column_dict[column_list[i]]=i
                #for item in column_list:
                #with open("All_div_"+"norm_volume_v1_lag_"+str(lag)+".json","r") as f:
                #	column_name = json.load(f)
                X_tr = np.concatenate(X_tr)
                X_tr = X_tr.reshape(len(X_tr),lag*len(column_list[0]))
                train_dataframe = pd.DataFrame(X_tr,columns=column_lag_list)
                train_X = train_dataframe.loc[:,column_lag_list]
                #train_data.corr().to_csv("All_div_"+"lag_"+str(lag)+"_correlation_"+split_date[2]+"_"+".csv")
                #print(train_data.corr())
                #X_tr=np.array(train_data).tolist()
                y_tr = np.concatenate(y_tr)
                train_y = pd.DataFrame(y_tr,columns=['result'])
                X_va = np.concatenate(X_va)
                y_va = np.concatenate(y_va)
                #train = np.append(X_tr,y_tr,axis = 1)
                X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
                test_dataframe = pd.DataFrame(X_va,columns=column_lag_list)
                test_X = test_dataframe.loc[:,column_lag_list]
                #test_y = pd.DataFrame(y_va,columns=['result'])
                #X_va = np.array(validation_data).tolist()
                #validation = np.append(X_va,y_va,axis=1)
                #train_X = train[:,:len(train[0])-2]
                #train_Y = train[:,len(train[0])-1]
                #validation_X = validation[:,:len(validation[0])-2]
                #validation_Y = validation[:,len(validation[0])-1]
                #print(train_X.index)
                #print(train_Y.index)
                n_splits=5
                from sklearn.metrics import accuracy_score
                model = xgb.XGBClassifier(max_depth=4,
                            learning_rate = 0.8,
                            n_estimators=500,
                            silent=True,
                            nthread=10,
                            gamma=0.8,
                            min_child_weight=3,
                            subsample=0.85,
                            colsample_bytree=0.7,
                            colsample_bylevel=1,
                            reg_alpha=0.0001,
                            reg_lambda=1,
                            scale_pos_weight=1,
                            seed=1440,
                            missing=None)
                folds = KFold(n_splits=5)
                scores = []
                prediction = np.zeros((len(X_va), 1))
                for fold_n, (train_index, valid_index) in enumerate(folds.split(train_X)):
                    print("the train_index is {}".format(train_index))
                    print("the valid_index is {}".format(valid_index))
                    X_train, X_valid = train_X[column_lag_list].iloc[train_index], train_X[column_lag_list].iloc[valid_index]
                    y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]
                    model.fit(X_train, y_train,eval_metric='error',verbose=True,eval_set=[(X_valid,y_valid)],early_stopping_rounds=5)
                    y_pred_valid = model.predict_proba(X_valid)[:, 1]
                    y_pred = model.predict_proba(test_X, ntree_limit=model.best_ntree_limit)[:, 1]
                    #oof[valid_index] = y_pred_valid.reshape(-1, 1)
                    scores.append(metrics.roc_auc_score(y_valid, y_pred_valid))
                    prediction += y_pred.reshape(-1, 1)
                prediction /= n_splits
                print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
                print('The test score is {}'.format(metrics.roc_auc_score(y_va, prediction)))
                    #importance_list.append(model.feature_importances_)
                    #print(model.feature_importances_)
                    #importance_list = model.feature_importances_
                    #for importance in importance_list:
                        #print(importance)
                    #print("the length of importance is {}".format(len(importance_list)))
                    #column_name_value_list = []
                    #for i in range(len(importance_list)):
                    #    if importance_list[i]!=0:
                    #        column_name_value_list.append((column_lag_list[i],str(importance_list[i])))
                    #with open("All_div_"+"norm_volume_v1_lag_"+str(lag)+".json","w") as f:
                    #	json.dump(column_name_value_list,f)
                    #importance_length = len(importance_list)/5
                    #pyplot.figure(lag)
                    #for i in range(lag):
                        #pyplot.subplot(lag,1,i+1)
                        #pyplot.bar(range(len(importance_list[125*i:125*(i+1)])),importance_list[125*i:125*(i+1)])
                    #pyplot.bar(range(len(model.feature_importances_)),model.feature_importances_)
                    #pyplot.show()
                    #plot_importance(model)
                    #pyplot.show()
                    #auc_score = accuracy_score(validation_Y,y_pred)
                    #print("the gamma is {}".format(gamma))
                    #print("the learning_rate is {}".format(learning_rate))
                    #print("the max_depth is {}".format(max_depth))
                    #print("the subsample is {}".format(subsample))
                    #print("the lag is {}".format(lag))
                    #print("the split_date is {}".format(split_date))
                    #print("accuracy is {}".format(auc_score))