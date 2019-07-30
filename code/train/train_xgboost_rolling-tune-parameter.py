import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from data.load_data_v5_rolling import load_data_v5_rolling
from model.logistic_regression import LogReg
from utils.transform_data import flatten
from utils.construct_data import rolling_half_year
from utils.log_reg_functions import objective_function, loss_function
import warnings
import xgboost as xgb
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
        '-v','--version', help='version', type = int, default = 1
    )
    parser.add_argument(
        '-mout', '--model_save_path', type=str, help='path to save model',
        default='../../exp/log_reg/model'
    )
    parser.add_argument(
        '-k','--k_folds', type=int, default = 5, help='number of folds to conduct cross validation'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
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
                if args.source =="NExT":
                    from utils.read_data import read_data_NExT
                    data_list, LME_dates = read_data_NExT(f, "2003-11-12")
                    time_series = pd.concat(data_list, axis = 1, sort = True)
                elif args.source == "4E":
                    from utils.read_data import read_data_v5_4E
                    time_series, LME_dates = read_data_v5_4E("2003-11-12")
                horizon = args.steps
                lag = 10
                norm_volume="v2"
                norm_ex = "v1"
                length = 5
                for max_depth in [4,5,6]:
                    for learning_rate in [0.7,0.8,0.9]:
                        for gamma in [0.7,0.8,0.9]:
                            for min_child_weight in [3,4,5]:
                                for subsample in [0.7,0.85,0.9]:
                                    split_dates = rolling_half_year("2003-01-01","2017-01-01",length)
                                    split_dates = split_dates[-args.k_folds:]
                                    for split_date in split_dates:
                                        norm_3m_spread = "v1"
                                        len_ma = 5
                                        len_update = 30
                                        tol = 1e-7
                                        norm_params = {'vol_norm':norm_volume, 'ex_spread_norm':norm_ex,'spot_spread_norm': norm_3m_spread, 
                                                        'len_ma':len_ma, 'len_update':len_update, 'both':3,'strength':0.01
                                                        }
                                        tech_params = {'strength':0.01,'both':3}
                                        ts = copy(time_series.loc[split_date[0]:split_date[2]])
                                        X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params = load_data_v5_rolling(ts, horizon, args.ground_truth, lag, 
                                                                                                        LME_dates, split_date, 
                                                                                                        norm_params, tech_params)

                                        for ind in range(len(X_tr)):
                                            X_tr[ind] = flatten(X_tr[ind])
                                            X_va[ind] = flatten(X_va[ind])

                                            if X_te is not None:
                                                X_te[ind] = flatten(X_te[ind])              
                                        X_tr = np.concatenate(X_tr)
                                        y_tr = np.concatenate(y_tr)
                                        X_va = np.concatenate(X_va)
                                        y_va = np.concatenate(y_va)
                                        X_tr=X_tr.reshape(len(X_tr),lag*123)
                                        train = np.append(X_tr,y_tr,axis=1)
                                        X_va=X_va.reshape(len(X_va),lag*123)
                                        validation = np.append(X_va,y_va,axis=1)
                                        train_X = train[:,:len(train[0])-2]
                                        train_Y = train[:,len(train[0])-1]
                                        validation_X = validation[:,:len(validation[0])-2]
                                        validation_Y = validation[:,len(validation[0])-1]
                                        # train the model and test the model
                                        from sklearn.metrics import accuracy_score
                                        xlf = xgb.XGBClassifier(max_depth=max_depth,
                                                     learning_rate=learning_rate,
                                                     n_estimators=500,
                                                     silent=True,
                                                     objective='binary:logistic',
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
                                        xlf.fit(train_X, train_Y, eval_metric='error',verbose=True,eval_set=[(validation_X, validation_Y)], early_stopping_rounds = 5)
                                        y_pred = xlf.predict(validation_X, ntree_limit=xlf.best_ntree_limit)
                                        auc_score = accuracy_score(validation_Y, y_pred)
                                        print("the max_depth is {}".format(max_depth))
                                        print("the learning_rate is {}".format(learning_rate))
                                        print("the gamma is {}".format(gamma))
                                        print("the min_child_weight is {}".format(min_child_weight))
                                        print("the subsample is {}".format(subsample))
                                        print("accuracy is {}".format(auc_score))
                                        n+=1
#                                    pure_LogReg = LogReg(parameters={})
#                                    max_iter = args.max_iter
#                                    parameters = {"penalty":"l2", "C":C, "solver":"lbfgs", "tol":tol,"max_iter":6*4*len(f)*max_iter, "verbose" : 0,"warm_start": False, "n_jobs": -1}
#                                    pure_LogReg.train(X_tr,y_tr.flatten(), parameters)
#                                    n_iter = pure_LogReg.n_iter()
#                            if norm_params["nVol"] is False:
#                                break  

        

                



