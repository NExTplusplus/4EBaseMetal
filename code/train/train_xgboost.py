#encoding:utf-8
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],'..')))
import time
import argparse
import json
from data.load_data_v5 import load_data_v5 
import xgboost as xgb

if __name__ == '__main__':
    desc = 'the Xgboost model'
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
    parser.add_argument(
        '-v','--version', help='version', type = int, default = 5
    )
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "4E"
    )
    args = parser.parse_args()
    if args.ground_truth =='None':
        args.ground_truth = None

    tra_date = '2003-11-12'
    val_date = '2016-06-01'
    tes_date = '2016-12-23'
    split_dates = [tra_date, val_date, tes_date]


    os.chdir(os.path.abspath(sys.path[0]))

    # read data configure file
    with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)
    args.ground_truth = args.ground_truth.split(",")
    if args.action == 'train':
        comparison = None
        n = 0
        for f in fname_columns:
            horizon = args.steps
            for lag in [5,10, 20, 30]:
                for norm_volume in ["v1","v2"]:
                        n+=1
                        norm_3m_spread = "v1"
                        norm_ex = "v1"
                        len_ma = 5
                        len_update = 30
                        tol = 1e-7
                        norm_params = {'vol_norm':norm_volume, 'ex_spread_norm':norm_ex,'spot_spread_norm': norm_3m_spread, 
                                                'len_ma':len_ma, 'len_update':len_update, 'both':3,'strength':0.01
                                                }
                        tech_params = {'strength':0.01,'both':3}
                        X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params = load_data_v5(f, horizon, args.ground_truth, lag, 
                                                                                                args.source, split_dates, 
                                                                                                norm_params, tech_params)
                                     
                        X_tr = np.concatenate(X_tr)
                        X_tr=X_tr.reshape(len(X_tr),lag*123)
                        y_tr = np.concatenate(y_tr)
                        X_va = np.concatenate(X_va)
                        y_va = np.concatenate(y_va)
                        train = np.append(X_tr,y_tr,axis=1)
                        X_va=X_va.reshape(len(X_va),lag*123)
                        validation = np.append(X_va,y_va,axis=1)
                        train_X = train[:,:len(train[0])-2]
                        train_Y = train[:,len(train[0])-1]
                        validation_X = validation[:,:len(validation[0])-2]
                        validation_Y = validation[:,len(validation[0])-1]
                        # train the model and test the model
                        from sklearn.metrics import roc_auc_score
                        xlf = xgb.XGBClassifier(max_depth=4,
                            learning_rate=0.8,
                            n_estimators=500,
                            silent=True,
                            objective='binary:logistic',
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
                        xlf.fit(train_X, train_Y, eval_metric='error',verbose=True,eval_set=[(validation_X, validation_Y)], early_stopping_rounds = 10)
                        y_pred = xlf.predict(validation_X, ntree_limit=xlf.best_ntree_limit)

                        auc_score = roc_auc_score(validation_Y, y_pred)
                        print("accuracy is {}".format(auc_score))
                        print("lag is {}".format(lag))
                        print('norm_volume is {}'.format(norm_volume))
