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
    parser.add_argument('-max_iter','--max_iter',type=int,default=100,
                        help='max number of iterations')
    parser.add_argument(
        '-min', '--model_path', help='path to load model',
        type=str, default='../../exp/3d/Co/Xgboost/'
    )
    parser.add_argument(
        '-v','--version', help='version', type = int, default = 5
    )
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "4E"
    )
    parser.add_argument(
        '-mout', '--model_save_path', type=str, help='path to save model',
        default='../../exp/3d/Co/Xgboost/'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="exp/3d/Co/Xgboost/v5/result.txt")
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    args = parser.parse_args()
    if args.ground_truth =='None':
        args.ground_truth = None

    tra_date = '2003-11-12'
    val_date = '2016-06-01'
    tes_date = '2016-12-16'
    split_dates = [tra_date, val_date, tes_date]

    # read data configure file
    with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)
    args.ground_truth = args.ground_truth.split(",")
    if args.action == 'train':
            comparison = None
            n = 0
            for f in fname_columns:
                horizon = args.steps
                lag=20
                norm_volume = "v1"
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
                for max_depth in [4,5,6]:
                    for learning_rate in [0.7,0.8,0.9]:
                        for gamma in [0.7,0.8,0.9]:
                            for min_child_weight in [3,4,5]:
                                for subsample in [0.7,0.85,0.9]:
                                    # train the model and test the model
                                    from sklearn.metrics import roc_auc_score
                                    xlf = xgb.XGBClassifier(max_depth=max_depth,
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
                                    xlf.fit(train_X, train_Y, eval_metric='error',verbose=True,eval_set=[(validation_X, validation_Y)], early_stopping_rounds = 5)
                                    y_pred = xlf.predict(test_X, ntree_limit=xlf.best_ntree_limit)

                                    start_ind, end_ind = find_zoom("2018-01-01::2018-12-31",split_dates[0])
                                    auc_score = roc_auc_score(test_Y[start_ind:end_ind+1], y_pred[start_ind:end_ind+1])
                                    print("accuracy is {}".format(auc_score))
                                    print("max_depth is {}".format(max_depth))
                                    print("learning_rate is {}".format(learning_rate))
                                    print("gamma is {}".format(gamma))
                                    print('min_child_weight is {}'.format(min_child_weight))
                                    print('subsample is {}'.format(subsample))

