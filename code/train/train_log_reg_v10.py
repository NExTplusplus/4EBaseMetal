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
            # initialize parameters for load data
            length = 5
            split_dates = rolling_half_year("2009-07-01","2017-01-01",length)
            split_dates  =  split_dates[:]
            importance_list = []
            version_params=generate_version_params(args.version)
            ans = {"C":[]}
            for split_date in split_dates:
                #print("the train date is {}".format(split_date[0]))
                #print("the test date is {}".format(split_date[1]))
                #generate parameters for load data
                horizon = args.steps
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
                ts = copy(time_series.loc[split_date[0]:split_date[2]])
                i = 0
                #iterate over ground truths
                for ground_truth in ['LME_Co_Spot','LME_Al_Spot','LME_Ni_Spot','LME_Ti_Spot','LME_Zi_Spot','LME_Le_Spot']:
                    print(ground_truth)
                    metal_id = [0,0,0,0,0,0]
                    metal_id[i] = 1
                    #load data
                    X_tr, y_tr, X_va, y_va, X_te, y_te, norm_check,column_list = load_data(copy(ts),LME_dates,horizon,[ground_truth],lag,split_date,norm_params,tech_params,version_params)
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

                #tune logistic regression hyper parameter
                for C in [0.00001,0.0001,0.001,0.01]:
                    if C not in ans['C']:
                        ans["C"].append(C)
                    if split_date[1] not in ans.keys():
                        ans[split_date[1]] = []
            
                    n+=1
                    
                    pure_LogReg = LogReg(parameters={})

                    max_iter = args.max_iter
                    parameters = {"penalty":"l2", "C":C, "solver":"lbfgs", "tol":tol,"max_iter":6*4*len(f)*max_iter, "verbose" : 0,"warm_start": False, "n_jobs": -1}
                    pure_LogReg.train(final_X_tr,final_y_tr.flatten(), parameters)
                    acc = pure_LogReg.test(final_X_va,final_y_va.flatten())
                    ans[split_date[1]].append(acc)
            print(ans)
            pd.DataFrame(ans).to_csv("_".join(["log_reg",str(args.lag),str(args.steps)+".csv"]))