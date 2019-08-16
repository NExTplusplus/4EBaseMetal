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
                for i in range(10):
                    if i == 0:
                        folder_1=np.loadtxt(str(lag)+"_"+str(split_date[1])+"_"+args.ground_truth[0]+"_"+str(i)+".txt")
                        folder_1=folder_1.reshape(len(folder_1),1)
                    elif i == 1:
                        folder_2=np.loadtxt(str(lag)+"_"+str(split_date[1])+"_"+args.ground_truth[0]+"_"+str(i)+".txt")
                        folder_2=folder_2.reshape(len(folder_2),1)
                    elif i==2:
                        folder_3=np.loadtxt(str(lag)+"_"+str(split_date[1])+"_"+args.ground_truth[0]+"_"+str(i)+".txt")
                        folder_3=folder_3.reshape(len(folder_3),1)
                    elif i==3:
                        folder_4=np.loadtxt(str(lag)+"_"+str(split_date[1])+"_"+args.ground_truth[0]+"_"+str(i)+".txt")
                        folder_4=folder_4.reshape(len(folder_4),1)
                    elif i==4:
                        folder_5=np.loadtxt(str(lag)+"_"+str(split_date[1])+"_"+args.ground_truth[0]+"_"+str(i)+".txt")
                        folder_5=folder_5.reshape(len(folder_5),1)
                    elif i==5:
                        folder_6=np.loadtxt(str(lag)+"_"+str(split_date[1])+"_"+args.ground_truth[0]+"_"+str(i)+".txt")
                        folder_6=folder_6.reshape(len(folder_6),1)
                    elif i==6:
                        folder_7=np.loadtxt(str(lag)+"_"+str(split_date[1])+"_"+args.ground_truth[0]+"_"+str(i)+".txt")
                        folder_7=folder_7.reshape(len(folder_7),1)
                    elif i==7:
                        folder_8=np.loadtxt(str(lag)+"_"+str(split_date[1])+"_"+args.ground_truth[0]+"_"+str(i)+".txt")
                        folder_8=folder_8.reshape(len(folder_8),1)
                    elif i==8:
                        folder_9=np.loadtxt(str(lag)+"_"+str(split_date[1])+"_"+args.ground_truth[0]+"_"+str(i)+".txt")
                        folder_9=folder_9.reshape(len(folder_9),1)
                    elif i==9:
                        folder_10=np.loadtxt(str(lag)+"_"+str(split_date[1])+"_"+args.ground_truth[0]+"_"+str(i)+".txt")
                        folder_10=folder_10.reshape(len(folder_10),1)
                # begin to ensemble voting
                
                #result = np.concatenate((folder_1,folder_2,folder_3,folder_4,folder_5,folder_6,folder_7,folder_8,folder_9,folder_10),axis=1)
                #if split_date[1].split('-')[1]=='01':
                #    result=np.concatenate((folder_2,folder_4,folder_6,folder_8,folder_10),axis=1)
                #else:
                #   result=np.concatenate((folder_1,folder_3,folder_5,folder_7,folder_9),axis=1)
                result=np.concatenate((folder_6,folder_7,folder_8,folder_9,folder_10),axis=1)
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
                print("the precision is {}".format(metrics.accuracy_score(y_va, final_list)))
                # each folder to calculate
                #if split_date[1].split('-')[1]=='07':
                #   result = np.concatenate((folder_2,folder_4,folder_6,folder_8,folder_10),axis=1)



