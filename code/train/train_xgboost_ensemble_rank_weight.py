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
import math
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
        '-sou','--source', help='source of data', type = str, default = "4E"
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
        '-v','--version', help='version', type = str, default = 'v5'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    parser.add_argument('-xgb','--xgboost',type = int,help='if you want to train the xgboost you need to inform us of that',default=1)
    parser.add_argument('-max_depth','--max_depth',type = int, help='feed the parameter into the model', default=0)
    parser.add_argument('-learning_rate','--learning_rate',type=float,help='feed the parameter into the model', default=0)
    parser.add_argument('-gamma','--gamma',type=float,help='feed the parameter into the model',default=0)
    parser.add_argument('-min_child','--min_child',type=int,help='feed the parameter into the model',default=0)
    parser.add_argument('-subsample','--subsample',type=float,help='feed the parameter into the model',default=0)
    parser.add_argument('-voting','--voting',type=str,help='there are five methods for voting: all,far,same,near,reverse',default='all')
    parser.add_argument('-window_size','--window_size',type=int,help='there is the size fo the window size',default=4)
    args = parser.parse_args()
    if args.ground_truth =='None':
        args.ground_truth = None
    os.chdir(os.path.abspath(sys.path[0]))
    # read data configure file
    with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)
    args.ground_truth = args.ground_truth.split(",")
    window_size = args.window_size
    length = 0
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
            split_dates = rolling_half_year("2009-07-01","2019-01-01",length)
            print("the length is {}".format(length))
            split_dates  =  split_dates[:]
            importance_list = []
            version_params=generate_version_params(args.version)
            result_v5_error = []
            result_v7_error = []
            result_v10_error = []
            result_lr_error = []
            length=0
            for split_date in split_dates:
                horizon = args.steps
                norm_volume = "v1"
                norm_3m_spread = "v1"
                norm_ex = "v1"
                len_ma = 5
                len_update = 30
                tol = 1e-7
                # detect whether you want to add another three time features to model
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
                print("the y_va length is {}".format(len(y_va)))
                X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
                test_dataframe = pd.DataFrame(X_va,columns=column_lag_list)
                test_X = test_dataframe.loc[:,column_lag_list]
                # load the XGBoost V5 feature 10 folder probability
                result_v5 = np.loadtxt(args.ground_truth[0]+"_horizon_"+str(horizon)+"_"+split_date[1]+"_"+"v5"+"_weight_4"+".txt")
                # load the XGBoost V7 feature 10 folder probability
                result_v7 = np.loadtxt(args.ground_truth[0]+"_horizon_"+str(horizon)+"_"+split_date[1]+"_"+"v7"+"_weight"+".txt")
                #load the XGBoost V10 feature 10 folder probability
                if args.ground_truth[0].split("_")[1]=="Co":
                    result_v10 = np.loadtxt("LMCADY"+"_"+"horizon_"+str(horizon)+"_"+split_date[1]+"_v10"+"_striplag30_weight"+".txt")
                elif args.ground_truth[0].split("_")[1]=="Al":
                    result_v10 = np.loadtxt("LMAHDY"+"_"+"horizon_"+str(horizon)+"_"+split_date[1]+"_v10"+"_striplag30_weight"+".txt")
                elif args.ground_truth[0].split("_")[1]=="Ni":
                    result_v10 = np.loadtxt("LMNIDY"+"_"+"horizon_"+str(horizon)+"_"+split_date[1]+"_v10"+"_striplag30_weight"+".txt")
                elif args.ground_truth[0].split("_")[1]=="Ti":
                    result_v10 = np.loadtxt("LMSNDY"+"_"+"horizon_"+str(horizon)+"_"+split_date[1]+"_v10"+"_striplag30_weight"+".txt")
                elif args.ground_truth[0].split("_")[1]=="Zi":
                    result_v10 = np.loadtxt("LMZSDY"+"_"+"horizon_"+str(horizon)+"_"+split_date[1]+"_v10"+"_striplag30_weight"+".txt")
                elif args.ground_truth[0].split("_")[1]=="Le":
                    result_v10 = np.loadtxt("LMPBDY"+"_"+"horizon_"+str(horizon)+"_"+split_date[1]+"_v10"+"_striplag30_weight"+".txt")
                # load the LR V5 feature classifier
                if args.ground_truth[0].split("_")[1]=="Co":
                    if split_date[1]>='2017-01-03':
                        LR_v5 = pd.read_csv('~/NEXT/LMCADY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+split_date[1]+".csv")
                    else:
                        LR_v5 = pd.read_csv('~/NEXT/LMCADY'+"_h"+str(horizon)+"_v5probh"+str(horizon)+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Al":
                    if split_date[1]>='2017-01-03':
                        LR_v5 = pd.read_csv('~/NEXT/LMAHDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+split_date[1]+".csv")
                    else:
                        LR_v5 = pd.read_csv('~/NEXT/LMAHDY'+"_h"+str(horizon)+"_v5probh"+str(horizon)+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Ni":
                    if split_date[1]>='2017-01-03':
                        LR_v5 = pd.read_csv('~/NEXT/LMNIDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+split_date[1]+".csv")
                    else:
                        LR_v5 = pd.read_csv('~/NEXT/LMNIDY'+"_h"+str(horizon)+"_v5probh"+str(horizon)+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Ti":
                    if split_date[1]>='2017-01-03':
                        LR_v5 = pd.read_csv('~/NEXT/LMSNDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+split_date[1]+".csv")
                    else:
                        LR_v5 = pd.read_csv('~/NEXT/LMSNDY'+"_h"+str(horizon)+"_v5probh"+str(horizon)+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Zi":
                    if split_date[1]>='2017-01-03':
                        LR_v5 = pd.read_csv('~/NEXT/LMZSDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+split_date[1]+".csv")
                    else:
                        LR_v5 = pd.read_csv('~/NEXT/LMZSDY'+"_h"+str(horizon)+"_v5probh"+str(horizon)+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Le":
                    if split_date[1]>='2017-01-03':
                        LR_v5 = pd.read_csv('~/NEXT/LMPBDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+split_date[1]+".csv")
                    else:
                        LR_v5 = pd.read_csv('~/NEXT/LMPBDY'+"_h"+str(horizon)+"_v5probh"+str(horizon)+split_date[1]+".csv")
                result_lr = list(LR_v5['Prediction'])
                # retrieve the probability result from the voting result
                final_list_v5 = []
                v5_voting_prob_list=[]
                for j in range(len(result_v5)):
                    count_1=0
                    count_0=0
                    pos_list = []
                    neg_list = []
                    for item in result_v5[j]:
                        if item > 0.5:
                            pos_list.append(item)
                            count_1+=1
                        else:
                            neg_list.append(item)
                            count_0+=1
                    if count_1>count_0:
                        v5_voting_prob_list.append(pos_list)
                        final_list_v5.append(1)
                    else:
                        v5_voting_prob_list.append(neg_list)
                        final_list_v5.append(0)
                # retrieve the probability result from the voting result
                final_list_v10=[]
                v10_voting_prob_list=[]
                for j in range(len(result_v10)):
                    count_1=0
                    count_0=0
                    pos_list = []
                    neg_list = []
                    for item in result_v10[j]:
                        if item > 0.5:
                            pos_list.append(item)
                            count_1+=1
                        else:
                            neg_list.append(item)
                            count_0+=1
                    if count_1>count_0:
                        v10_voting_prob_list.append(pos_list)
                        final_list_v10.append(1)
                    else:
                        v10_voting_prob_list.append(neg_list)
                        final_list_v10.append(0)
                # retrieve the probability result from the voting result
                final_list_v7=[]
                v7_voting_prob_list=[]
                for j in range(len(result_v7)):
                    count_1=0
                    count_0=0
                    pos_list = []
                    neg_list = []
                    for item in result_v7[j]:
                        if item > 0.5:
                            pos_list.append(item)
                            count_1+=1
                        else:
                            neg_list.append(item)
                            count_0+=1
                    if count_1>count_0:
                        v7_voting_prob_list.append(pos_list)
                        final_list_v7.append(1)
                    else:
                        v7_voting_prob_list.append(neg_list)
                        final_list_v7.append(0)

                # calculate the precision weight of the model
                if len(result_v5_error)==0:
                    for i in range(len(result_v5)):
                        count_1=0
                        count_0=0
                        for item in result_v5[i]:
                            if item>0.5:
                                count_1+=1
                            else:
                                count_0+=1
                        if count_1>count_0:
                            result=1
                        else:
                            result=0
                        if y_va[i]!=result:
                            result_v5_error.append(1)
                        else:
                            result_v5_error.append(0)
                        count_1=0
                        count_0=0
                        for item in result_v7[i]:
                            if item > 0.5:
                                count_1+=1
                            else:
                                count_0+=1
                        if count_1>count_0:
                            result=1
                        else:
                            result=0
                        if y_va[i]!=result:
                            result_v7_error.append(1)
                        else:
                            result_v7_error.append(0)
                        count_1=0
                        count_0=0
                        for item in result_v10[i]:
                            if item > 0.5:
                                count_1+=1
                            else:
                                count_0+=1
                        if count_1>count_0:
                            result=1
                        else:
                            result=0
                        if y_va[i]!=result:
                            result_v10_error.append(1)
                        else:
                            result_v10_error.append(0)
                        for item in result_lr:
                            if item > 0.5:
                                result=1
                            else:
                                result=0
                        if y_va[i]!=result:
                            result_lr_error.append(1)
                        else:
                            result_lr_error.append(0)
                    final_list = []
                    true_result = []
                    # we choose a specific window size to calculate the precision weight to ensemble the models results together
                    for i in range(window_size,len(y_va)):
                        true_result.append(y_va[i])
                        error_xgb_v5 = np.sum(result_v5_error[length:length+window_size])
                        error_xgb_v7 = np.sum(result_v7_error[length:length+window_size])
                        error_xgb_v10 = np.sum(result_v10_error[length:length+window_size])
                        error_lr = np.sum(result_lr_error[length:length+window_size])
                        weight_xgb_v5 = float(1/error_xgb_v5)/(1/error_xgb_v5+1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
                        weight_xgb_v7 = float(1/error_xgb_v7)/(1/error_xgb_v5+1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
                        weight_xgb_v10 = float(1/error_xgb_v10)/(1/error_xgb_v5+1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
                        weight_lr = float(1/error_lr)/(1/error_xgb_v5+1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
                        result=0
                        v5_item=1
                        v7_item=1
                        v10_item=1
                        for item in v5_voting_prob_list[i]:
                            v5_item*=item
                        v5_item = math.pow(v5_item,1/len(v5_voting_prob_list[i]))
                        for item in v7_voting_prob_list[i]:
                            v7_item*=item
                        v7_item = math.pow(v7_item,1/len(v7_voting_prob_list[i]))
                        for item in v10_voting_prob_list[i]:
                            v10_item*=item
                        v10_item = math.pow(v10_item,1/len(v10_voting_prob_list[i]))
                        result+=weight_xgb_v5*v5_item
                        result+=weight_xgb_v7*v7_item
                        result+=weight_xgb_v10*v10_item
                        result+=weight_lr*result_lr[i]
                        # detect whether the result is 1 or 0
                        if result>0.5:
                            final_list.append(1)
                        else:
                            final_list.append(0)
                        length+=1
                    print("the weight ensebmle for V5 V10 LR rank rank precision is {}".format(metrics.accuracy_score(true_result, final_list)))
                    print("the horizon is {}".format(horizon))
                    print("the lag is {}".format(lag))
                    print("the train date is {}".format(split_date[0]))
                    print("the test date is {}".format(split_date[1]))
                else:
                    # the same as above
                    for i in range(len(result_v5)):
                        count_1=0
                        count_0=0
                        for item in result_v5[i]:
                            if item>0.5:
                                count_1+=1
                            else:
                                count_0+=1
                        if count_1>count_0:
                            result=1
                        else:
                            result=0
                        if y_va[i]!=result:
                            result_v5_error.append(1)
                        else:
                            result_v5_error.append(0)
                        count_1=0
                        count_0=0
                        for item in result_v7[i]:
                            if item > 0.5:
                                count_1+=1
                            else:
                                count_0+=1
                        if count_1>count_0:
                            result=1
                        else:
                            result=0
                        if y_va[i]!=result:
                            result_v7_error.append(1)
                        else:
                            result_v7_error.append(0)
                        count_1=0
                        count_0=0
                        for item in result_v10[i]:
                            if item > 0.5:
                                count_1+=1
                            else:
                                count_0+=1
                        if count_1>count_0:
                            result=1
                        else:
                            result=0
                        if y_va[i]!=result:
                            result_v10_error.append(1)
                        else:
                            result_v10_error.append(0)
                        for item in result_lr:
                            if item > 0.5:
                                result=1
                            else:
                                result=0
                        if y_va[i]!=result:
                            result_lr_error.append(1)
                        else:
                            result_lr_error.append(0)
                    final_list = []
                    # the same as above
                    for i in range(len(y_va)):
                        error_xgb_v5 = np.sum(result_v5_error[length:length+window_size])
                        error_xgb_v7 = np.sum(result_v7_error[length:length+window_size])
                        error_xgb_v10 = np.sum(result_v10_error[length:length+window_size])
                        error_lr = np.sum(result_lr_error[length:length+window_size])
                        weight_xgb_v5 = float(1/error_xgb_v5)/(1/error_xgb_v5+1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
                        weight_xgb_v7 = float(1/error_xgb_v7)/(1/error_xgb_v5+1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
                        weight_xgb_v10 = float(1/error_xgb_v10)/(1/error_xgb_v5+1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
                        weight_lr = float(1/error_lr)/(1/error_xgb_v5+1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
                        result=0
                        v5_item=1
                        v7_item=1
                        v10_item=1
                        for item in v5_voting_prob_list[i]:
                            v5_item*=item
                        v5_item = math.pow(v5_item,1/len(v5_voting_prob_list[i]))
                        for item in v7_voting_prob_list[i]:
                            v7_item*=item
                        v7_item = math.pow(v7_item,1/len(v7_voting_prob_list[i]))
                        for item in v10_voting_prob_list[i]:
                            v10_item*=item
                        v10_item = math.pow(v10_item,1/len(v10_voting_prob_list[i]))
                        result+=weight_xgb_v5*v5_item
                        result+=weight_xgb_v7*v7_item
                        result+=weight_xgb_v10*v10_item
                        result+=weight_lr*result_lr[i]
                        if result>0.5:
                            final_list.append(1)
                        else:
                            final_list.append(0)
                        length+=1
                    print("the weight ensebmle for V5 V10 LR rank rank precision is {}".format(metrics.accuracy_score(y_va, final_list)))
                    print("the horizon is {}".format(horizon))
                    print("the lag is {}".format(lag))
                    print("the train date is {}".format(split_date[0]))
                    print("the test date is {}".format(split_date[1]))
