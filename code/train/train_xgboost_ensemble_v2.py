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
    parser.add_argument('-s','--steps',type=int,default=3,
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
            split_dates = rolling_half_year("2009-07-01","2019-01-01",length)
            split_dates  =  split_dates[-4:]
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
                
                # detect whether we want to add another three time features to model
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
                
                # load the XGBoost V5 feature 10 folder probability
                result_v5 = np.loadtxt(args.ground_truth[0]+"_horizon_"+str(horizon)+"_"+split_date[1]+"_v5"+".txt")
                
                # load the XGBoost V5 feature 10 folder probability
                final_list_v5 = []
                v5_voting_prob_list=[]
                
                # retrieve the probability result from the voting result
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
                
                #load the XGBoost V10 feature 10 folder probability
                if args.ground_truth[0].split("_")[1]=="Co":
                    result_v10 = np.loadtxt("LMCADY"+"_"+"horizon_"+str(horizon)+"_"+split_date[1]+"_v10"+"_striplag30"+".txt")
                elif args.ground_truth[0].split("_")[1]=="Al":
                    result_v10 = np.loadtxt("LMAHDY"+"_"+"horizon_"+str(horizon)+"_"+split_date[1]+"_v10"+"_striplag30"+".txt")
                elif args.ground_truth[0].split("_")[1]=="Ni":
                    result_v10 = np.loadtxt("LMNIDY"+"_"+"horizon_"+str(horizon)+"_"+split_date[1]+"_v10"+"_striplag30"+".txt")
                elif args.ground_truth[0].split("_")[1]=="Ti":
                    result_v10 = np.loadtxt("LMSNDY"+"_"+"horizon_"+str(horizon)+"_"+split_date[1]+"_v10"+"_striplag30"+".txt")
                elif args.ground_truth[0].split("_")[1]=="Zi":
                    result_v10 = np.loadtxt("LMZSDY"+"_"+"horizon_"+str(horizon)+"_"+split_date[1]+"_v10"+"_striplag30"+".txt")
                elif args.ground_truth[0].split("_")[1]=="Le":
                    result_v10 = np.loadtxt("LMPBDY"+"_"+"horizon_"+str(horizon)+"_"+split_date[1]+"_v10"+"_striplag30"+".txt")
                
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
                
                # load the XGBoost V7 feature 10 folder probability
                result_v7 = np.loadtxt(args.ground_truth[0]+"_horizon_"+str(horizon)+"_"+split_date[1]+"_v7"+".txt")
                
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
                
                # load the LR V5 feature classifier
                if args.ground_truth[0].split("_")[1]=="Co":
                    LR_v5 = pd.read_csv('~/NEXT/LMCADY'+"_h"+str(horizon)+"_v5res"+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Al":
                    LR_v5 = pd.read_csv('~/NEXT/LMAHDY'+"_h"+str(horizon)+"_v5resh"+str(horizon)+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Ni":
                    LR_v5 = pd.read_csv('~/NEXT/LMNIDY'+"_h"+str(horizon)+"_v5resh"+str(horizon)+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Ti":
                    LR_v5 = pd.read_csv('~/NEXT/LMSNDY'+"_h"+str(horizon)+"_v5resh"+str(horizon)+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Zi":
                    LR_v5 = pd.read_csv('~/NEXT/LMZSDY'+"_h"+str(horizon)+"_v5resh"+str(horizon)+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Le":
                    LR_v5 = pd.read_csv('~/NEXT/LMPBDY'+"_h"+str(horizon)+"_v5resh"+str(horizon)+split_date[1]+".csv")
                
                # load the LR V5 feature probability
                LR_v5_prediction_list = list(LR_v5['Prediction'])
                if args.ground_truth[0].split("_")[1]=="Co":
                    LR_v5_prob = pd.read_csv('~/NEXT/LMCADY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Al":
                    LR_v5_prob = pd.read_csv('~/NEXT/LMAHDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Ni":
                    LR_v5_prob = pd.read_csv('~/NEXT/LMNIDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Ti":
                    LR_v5_prob = pd.read_csv('~/NEXT/LMSNDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Zi":
                    LR_v5_prob = pd.read_csv('~/NEXT/LMZSDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+split_date[1]+".csv")
                elif args.ground_truth[0].split("_")[1]=="Le":
                    LR_v5_prob = pd.read_csv('~/NEXT/LMPBDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+split_date[1]+".csv")
                LR_v5_prediction_list = list(LR_v5['Prediction'])
                LR_v5_prob_prediction_list = list(LR_v5_prob['Prediction'])
                for i in range(len(LR_v5_prediction_list)):
                    if LR_v5_prediction_list[i]==-1.0:
                        LR_v5_prediction_list[i]=0.0
                    final_list=[]
                    for j in range(len(final_list_v5)):
                        if (final_list_v5[j]+final_list_v10[j]+LR_v5_prediction_list[j]+final_list_v7[j])>2:
                            final_list.append(1)
                        elif (final_list_v5[j]+final_list_v10[j]+LR_v5_prediction_list[j]+final_list_v7[j])<2:
                            final_list.append(0)
                        else:
                            #print("the voting result can not get the result")
                            final_list.append(0)
                print("the XGBOOST v5 test precision is {}".format(metrics.accuracy_score(y_va, final_list_v5)))
                print("the XGBOOST v10 test precision is {}".format(metrics.accuracy_score(y_va, final_list_v10)))
                print("the LR test precision is {}".format(metrics.accuracy_score(y_va, LR_v5_prediction_list)))
                #print("the ensemble for V5 V7 V10 LR voting precision is {}".format(metrics.accuracy_score(y_va, final_list)))
                
                # try the XGBoost V5 feature XGBoost V7 feature LR ensemble
                final_list=[]
                for j in range(len(final_list_v5)):
                    if (final_list_v5[j]+LR_v5_prediction_list[j]+final_list_v7[j])>=2:
                        final_list.append(1)
                    else:
                        final_list.append(0)
                v5_v7_lr = final_list
                print("the ensebmle for V5 V7 LR voting precision is {}".format(metrics.accuracy_score(y_va, final_list)))
                
                # try the XGBoost V7 feature XGBoost V10 feature LR ensemble
                final_list=[]
                for j in range(len(final_list_v5)):
                    if (final_list_v10[j]+LR_v5_prediction_list[j]+final_list_v7[j])>=2:
                        final_list.append(1)
                    else:
                        final_list.append(0)
                v7_v10_lr = final_list
                print("the ensebmle for V7 V10 LR voting precision is {}".format(metrics.accuracy_score(y_va, final_list)))
                
                # try the XGBoost V5 featur XGBoost V10 feature LR ensemble
                final_list=[]
                for j in range(len(final_list_v5)):
                    if (final_list_v10[j]+LR_v5_prediction_list[j]+final_list_v5[j])>=2:
                        final_list.append(1)
                    else:
                        final_list.append(0)
                v5_v10_lr = final_list
                print("the ensebmle for V5 V10 LR voting precision is {}".format(metrics.accuracy_score(y_va, final_list)))
                
                # try the XGBoost V5 feature XGBoost V7 feature XGBoost V10 feature together
                final_list = []
                for i in range(len(v5_v7_lr)):
                    if (v5_v7_lr[i]+v5_v10_lr[i]+v7_v10_lr[i])>=2:
                        final_list.append(1)
                    else:
                        final_list.append(0)
                print("the ensemble for V5 V7 V10 voting precision is {}".format(metrics.accuracy_score(y_va, final_list)))
                '''final_average_list = []
                for j in range(len(result_v7)):
                    result = 0
                    for item in result_v7[j]:
                        result+=item
                    for item in result_v5[j]:
                        result+=item
                    for item in result_v10[j]:
                        result+=item
                    result+=LR_v5_prob_prediction_list[j]
                    result = result/31.0
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 v10 LR average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(result_v7)):
                    result = 0
                    for item in result_v7[j]:
                        result+=item
                    for item in result_v5[j]:
                        result+=item
                    for item in result_v10[j]:
                        result+=item
                    result = result/30.0
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 v10 average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(result_v7)):
                    result = 0
                    for item in result_v7[j]:
                        result+=item
                    for item in result_v5[j]:
                        result+=item
                    result+=LR_v5_prob_prediction_list[j]
                    result = result/21.0
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 LR average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(result_v10)):
                    result = 0
                    for item in result_v10[j]:
                        result+=item
                    for item in result_v5[j]:
                        result+=item
                    result+=LR_v5_prob_prediction_list[j]
                    result = result/21.0
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v10 LR average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(result_v10)):
                    result = 0
                    for item in result_v10[j]:
                        result+=item
                    for item in result_v7[j]:
                        result+=item
                    result+=LR_v5_prob_prediction_list[j]
                    result = result/21.0
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v7 v10 LR average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(result_v7)):
                    result = 1
                    for item in result_v7[j]:
                        if item!=0:
                            result*=item
                        else:
                            result*=1
                    for item in result_v5[j]:
                        if item!=0:
                            result*=item
                        else:
                            result*=1
                    for item in result_v10[j]:
                        if item!=0:
                            result*=item
                        else:
                            result*=1
                    if LR_v5_prob_prediction_list[j]!=0:
                        result*=LR_v5_prob_prediction_list[j]
                    result = math.pow(result,1/31)
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 v10 LR rank average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(result_v7)):
                    result = 1
                    for item in result_v7[j]:
                        if item!=0:
                            result*=item
                        else:
                            result*=1
                    for item in result_v5[j]:
                        if item!=0:
                            result*=item
                        else:
                            result*=1
                    if LR_v5_prob_prediction_list[j]!=0:
                        result*=LR_v5_prob_prediction_list[j]
                    result = math.pow(result,1/21)
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 LR rank average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(result_v7)):
                    result = 1
                    for item in result_v7[j]:
                        if item!=0:
                            result*=item
                        else:
                            result*=1
                    for item in result_v10[j]:
                        if item!=0:
                            result*=item
                        else:
                            result*=1
                    if LR_v5_prob_prediction_list[j]!=0:
                        result*=LR_v5_prob_prediction_list[j]
                    result = math.pow(result,1/21)
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v7 v10 LR rank average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(result_v7)):
                    result = 1
                    for item in result_v5[j]:
                        if item!=0:
                            result*=item
                        else:
                            result*=1
                    for item in result_v10[j]:
                        if item!=0:
                            result*=item
                        else:
                            result*=1
                    if LR_v5_prob_prediction_list[j]!=0:
                        result*=LR_v5_prob_prediction_list[j]
                    result = math.pow(result,1/21)
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v10 LR rank average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))'''
                
                '''final_average_list = []
                for j in range(len(v7_voting_prob_list)):
                    result = 0
                    for item in v7_voting_prob_list[j]:
                        result+=item
                    for item in v5_voting_prob_list[j]:
                        result+=item
                    for item in v10_voting_prob_list[j]:
                        result+=item
                    result+=LR_v5_prob_prediction_list[j]
                    result = float(result)/float((len(v7_voting_prob_list[j])+len(v5_voting_prob_list[j])+len(v10_voting_prob_list[j])+1))
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 v10 LR average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(v7_voting_prob_list)):
                    result = 0
                    for item in v7_voting_prob_list[j]:
                        result+=item
                    for item in v5_voting_prob_list[j]:
                        result+=item
                    for item in v10_voting_prob_list[j]:
                        result+=item
                    result = float(result)/float((len(v7_voting_prob_list[j])+len(v5_voting_prob_list[j])+len(v10_voting_prob_list[j])))
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 v10 average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(v7_voting_prob_list)):
                    result = 0
                    for item in v7_voting_prob_list[j]:
                        result+=item
                    for item in v5_voting_prob_list[j]:
                        result+=item
                    result+=LR_v5_prob_prediction_list[j]
                    result = float(result)/float((len(v7_voting_prob_list[j])+len(v5_voting_prob_list[j])+1))
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 LR average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(v10_voting_prob_list)):
                    result = 0
                    for item in v10_voting_prob_list[j]:
                        result+=item
                    for item in v5_voting_prob_list[j]:
                        result+=item
                    result+=LR_v5_prob_prediction_list[j]
                    result = float(result)/float((len(v5_voting_prob_list[j])+len(v10_voting_prob_list[j])+1))
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v10 LR average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(v10_voting_prob_list)):
                    result = 0
                    for item in v10_voting_prob_list[j]:
                        result+=item
                    for item in v7_voting_prob_list[j]:
                        result+=item
                    result+=LR_v5_prob_prediction_list[j]
                    result = float(result)/float((len(v7_voting_prob_list[j])+len(v10_voting_prob_list[j])+1))
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v7 v10 LR average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(v7_voting_prob_list)):
                    result = 1
                    result*=np.mean(v7_voting_prob_list[j])
                    result*=np.mean(v10_voting_prob_list[j])
                    result*=np.mean(v5_voting_prob_list[j])
                    if LR_v5_prob_prediction_list[j]!=0:
                        result*=LR_v5_prob_prediction_list[j]
                    result = math.pow(result,1/4.0)
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 v10 LR rank average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(v7_voting_prob_list)):
                    result = 1
                    result*=np.mean(v7_voting_prob_list[j])
                    result*=np.mean(v5_voting_prob_list[j])
                    if LR_v5_prob_prediction_list[j]!=0:
                        result*=LR_v5_prob_prediction_list[j]
                    result = math.pow(result,1/3.0)
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 LR rank average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(v7_voting_prob_list)):
                    result = 1
                    result*=np.mean(v7_voting_prob_list[j])
                    result*=np.mean(v10_voting_prob_list[j])
                    if LR_v5_prob_prediction_list[j]!=0:
                        result*=LR_v5_prob_prediction_list[j]
                    result = math.pow(result,1/3.0)
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v7 v10 LR rank average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                final_average_list = []
                for j in range(len(v7_voting_prob_list)):
                    result = 1
                    result*=np.mean(v10_voting_prob_list[j])
                    result*=np.mean(v5_voting_prob_list[j])
                    if LR_v5_prob_prediction_list[j]!=0:
                        result*=LR_v5_prob_prediction_list[j]
                    result = math.pow(result,1/3.0)
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v10 LR rank average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))'''
                
                # try the XGBoost V5 feature XGBoost V7 feature XGBoost V10 feature LR ensemble together
                final_average_list = []
                for j in range(len(v7_voting_prob_list)):
                    result = 0
                    v7_item = 1
                    v5_item = 1
                    v10_item = 1
                    for item in v7_voting_prob_list[j]:
                        v7_item*=item
                    v7_item = math.pow(v7_item,1/len(v7_voting_prob_list[j]))
                    for item in v5_voting_prob_list[j]:
                        v5_item*=item
                    v5_item = math.pow(v5_item,1/len(v5_voting_prob_list[j]))
                    for item in v10_voting_prob_list[j]:
                        v10_item*=item
                    v10_item = math.pow(v10_item, 1/len(v10_voting_prob_list[j]))
                    result+=LR_v5_prob_prediction_list[j]
                    result = float(result)/4.0
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 v10 LR rank average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                
                # try the XGBoost V5 feature XGBoost V7 feature XGBoost V10 feature ensemble
                final_average_list = []
                for j in range(len(v7_voting_prob_list)):
                    result = 0
                    for item in v7_voting_prob_list[j]:
                        result+=item
                    for item in v5_voting_prob_list[j]:
                        result+=item
                    for item in v10_voting_prob_list[j]:
                        result+=item
                    result = float(result)/float((len(v7_voting_prob_list[j])+len(v5_voting_prob_list[j])+len(v10_voting_prob_list[j])))
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 v10 average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                
                # try the XGBoost V5 feature XGBoost V7 feature LR ensemble
                final_average_list = []
                for j in range(len(v7_voting_prob_list)):
                    result = 0
                    v7_item = 1
                    v5_item = 1
                    for item in v7_voting_prob_list[j]:
                        v7_item*=item
                    v7_item = math.pow(v7_item,1/len(v7_voting_prob_list[j]))
                    for item in v5_voting_prob_list[j]:
                        v5_item*=item
                    v5_item = math.pow(v5_item,1/len(v5_voting_prob_list[j]))
                    result+=LR_v5_prob_prediction_list[j]
                    result = float(result)/3.0
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 LR rank average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                
                # try the XGBoost V5 feature XGBoost V10 feature LR ensemble
                final_average_list = []
                for j in range(len(v10_voting_prob_list)):
                    result = 0
                    v5_item = 1
                    v10_item = 1
                    for item in v5_voting_prob_list[j]:
                        v5_item*=item
                    v5_item = math.pow(v5_item,1/len(v5_voting_prob_list[j]))
                    for item in v10_voting_prob_list[j]:
                        v10_item*=item
                    v10_item = math.pow(v10_item, 1/len(v10_voting_prob_list[j]))
                    result+=LR_v5_prob_prediction_list[j]
                    result = float(result)/3.0
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v10 LR rank average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                
                # try the XGBoost V7 feature XGBoost V10 feature LR ensemble
                final_average_list = []
                for j in range(len(v10_voting_prob_list)):
                    result = 0
                    v7_item = 1
                    v5_item = 1
                    v10_item = 1
                    for item in v7_voting_prob_list[j]:
                        v7_item*=item
                    v7_item = math.pow(v7_item,1/len(v7_voting_prob_list[j]))
                    for item in v10_voting_prob_list[j]:
                        v10_item*=item
                    v10_item = math.pow(v10_item, 1/len(v10_voting_prob_list[j]))
                    result+=LR_v5_prob_prediction_list[j]
                    result = float(result)/3.0
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v7 v10 LR rank average precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                
                # try the XGBoost V7 feature XGBoost V10 feature XGBoost V5 feature ensemble
                final_average_list = []
                for j in range(len(v7_voting_prob_list)):
                    result = 1
                    v7_item = 1
                    v5_item = 1
                    v10_item = 1
                    for item in v7_voting_prob_list[j]:
                        v7_item*=item
                    v7_item = math.pow(v7_item,1/len(v7_voting_prob_list[j]))
                    for item in v5_voting_prob_list[j]:
                        v5_item*=item
                    v5_item = math.pow(v5_item,1/len(v5_voting_prob_list[j]))
                    for item in v10_voting_prob_list[j]:
                        v10_item*=item
                    v10_item = math.pow(v10_item, 1/len(v10_voting_prob_list[j]))
                    result*=v7_item
                    result*=v5_item
                    result*=v10_item
                    if LR_v5_prob_prediction_list[j]!=0:
                        result*=LR_v5_prob_prediction_list[j]
                    result = math.pow(result,1/4.0)
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 v10 LR rank rank precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                
                # try the XGBoost V7 feature  XGBoost V5 feature LR ensemble
                final_average_list = []
                for j in range(len(v7_voting_prob_list)):
                    result = 1
                    v7_item = 1
                    v5_item = 1
                    for item in v7_voting_prob_list[j]:
                        v7_item*=item
                    v7_item = math.pow(v7_item,1/len(v7_voting_prob_list[j]))
                    for item in v5_voting_prob_list[j]:
                        v5_item*=item
                    v5_item = math.pow(v5_item,1/len(v5_voting_prob_list[j]))
                    result*=v7_item
                    result*=v5_item
                    if LR_v5_prob_prediction_list[j]!=0:
                        result*=LR_v5_prob_prediction_list[j]
                    result = math.pow(result,1/3.0)
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v7 LR rank rank precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                
                # try the XGBoost V7 feature  XGBoost V10 feature LR ensemble
                final_average_list = []
                for j in range(len(v7_voting_prob_list)):
                    result = 1
                    v7_item = 1
                    v10_item = 1
                    for item in v7_voting_prob_list[j]:
                        v7_item*=item
                    v7_item = math.pow(v7_item,1/len(v7_voting_prob_list[j]))
                    for item in v10_voting_prob_list[j]:
                        v10_item*=item
                    v10_item = math.pow(v10_item, 1/len(v10_voting_prob_list[j]))
                    result*=v7_item
                    result*=v10_item
                    if LR_v5_prob_prediction_list[j]!=0:
                        result*=LR_v5_prob_prediction_list[j]
                    result = math.pow(result,1/3.0)
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v7 v10 LR rank rank precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                
                # try the XGBoost V5 feature  XGBoost V10 feature LR ensemble
                final_average_list = []
                for j in range(len(v7_voting_prob_list)):
                    result = 1
                    result*=np.mean(v10_voting_prob_list[j])
                    result*=np.mean(v5_voting_prob_list[j])
                    if LR_v5_prob_prediction_list[j]!=0:
                        result*=LR_v5_prob_prediction_list[j]
                    result = math.pow(result,1/3.0)
                    if result>0.5:
                        final_average_list.append(1)
                    else:
                        final_average_list.append(0)
                print("the ensemble for v5 v10 LR rank rank precision is {}".format(metrics.accuracy_score(y_va, final_average_list)))
                print("the horizon is {}".format(horizon))
                print("the lag is {}".format(lag))
                print("the train date is {}".format(split_date[0]))
                print("the test date is {}".format(split_date[1]))
