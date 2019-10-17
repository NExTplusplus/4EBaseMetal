import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy,deepcopy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from data.load_data import load_data
from model.logistic_regression import LogReg
from utils.transform_data import flatten
from utils.construct_data import rolling_half_year
from utils.log_reg_functions import objective_function, loss_function
from utils.post_process import *
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
        default='exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf'
    )
    parser.add_argument('-gt','--ground_truth',help = "ground truth",type = str, default = "LME_Co_Spot")
    parser.add_argument('-max_iter','--max_iter',type=int,default=100,
                        help='max number of iterations')
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "NExT"
    )
    parser.add_argument(
        '-s','--steps', type=int, default = 5, help='horizon'
    )
    parser.add_argument(
        '-l','--lag', type=str, default = "20,20,5", help='lag'
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
    parser.add_argument('-max_depth','--max_depth',type = str, help='feed the parameter into the model', default="5,5,3")
    parser.add_argument('-learning_rate','--learning_rate',type=str,help='feed the parameter into the model', default="0.7,0.7,0.6")
    parser.add_argument('-gamma','--gamma',type=str,help='feed the parameter into the model',default="0.8,0.7,0.8")
    parser.add_argument('-min_child','--min_child',type=str,help='feed the parameter into the model',default="5,3,6")
    parser.add_argument('-subsample','--subsample',type=str,help='feed the parameter into the model',default="0.85,0.7,0.9")
    parser.add_argument('-voting','--voting',type=str,help='there are five methods for voting: all,far,same,near,reverse', default = "all")
    parser.add_argument('-w','--W_version',type=int,help='there are five methods for voting: all,far,same,near,reverse', default = 1)
    args = parser.parse_args()
    args.lag = [int(s) for s in args.lag.split(",")]
    args.max_depth = args.max_depth.split(",")
    args.learning_rate = args.learning_rate.split(",")
    args.gamma = args.gamma.split(",")
    args.min_child = args.min_child.split(",")
    args.subsample = args.subsample.split(",")
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
            ans = {'test_date':[],'hp':[]}
            cp = pd.DataFrame()
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
                if args.xgboost==1:
                    print(args.xgboost)
                    norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                                'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':True}
                else:
                    norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                                'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
                tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                                                'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}

                final_y_tr = pd.DataFrame()
                final_y_pred = pd.DataFrame()
                final_y_va = []
                ts = copy(time_series.loc[split_date[0]:split_date[2]])
                for a,h in enumerate([1,3,5]):
                    temp_X_tr = []
                    temp_y_tr = []
                    temp_X_va = []
                    temp_y_va = []
                    temp_X_te = None
                    temp_y_te = None 
                    
                    i = 0
                    corr_col = []
                    y_ = []
                    dates = pd.DataFrame(index = LME_dates)
                    for ground_truth in ['LME_Co_Spot','LME_Al_Spot','LME_Ni_Spot','LME_Ti_Spot','LME_Zi_Spot','LME_Le_Spot']:
                        print(ground_truth,h)
                        corr_col.append(ground_truth+str(h))
                        metal_id = [0,0,0,0,0,0]
                        metal_id[i] = 1
                        X_tr, y_tr, X_va, y_va, X_te, y_te, norm_check,column_list = load_data(copy(ts),LME_dates,h,[ground_truth],lag[a],split_date,norm_params,tech_params,version_params)
                        print(split_date)
                        X_tr = np.concatenate(X_tr)
                        X_tr = X_tr.reshape(len(X_tr),lag[a]*len(column_list[0]))
                        X_tr = np.append(X_tr,[metal_id]*len(X_tr),axis = 1)
                        y_tr = np.concatenate(y_tr)
                        X_va = np.concatenate(X_va)
                        X_va = X_va.reshape(len(X_va),lag[a]*len(column_list[0]))
                        X_va = np.append(X_va,[metal_id]*len(X_va),axis = 1)
                        y_va = np.concatenate(y_va)
                        temp_X_tr.append(X_tr)
                        temp_y_tr.append(y_tr)
                        y_.append(y_tr.flatten())
                        temp_X_va.append(X_va)
                        temp_y_va.append(y_va)
                        i+=1
                    for j in temp_y_va:
                        final_y_va.append(j)
                    to_be_concat = pd.DataFrame(np.array(y_).transpose(), columns = corr_col, index = list(dates.loc[(dates.index >=split_date[0])&(dates.index < split_date[1])].index[(args.lag[a]-1):]))
                    final_y_tr = pd.concat([final_y_tr,to_be_concat], axis = 1)
                    temp_X_tr = [np.transpose(arr) for arr in np.dstack(temp_X_tr)]
                    temp_y_tr = [np.transpose(arr) for arr in np.dstack(temp_y_tr)]
                    temp_X_tr = np.reshape(temp_X_tr,[np.shape(temp_X_tr)[0]*np.shape(temp_X_tr)[1],np.shape(temp_X_tr)[2]])
                    temp_y_tr = np.reshape(temp_y_tr,[np.shape(temp_y_tr)[0]*np.shape(temp_y_tr)[1],np.shape(temp_y_tr)[2]])
                    column_lag_list = []
                    column_name = []
                    for i in range(lag[a]):
                        for item in column_list[0]:
                            new_item = item+"_"+str(lag[a]-i)
                            column_lag_list.append(new_item)
                    column_lag_list.append("Co")
                    column_lag_list.append("Al")
                    column_lag_list.append("Ni")
                    column_lag_list.append("Ti")
                    column_lag_list.append("Zi")
                    column_lag_list.append("Le")
                    train_dataframe = pd.DataFrame(temp_X_tr,columns=column_lag_list)
                    train_X = train_dataframe.loc[:,column_lag_list]
                    train_y = pd.DataFrame(temp_y_tr,columns=['result'])
                    for i,gt in enumerate(["LMCADY","LMAHDY","LMNIDY","LMSNDY","LMZSDY","LMPBDY"]):
                        print(split_date)
                        test_dataframe = pd.DataFrame(temp_X_va[i],columns=column_lag_list)
                        test_X = test_dataframe.loc[:,column_lag_list] 
                        n_splits=args.k_folds
                        from sklearn.metrics import accuracy_score
                        model = xgb.XGBClassifier(max_depth=int(args.max_depth[a]),
                                    learning_rate = float(args.learning_rate[a]),
                                    n_estimators=500,
                                    silent=True,
                                    nthread=10,
                                    gamma=float(args.gamma[a]),
                                    min_child_weight=int(args.min_child[a]),
                                    subsample=float(args.subsample[a]),
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
                        if args.voting=='all':
                            result = np.concatenate((folder_1,folder_2,folder_3,folder_4,folder_5,folder_6,folder_7,folder_8,folder_9,folder_10),axis=1)
                            temp_list = []
                            for j in range(len(result)):
                                count_1=[]
                                count_0=[]
                                for item in result[j]:
                                    if item > 0.5:
                                        count_1.append(item)
                                    else:
                                        count_0.append(item)
                                if len(count_1)>len(count_0):
                                    temp_list.append(np.average(count_1))
                                else:
                                    temp_list.append(np.average(count_0))
                            #print("the lag is {}".format(lag))
                            temp_list = pd.DataFrame({gt+str(h):temp_list},index = list(dates.loc[(dates.index >=split_date[1])&(dates.index < split_date[2])].index))
                            final_y_pred = pd.concat([final_y_pred, temp_list], axis = 1)
                final_y_tr.dropna(inplace = True)
                final_y_va = [s.flatten() for s in final_y_va]
                final_y_va = pd.DataFrame(np.transpose(np.array(final_y_va)),columns = final_y_pred.columns.values.tolist(),index =final_y_pred.index)

                for hp in np.arange(0.00,0.5,0.05):
                    corrected_y_pred = pd.DataFrame()
                    W = get_W(final_y_tr,hp,args.W_version)
                    original_prediction = deepcopy(final_y_pred)
                    for j in range(len(final_y_pred)):
                        y = prediction_correction(W,original_prediction.iloc[j,:])
                        y = [[s] for s in y]
                        corrected_y_pred = pd.concat([corrected_y_pred,pd.DataFrame(np.transpose(y),columns = final_y_pred.columns.values.tolist(),index = [final_y_pred.index[j]])], axis = 0)
                        for k in original_prediction.columns.values.tolist():
                            if corrected_y_pred.loc[corrected_y_pred.index[j],k] > 0.5:
                                corrected_y_pred.loc[corrected_y_pred.index[j],k] = 1
                            else:
                                corrected_y_pred.loc[corrected_y_pred.index[j],k] = 0
                    acc = np.sum(np.array(final_y_va == corrected_y_pred), axis = 0)/len(final_y_va.index)
                    ans['hp'].append(hp)
                    ans['test_date'].append(split_date[1])
                    for i,k in enumerate(final_y_pred.columns.values.tolist()):
                        if k not in ans.keys():
                            ans[k] = [acc[i]]       
                        else:
                            ans[k].append(acc[i])
                    cp = pd.concat([cp,corrected_y_pred.reset_index()], ignore_index = True, axis = 0)
            cp.to_csv("pred.csv")
            ans = pd.DataFrame(ans)
            ans = ans[["hp","test_date"]+sorted(ans.columns)[:-2]]
            ans.to_csv("corr_test"+str(args.lag)+str(args.W_version)+".csv")    
                