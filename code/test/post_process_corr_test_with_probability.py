import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy,deepcopy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from data.load_data import load_data
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
    parser.add_argument('-voting','--voting',type=str,help='there are five methods for voting: all,far,same,near,reverse', default = "all")
    
    parser.add_argument('-w','--W_version',type=int,help='there are five methods for voting: all,far,same,near,reverse', default = 1)
    args = parser.parse_args()
    
    if args.ground_truth =='None':
        args.ground_truth = None
    os.chdir(os.path.abspath(sys.path[0]))
    
    # read data configure file
    with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)
    args.lag = [int(s) for s in args.lag.split(",")]
    args.ground_truth = args.ground_truth.split(",")
    comparison = None
    n = 0
    for f in fname_columns:
        ans = {'test_date':[],'hp':[]}
        lag = args.lag
        
        #read data
        if args.source == "NExT":
            from utils.read_data import read_data_NExT
            data_list, LME_dates = read_data_NExT(f, "2003-11-12")
            time_series = pd.concat(data_list, axis = 1, sort = True)
        elif args.source == "4E":
            from utils.read_data import read_data_v5_4E
            time_series, LME_dates = read_data_v5_4E("2003-11-12")
        
        #generate parameters for load data
        length = 5
        split_dates = rolling_half_year("2009-07-01","2019-01-01",length)
        split_dates  =  split_dates[-4:]
        version_params=generate_version_params(args.version)
        
        #iterate over split dates
        for split_date in split_dates:
            horizon = args.steps
            norm_volume = "v1"
            norm_3m_spread = "v1"
            norm_ex = "v1"
            len_ma = 5
            len_update = 30
            tol = 1e-7

            norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                        'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
            tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                                            'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}

            final_y_tr = pd.DataFrame()
            final_y_pred = pd.DataFrame()
            final_y_va = []
            ts = copy(time_series.loc[split_date[0]:split_date[2]])
            
            # iterate over prediction horizon
            for a,h in enumerate([1,3,5]):
                temp_y_tr = []
                temp_y_va = []
                temp_X_te = None
                temp_y_te = None 
                
                i = 0
                corr_col = []
                y_ = []
                dates = pd.DataFrame(index = LME_dates)
                
                #iterate over different ground truths 
                for ground_truth in ['LME_Co_Spot','LME_Al_Spot','LME_Ni_Spot','LME_Ti_Spot','LME_Zi_Spot','LME_Le_Spot']:
                    print(ground_truth,h)
                    corr_col.append(ground_truth+str(h))
                    metal_id = [0,0,0,0,0,0]
                    metal_id[i] = 1
                    
                    # load data
                    X_tr, y_tr, X_va, y_va, X_te, y_te, norm_check,column_list = load_data(copy(ts),LME_dates,h,[ground_truth],lag[a],split_date,norm_params,tech_params,version_params)
                    
                    # post load process and metal id extension
                    y_tr = np.concatenate(y_tr)
                    y_va = np.concatenate(y_va)
                    temp_y_tr.append(y_tr)
                    y_.append(y_tr.flatten())
                    temp_y_va.append(y_va)
                    i+=1
                for j in temp_y_va:
                    final_y_va.append(j)
                to_be_concat = pd.DataFrame(np.array(y_).transpose(), columns = corr_col, index = list(dates.loc[(dates.index >=split_date[0])&(dates.index < split_date[1])].index[(args.lag[a]-1):]))
                final_y_tr = pd.concat([final_y_tr,to_be_concat], axis = 1)

                #load precalculated probabilities
                co_pred = np.loadtxt("LMCADY_horizon_"+str(h)+"_"+split_date[1]+"_v10_striplag30.txt")            
                al_pred = np.loadtxt("LMAHDY_horizon_"+str(h)+"_"+split_date[1]+"_v10_striplag30.txt")            
                ni_pred = np.loadtxt("LMNIDY_horizon_"+str(h)+"_"+split_date[1]+"_v10_striplag30.txt")            
                ti_pred = np.loadtxt("LMSNDY_horizon_"+str(h)+"_"+split_date[1]+"_v10_striplag30.txt")            
                zi_pred = np.loadtxt("LMZSDY_horizon_"+str(h)+"_"+split_date[1]+"_v10_striplag30.txt")            
                le_pred = np.loadtxt("LMPBDY_horizon_"+str(h)+"_"+split_date[1]+"_v10_striplag30.txt")   

                #iterate over ground truths for prediction generation
                for metal, pred in enumerate([co_pred,al_pred
                ,ni_pred,ti_pred,zi_pred,le_pred]):
                    arr = []
                    for j in pred:
                        count_1 = []
                        count_0 = []
                        for k in j: 
                            if k > 0.5:
                                count_1.append(k)
                            else:
                                count_0.append(k)
                        if len(count_1) > len(count_0):
                            arr.append(np.average(count_1))
                        else:
                            arr.append(np.average(count_0))
                    final_y_pred = pd.concat([final_y_pred, pd.DataFrame({corr_col[metal]:arr})], axis = 1)

            final_y_tr.dropna(inplace = True)
            final_y_va = [s.flatten() for s in final_y_va]
            final_y_va = pd.DataFrame(np.transpose(np.array(final_y_va)),columns = final_y_pred.columns.values.tolist(),index =final_y_pred.index)
                
            #post process
            for hp in np.arange(0.00,0.5,0.05):
                corrected_y_pred = pd.DataFrame()
                
                #calculate matrix to be solved
                W = get_W(final_y_tr,hp,args.W_version)
                original_prediction = deepcopy(final_y_pred)
                
                #generate corrected prediction
                for j in range(len(final_y_pred)):
                    y = prediction_correction(W,original_prediction.iloc[j,:])
                    y = [[s] for s in y]
                    corrected_y_pred = pd.concat([corrected_y_pred,pd.DataFrame(np.transpose(y),columns = final_y_pred.columns.values.tolist(),index = [final_y_pred.index[j]])], axis = 0)
                    for k in original_prediction.columns.values.tolist():
                        if corrected_y_pred.loc[corrected_y_pred.index[j],k] > 0.5:
                            corrected_y_pred.loc[corrected_y_pred.index[j],k] = 1
                        else:
                            corrected_y_pred.loc[corrected_y_pred.index[j],k] = 0
                
                #calculate accuracy
                acc = np.sum(np.array(final_y_va == corrected_y_pred), axis = 0)/len(final_y_va.index)
                ans['hp'].append(hp)
                ans['test_date'].append(split_date[1])
                for i,k in enumerate(final_y_pred.columns.values.tolist()):
                    if k not in ans.keys():
                        ans[k] = [acc[i]]       
                    else:
                        ans[k].append(acc[i])
        ans = pd.DataFrame(ans)
        ans = ans[["hp","test_date"]+sorted(ans.columns)[:-2]]
        ans.to_csv("corr_test_with_prob"+str(args.lag)+str(args.W_version)+".csv")    
            