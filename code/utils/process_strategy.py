'''
    
'''
import os
import sys


import json
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool as pl
from utils.Technical_indicator import *
from itertools import permutations, product
from copy import copy

def strategy_testing(X,ground_truth,strategy_params,activation_params,cov = ""):
    cols = X.columns.values.tolist()
    ground_truth = ground_truth[:-5]
    for col in cols:
        if ground_truth+"_High" == col and activation_params["strat3"]:
            X[col+cov+"_strat3"] = strategy_3(X[col],strategy_params['strat3']['window'])

        if ground_truth+"_Close" == col:
            setting = col[:-5]
            if setting+"High" in cols and setting+"Low" in cols and activation_params['sar']:
                X[setting+cov+'sar'] = sar(X[setting+"High"],X[setting+"Low"],X[col],strategy_params['sar']['initial'],strategy_params['sar']['maximum'])
            if activation_params['rsi']:
                X[col+cov+"_rsi"] = rsi(copy(X[col]),strategy_params['rsi']['window'],strategy_params['rsi']['upper'],strategy_params['rsi']['lower'])
            if activation_params["strat1"]:
                X[col+cov+"_strat1"] = strategy_1(X[col],strategy_params['strat1']['short window'],strategy_params['strat1']['med window'])
            if activation_params["strat2"]:
                X[col+cov+"_strat2"] = strategy_2(X[col],strategy_params['strat2']['window'])
            if activation_params["strat3"]:
                X[col+cov+"_strat3"] = strategy_3(X[col],strategy_params['strat3']['window'])
            if activation_params["strat7"]:
                X[col+cov+"_strat7"] = strategy_7(X[col],strategy_params['strat7']['window'],strategy_params['strat7']['limiting_factor'])
            if activation_params["strat9"]:
                X[col+cov+"_strat9"] = strategy_9(X[col],strategy_params['strat9']['FastLength'],strategy_params['strat9']['SlowLength'],strategy_params['strat9']['MACDLength'])
            if ground_truth+"_High" in cols and ground_truth+"_Low" in cols and activation_params["strat6"]:
                X[setting+cov+"strat6"] = strategy_6(X[setting+"High"],X[setting+"Low"],X[col],strategy_params['strat6']['window'],strategy_params['strat6']['limiting_factor'])
            
    return X

def output(time_series,split_dates,ground_truth,strategy_params,activation_params,array,check = True):
    org_cols = set(time_series.columns.values.tolist()) - set(["Label"])
    strat = None
    sp = copy(strategy_params)
    for key in activation_params.keys():
        if activation_params[key]:
            strat = key
    if strat is None:
        return 
    n = 0
    for key in sp[strat]:
        sp[strat][key] = array[n]
        n+=1
    if strat =='strat9' and sp[strat]['SlowLength'] < sp[strat]['FastLength']:
        return 
    ts = strategy_testing(copy(time_series),ground_truth,sp, activation_params)
    ts = ts[sorted(list(set(ts.columns.values.tolist()) - org_cols))]
    temp_list = array
    if check:
        ts = ts[(ts.index >= split_dates[0])&(ts.index < split_dates[1])]
    else:
        ts = ts[(ts.index >= split_dates[1])&(ts.index < split_dates[2])]
    
    for col in ts.columns.values.tolist():
        if col == "Label":
            continue
        temp_list.append(col)
        labels = copy(ts['Label'])
        length = len(labels)
        column = copy(ts[col])
        column = column.replace(0,np.nan)
        column = column.dropna()
        labels = labels.loc[column.index]
        labels = np.array(labels)*2-1
        column = np.array(column)
        if len(labels) == 0:
            compared = 0
        else:
            if check:
                compared = abs(sum(labels == column)/len(labels)-0.5)
            else:
                compared = sum(labels == column)/len(labels)
        temp_list.append(compared)
        temp_list.append(float(len(labels)/length))
    
    return temp_list

def parallel_process(ts,split_dates,strat,strat_results,ground_truth,strategy_params,activation_params,cov_inc,combination,mnm):
    ls = [list([copy(ts),split_dates,ground_truth,strategy_params,activation_params,list(com)]) for com in combination]
    pool = pl()
    results = pool.starmap_async(output,ls)
    pool.close()
    pool.join()
    results,idx = list(np.unique([i for i in results.get() if i],return_inverse = True,axis = 1))
    results = pd.DataFrame([list(res[idx]) for res in results])
    cov_col = pd.to_numeric(results[results.columns.values.tolist()[-1]])
    mn = float(str(min(cov_col))[0:3])
    if mn < mnm:
        mn = mnm
    mx = mn+cov_inc
    cov_array = []
    while len(results.loc[cov_col>=mn]) > 0:
        cov_array.append("_"+str(mn)+"-"+str(mx))
        temp_results = results[(cov_col<mx)&(cov_col>=mn)]
        if strat == 'sar':
            temp_results = temp_results.drop(2,axis = 1)
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Initial","Maximum","Acc","Cov"]
            strat_results['sar']['initial'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Initial"])
            strat_results['sar']['maximum'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Maximum"])

        elif strat =='rsi':
            temp_results = (temp_results.drop(3,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Window","Upper","Lower","Acc","Cov"]
            strat_results['rsi']['window'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Window"])
            strat_results['rsi']['upper'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Upper"])
            strat_results['rsi']['lower'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Lower"])

        elif strat == 'strat1':
            temp_results = (temp_results.drop(2,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Short Window","Med Window","Acc","Cov"]
            strat_results['strat1']['short window'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Short Window"])
            strat_results['strat1']['med window'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Med Window"])

        elif strat == 'strat2':
            temp_results = (temp_results.drop(1,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Window","Acc","Cov"]
            strat_results['strat2']['window'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Window"])

        elif strat == "strat3":
            temp_results = (temp_results.drop(1,axis = 1))
            temp_results = (temp_results.drop(4,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Window","Close Acc","Close Cov","High Acc","High Cov"]
            strat_results['strat3']['high window'].append(temp_results.loc[temp_results["High Acc"].idxmax(),"Window"])
            strat_results['strat3']['close window'].append(temp_results.loc[temp_results["Close Acc"].idxmax(),"Window"])

        elif strat == "strat6":
            temp_results = (temp_results.drop(2,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Window","Limiting Factor","Acc","Cov"]
            strat_results['strat6']['window'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Window"])
            strat_results['strat6']['limiting_factor'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Limiting Factor"])

        elif strat == "strat7":
            temp_results = (temp_results.drop(2,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Window","Limiting Factor","Acc","Cov"]
            strat_results['strat7']['window'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Window"])
            strat_results['strat7']['limiting_factor'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Limiting Factor"])

        elif strat == "strat9":
            temp_results = (temp_results.drop(3,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Slow Window","Fast Window","MACD Length","Acc","Cov"]
            strat_results['strat9']['SlowLength'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Slow Window"])
            strat_results['strat9']['FastLength'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Fast Window"])
            strat_results['strat9']['MACDLength'].append(temp_results.loc[temp_results["Acc"].idxmax(),"MACD Length"])

        mn += cov_inc
        mx += cov_inc

    keys = strategy_params[strat].keys()
    org_cols = set(ts.columns.values.tolist())
    ans = None
    if strat == "strat3":
        cov_col = pd.to_numeric(results[results.columns.values.tolist()[3]])
        while len(results.loc[cov_col>=mn]) > 0:
            cov_array.append("_"+str(mn)+"-"+str(mx))
            temp_results = results[(cov_col<mx)&(cov_col>=mn)]
            temp_results = copy(results)
            temp_results = (temp_results.drop(1,axis = 1))
            temp_results = (temp_results.drop(4,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Window","Close Acc","Close Cov","High Acc","High Cov"]
            strat_results['strat3']['close window'].append(temp_results.loc[temp_results["Close Acc"].idxmax(),"Window"])
        mn += cov_inc
        mx += cov_inc
        i=0
        for window in strat_results[strat]['high window']:
            sp = copy(strategy_params)
            sp[strat]['window'] = window
            temp_ans = strategy_testing(copy(ts),ground_truth,sp, activation_params,cov_array[i])
            temp_ans = temp_ans[sorted(list(set(temp_ans.columns.values.tolist()) - org_cols))]
            if ans is None:
                ans = temp_ans
            else:
                ans = pd.concat([ans,temp_ans], sort = True, axis = 1)
            i+=1
        i=0
        for window in strat_results[strat]['close window']:
            sp = copy(strategy_params)
            sp[strat]['window'] = window
            temp_ans = strategy_testing(copy(ts),ground_truth,sp, activation_params,cov_array[i])
            temp_ans = temp_ans[sorted(list(set(temp_ans.columns.values.tolist()) - org_cols))]
            if ans is None:
                ans = temp_ans
            else:
                ans = pd.concat([ans,temp_ans],sort = True, axis = 1)
            i+=1
        

    else:
        for i in range(max([len(res) for res in strat_results[strat].values()])):
            sp = copy(strategy_params)
            for key in keys:
                sp[strat][key] = strat_results[strat][key][i]
            temp_ans = strategy_testing(copy(ts),ground_truth,sp, activation_params,cov_array[i])
            temp_ans = temp_ans[sorted(list(set(temp_ans.columns.values.tolist()) - org_cols))]
            if ans is None:
                ans = temp_ans
            else:
                ans = pd.concat([ans,temp_ans], sort = True,axis = 1)
    ans = ans.dropna()
        
    return ans