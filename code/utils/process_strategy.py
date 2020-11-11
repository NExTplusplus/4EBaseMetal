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
from copy import copy,deepcopy

'''
    This file implements the generation of strategy signals based on activation paramas
'''

def strategy_testing(X,ground_truth,strategy_params,activation_params,cov = ""):
    '''
    input:  X                   : timeseries
            ground_truth        : string containing the ground truth column
            strategy_params     : dictionary holding the values for the parameters for all strategies
            activation_params   : dictionary representing which strategies are to be calculated
            cov                 : coverage

    out:    X                   : timeseries that includes the strategies
    '''
    cols = X.columns.values.tolist()
    ground_truth = ground_truth[:-5]
    for col in cols:
        if ground_truth+"_High" == col and activation_params["strat3_high"]:
            X[col+cov+"_strat3"] = strategy_3(X[col],strategy_params['strat3_high']['window'])
        if ground_truth+"_Spread" == col and activation_params["strat8"]:
            X[ground_truth+cov+"_strat8"] = strategy_7(X[col],strategy_params['strat8']['window'],strategy_params['strat8']['limiting_factor'])
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
            if activation_params["strat3_close"]:
                X[col+cov+"_strat3"] = strategy_3(X[col],strategy_params['strat3_close']['window'])
            if activation_params["strat5"]:
                X[col+cov+"_strat5"] = strategy_5(X[col],strategy_params['strat5']['window'])
                
            if activation_params["strat7"]:
                X[col+cov+"_strat7"] = strategy_7(X[col],strategy_params['strat7']['window'],strategy_params['strat7']['limiting_factor'])
            if activation_params["strat9"]:
                X[col+cov+"_strat9"] = strategy_9(X[col],strategy_params['strat9']['FastLength'],strategy_params['strat9']['SlowLength'],strategy_params['strat9']['MACDLength'])
            if activation_params["trend_1"]:
                X[col+cov+"_trend1"],X[col+cov+"_trend2"] = trend_1(X.index,X[col],strategy_params['trend_1']['window'])
            if ground_truth+"_High" in cols and ground_truth+"_Low" in cols and activation_params["strat4"]:
                X[setting+cov+"strat4"] = strategy_4(X[setting+"High"],X[setting+"Low"],X[col],strategy_params['strat4']['window'],strategy_params['strat4']['limiting_factor'])
            if ground_truth+"_High" in cols and ground_truth+"_Low" in cols and activation_params["strat6"]:
                X[setting+cov+"strat6"] = strategy_6(X[setting+"High"],X[setting+"Low"],X[col],strategy_params['strat6']['window'],strategy_params['strat6']['limiting_factor'])
                
    return X

def output(time_series,split_dates,ground_truth,strategy_params,activation_params,dc,check = True):
    '''
    input:  time_series         : time_series
            split_dates         : a list of dates that hold the start of training set, validation set and testing set
            ground_truth        : string containing the ground truth column
            strategy_params     : dictionary holding the dummy value for strategies
            activation_params   : dictionary representing which strategies are to be calculated
            dc                  : dictionary holding the values for the parameters for all strategies
            check               : if true calculates deviation from 0.5 in accuracy for training set, else calculates true accuracy of validation set
    output: temp_list           : list holding the values of dc and the final accuracy calculated     
    '''
    org_cols = set(time_series.columns.values.tolist()) - set(["Label"])
    strat = None
    sp = deepcopy(strategy_params)
    #identify which strategy to calculate
    for key in activation_params.keys():
        if activation_params[key]:
            strat = key
    if strat is None:
        return 
    n = 0
    #set the strategy parameters to values in dc
    for key in sp[strat]:
        sp[strat][key] = dc[key]
        n+=1
    
    # for strategy 9, remove those with slowlength smaller than fastlength
    if strat =='strat9' and sp[strat]['SlowLength'] < sp[strat]['FastLength']:
        return 
    
    #calculate strategies
    ts = strategy_testing(copy(time_series),ground_truth,sp, activation_params)
    ts = ts[sorted(list(set(ts.columns.values.tolist()) - org_cols))]
    temp_list = list(dc.values())
    if check:
        ts = ts[(ts.index >= split_dates[0])&(ts.index < split_dates[1])]
    else:
        ts = ts[(ts.index >= split_dates[1])&(ts.index < split_dates[2])]
    
    #calculate accuracy
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
                compared = abs(sum(labels == column)/len(labels))
            else:
                compared = sum(labels == column)/len(labels)
        temp_list.append(compared)
        temp_list.append(float(len(labels)/length))
    
    return temp_list

# tune for strategy parameters and output the timeseries based on the strategy parameters it generates
def parallel_process(ts,split_dates,strat,strat_results,ground_truth,strategy_params,activation_params,cov_inc,combination,mnm,op= "v1"):    
    '''
    input:  ts                  : timeseries
            split_dates         : a list of dates that hold the start of training set, validation set and testing set
            strat               : strategy that will be activated
            strat_results       : dictionary that holds the best strategy parameters across different strategies
            ground_truth        : string containing the ground truth column
            strategy_params     : dictionary holding the dummy value for strategies
            activation_params   : dictionary representing which strategies are to be calculated
            cov_inc             : coverage increment 
            combination         : combination of parameter values to test
            mnm                 : minimum coverage
            op                  : v1 is for single metal, v2 is for multiple metals
    output: ans                 : timeseries with strategies calculated with best strategy parameters 
    '''
    strat_dc = [create_dc_from_comb(strat,strategy_params,comb) for comb in combination]
    strat_keys = list(strategy_params[strat].keys())
    #parallel processing to test all combinations
    if len(strategy_params[strat][strat_keys[0]]) == 0:
        ls = [list([copy(ts),split_dates,ground_truth,strategy_params,activation_params,com]) for com in strat_dc]
        pool = pl()
        if op == 'v1':
            results = pool.starmap_async(output,ls)
        else:
            results = pool.starmap_async(output_v2,ls)
            
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
        while len(results.loc[cov_col>=mn]) > 0:
            temp_results = results[(cov_col<mx)&(cov_col>=mn)]
            # for sar
            if strat == 'sar':
                temp_results = temp_results.drop(2,axis = 1)
                for col in temp_results.columns:
                    temp_results[col] = pd.to_numeric(temp_results[col])
                temp_results.columns = ["Initial","Maximum","Acc","Cov"]
                strat_results['sar']['initial'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Initial"])
                strat_results['sar']['maximum'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Maximum"])
            # rsi
            elif strat =='rsi':
                temp_results = (temp_results.drop(3,axis = 1))
                for col in temp_results.columns:
                    temp_results[col] = pd.to_numeric(temp_results[col])
                temp_results.columns = ["Window","Upper","Lower","Acc","Cov"]
                strat_results['rsi']['window'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Window"])
                strat_results['rsi']['upper'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Upper"])
                strat_results['rsi']['lower'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Lower"])
            #strat1
            elif strat == 'strat1':
                temp_results = (temp_results.drop(2,axis = 1))
                for col in temp_results.columns:
                    temp_results[col] = pd.to_numeric(temp_results[col])
                temp_results.columns = ["Short Window","Med Window","Acc","Cov"]
                strat_results['strat1']['short window'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Short Window"])
                strat_results['strat1']['med window'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Med Window"])
            #strat2
            elif strat == 'strat2':
                temp_results = (temp_results.drop(1,axis = 1))
                for col in temp_results.columns:
                    temp_results[col] = pd.to_numeric(temp_results[col])
                temp_results.columns = ["Window","Acc","Cov"]
                strat_results['strat2']['window'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Window"])
            #strat3 high
            elif strat == "strat3_high":
                temp_results = (temp_results.drop(1,axis = 1))
                for col in temp_results.columns:
                    temp_results[col] = pd.to_numeric(temp_results[col])
                temp_results.columns = ["Window","Acc","Cov"]
                strat_results['strat3_high']['window'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Window"])
            #strat3 close          
            elif strat == "strat3_close":
                temp_results = (temp_results.drop(1,axis = 1))
                for col in temp_results.columns:
                    temp_results[col] = pd.to_numeric(temp_results[col])
                temp_results.columns = ["Window","Acc","Cov"]
                strat_results['strat3_close']['window'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Window"])
            #strat6
            elif strat == "strat6":
                temp_results = (temp_results.drop(2,axis = 1))
                for col in temp_results.columns:
                    temp_results[col] = pd.to_numeric(temp_results[col])
                temp_results.columns = ["Window","Limiting Factor","Acc","Cov"]
                strat_results['strat6']['window'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Window"])
                strat_results['strat6']['limiting_factor'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Limiting Factor"])
            #strat7
            elif strat == "strat7":
                temp_results = (temp_results.drop(2,axis = 1))
                for col in temp_results.columns:
                    temp_results[col] = pd.to_numeric(temp_results[col])
                temp_results.columns = ["Window","Limiting Factor","Acc","Cov"]
                strat_results['strat7']['window'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Window"])
                strat_results['strat7']['limiting_factor'].append(temp_results.loc[temp_results["Acc"].idxmax(),"Limiting Factor"])
            #strat9
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
    else:
        strat_results[strat] = strategy_params[strat]

    #generate time series
    keys = list(strategy_params[strat].keys())
    org_cols = set(ts.columns.values.tolist())
    ans = None
    cov_array = [str(i) for i in range(len(strat_results[strat][keys[0]]))]
    for i in range(max([len(res) for res in strat_results[strat].values()])):
        sp = deepcopy(strategy_params)
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

def post_process(ts,split_dates,strat,strat_results,ground_truth,strategy_params,activation_params,combination,mn):
    
    strat_dc = [create_dc_from_comb(strat,strategy_params,comb) for comb in combination]
    ls = [list([copy(ts),split_dates,ground_truth,strategy_params,activation_params,com]) for com in strat_dc]
    pool = pl()
    results = pool.starmap_async(output,ls)
    pool.close()
    pool.join()
    results,idx = list(np.unique([i for i in results.get() if i],return_inverse = True,axis = 1))
    results = pd.DataFrame([list(res[idx]) for res in results])
    
    tmp = abs(pd.to_numeric(results.iloc[:,-2])-0.5)
    
    if  len(results.loc[tmp>=mn])>0:
        temp_results = results.loc[tmp>=mn]
        
        if strat == 'sar':
            temp_results = temp_results.drop(2,axis = 1)
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Initial","Maximum","Acc","Cov"]
            strat_results['sar']['initial'] = list(temp_results["Initial"])
            strat_results['sar']['maximum'] = list(temp_results["Maximum"])
        
        elif strat =='rsi':
            temp_results = (temp_results.drop(3,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Window","Upper","Lower","Acc","Cov"]
            strat_results['rsi']['window'] = list(temp_results["Window"])
            strat_results['rsi']['upper'] = list(temp_results["Upper"])
            strat_results['rsi']['lower'] = list(temp_results["Lower"])

        elif strat == 'strat1':
            temp_results = (temp_results.drop(2,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Short Window","Med Window","Acc","Cov"]
            strat_results['strat1']['short window'] = list(temp_results["Short Window"])
            strat_results['strat1']['med window'] = list(temp_results["Med Window"])

        elif strat == 'strat2':
            temp_results = (temp_results.drop(1,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Window","Acc","Cov"]
            strat_results['strat2']['window'] = list(temp_results["Window"])

        elif strat == "strat3_high":
            temp_results = (temp_results.drop(1,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Window","Acc","Cov"]
            strat_results['strat3_high']['window'] = list(temp_results["Window"])
        
        elif strat == "strat3_close":
            temp_results = (temp_results.drop(1,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Window","Acc","Cov"]
            strat_results['strat3_close']['window'] = list(temp_results["Window"])

        elif strat == "strat6":
            temp_results = (temp_results.drop(2,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Window","Limiting Factor","Acc","Cov"]
            strat_results['strat6']['window'] = list(temp_results["Window"])
            strat_results['strat6']['limiting_factor'] = list(temp_results["Limiting Factor"])

        elif strat == "strat7":
            temp_results = (temp_results.drop(2,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Window","Limiting Factor","Acc","Cov"]
            strat_results['strat7']['window']=list(temp_results["Window"])
            strat_results['strat7']['limiting_factor'] = list(temp_results["Limiting Factor"])

        elif strat == "strat9":
            temp_results = (temp_results.drop(3,axis = 1))
            for col in temp_results.columns:
                temp_results[col] = pd.to_numeric(temp_results[col])
            temp_results.columns = ["Slow Window","Fast Window","MACD Length","Acc","Cov"]
            strat_results['strat9']['SlowLength'] = list(temp_results["Slow Window"])
            strat_results['strat9']['FastLength'] = list(temp_results["Fast Window"])
            strat_results['strat9']['MACDLength'] = list(temp_results["MACD Length"])
    else:
        print("No strong indicator for ",strat)
        return pd.DataFrame([])
            
    keys = list(strategy_params[strat].keys())
    org_cols = set(ts.columns.values.tolist())
    ans = None
    cov_array = [str(i) for i in range(len(strat_results[strat][keys[0]]))]
    for i in range(max([len(res) for res in strat_results[strat].values()])):
        sp = deepcopy(strategy_params)
        for key in keys:
            sp[strat][key] = strat_results[strat][key][i]
        #print(sp)
        temp_ans = strategy_testing(copy(ts),ground_truth,sp, activation_params,cov_array[i])
        temp_ans = temp_ans[sorted(list(set(temp_ans.columns.values.tolist()) - org_cols))]
        if ans is None:
            ans = temp_ans
        else:
            ans = pd.concat([ans,temp_ans], sort = True,axis = 1)
    ans = ans.dropna()
        
    return ans

# create dictionary from the combinations
def create_dc_from_comb(strat,strategy_params,combination):
    ans = {}
    strat_keys = list(strategy_params[strat].keys())
    for i in range(len(strat_keys)):
        ans[strat_keys[i]] = combination[i]
    return ans

# output for v10
def output_v2(time_series,split_dates,ground_truth,strategy_params,activation_params,dc,check = True):
    '''
    input:  time_series         : time_series
            split_dates         : a list of dates that hold the start of training set, validation set and testing set
            ground_truth        : string containing the ground truth column
            strategy_params     : dictionary holding the dummy value for strategies
            activation_params   : dictionary representing which strategies are to be calculated
            dc                  : dictionary holding the values for the parameters for all strategies
            check               : if true calculates deviation from 0.5 in accuracy for training set, else calculates true accuracy of validation set
    output: temp_list           : list holding the values of dc and the final accuracy calculated     
    '''
    org_cols = set(time_series.columns.values.tolist())
    strat = None
    sp = deepcopy(strategy_params)
    for key in activation_params.keys():
        if activation_params[key]:
            strat = key
    if strat is None:
        return 
    n = 0
    for key in sp[strat]:
        sp[strat][key] = dc[key]
        n+=1
    if strat =='strat9' and sp[strat]['SlowLength'] < sp[strat]['FastLength']:
        return 
    ts = None
    for gt in ['LME_Co_Spot','LME_Al_Spot',"LME_Ni_Spot",'LME_Zi_Spot','LME_Ti_Spot','LME_Le_Spot']:
        two_alph = gt.split("_")[1]
        label = copy(time_series[two_alph+' Label'])
        temp_ts = strategy_testing(copy(time_series), gt,sp, activation_params)
        generated_col = (list(set(temp_ts.columns.values.tolist()) - org_cols))
        temp_ts['LME_Co'+generated_col[0][7:]] = temp_ts[generated_col[0]]
        temp_ts = temp_ts[sorted(list(set(temp_ts.columns.values.tolist()) - org_cols - set(generated_col)))]
        temp_ts['Label'] = label
        if ts is None:
            ts = temp_ts
        else:
            ts = pd.concat([ts,temp_ts], sort = False, axis = 0)
    temp_list = list(dc.values())
    if check:
        ts = ts[(ts.index >= split_dates[0])&(ts.index < split_dates[1])]
    else:
        ts = ts[(ts.index >= split_dates[1])&(ts.index < split_dates[2])]
    ts = ts.reset_index(drop = True)
    #calculate accuracy
    for col in ts.columns.values.tolist():
        if "Label" == col:
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
