import numpy as np
import pandas as pd
from copy import copy
import os
import sys
from datetime import datetime
from itertools import product, permutations
exp_path = sys.path[0]
sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],"..")))
from utils.normalize_feature import * 
from utils.Technical_indicator import *
from utils.process_strategy import *
from sklearn import preprocessing
import json
import scipy.stats as sct

#load the strategy parameters for version 9
def generate_strat_params_v1(ground_truth,steps):
    with open("exp/strat_param_v9.conf") as f:
        all_params = json.load(f)
    strat_params = all_params[ground_truth.split("_")[1]][str(steps)+"d"]
    activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":False,"strat5":False,"strat6":True,"strat7":True,"strat8":False,"strat9":True,"trend_1":False}
    return strat_params,activation_params

#load the strategy parameters for version 10
def generate_strat_params_v2(ground_truth,steps):
    file_path = ("/").join(exp_path.split("/")[:-1])
    with open(file_path+"/exp/strat_param_v10.conf") as f:
        all_params = json.load(f)
    strat_params = all_params[ground_truth.split("_")[1]][str(steps)+"d"]
    activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":False,"strat5":False,"strat6":True,"strat7":True,"strat8":False,"strat9":True,"trend_1":False}
    return strat_params,activation_params

#load the strategy parameters for version 11
def generate_strat_params_v3(ground_truth,steps):
    with open("exp/strat_param_v11.conf") as f:
        all_params = json.load(f)
    strat_params = all_params[ground_truth.split("_")[1]][str(steps)+"d"]
    activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":False,"strat5":False,"strat6":True,"strat7":True,"strat8":False,"strat9":True,"trend_1":False}
    return strat_params,activation_params

#load the strategy parameters for version 12
def generate_strat_params_v4(ground_truth,steps):
    file_path = ("/").join(exp_path.split("/")[:-1])
    with open(file_path+"/exp/strat_param_v10.conf") as f:
        all_params = json.load(f)
    strat_params = all_params[ground_truth.split("_")[1]][str(steps)+"d"]
    activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":False,"strat5":False,"strat6":True,"strat7":True,"strat8":False,"strat9":True,"trend_1":False}
    return strat_params,activation_params

#load the strategy parameters for version 14
def generate_strat_params_v5(ground_truth,steps):
    print("################generate_strat_params_v5##################")
    with open(os.path.join(sys.path[0],"exp/strat_param_v14.conf")) as f:
        all_params = json.load(f)
    strat_params = all_params[ground_truth.split("_")[1]][str(steps)+"d"]
    activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":False,"strat5":False,"strat6":True,"strat7":True,"strat8":True,"strat9":True,"trend_1":False}
    return strat_params,activation_params

#load the strategy parameters for version 18
def generate_strat_params_v6(ground_truth,steps):
    print("################generate_strat_params_v6##################")
    with open(os.path.join(sys.path[0],"exp/strat_param_v18.conf")) as f:
        all_params = json.load(f)
    print(all_params)
    strat_params = all_params[ground_truth.split("_")[1]][str(steps)+"d"]
    activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":True,"strat5":True,"strat6":True,"strat7":True,"strat8":False,"strat9":True,"trend_1":False}
    return strat_params,activation_params

#load the strategy parameters for version 20
def generate_strat_params_v7(ground_truth,steps):
    print("################generate_strat_params_v7##################")
    with open(os.path.join(sys.path[0],"exp/strat_param_v20.conf")) as f:
        all_params = json.load(f)
    strat_params = all_params[ground_truth.split("_")[1]][str(steps)+"d"]
    activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":False,"strat5":False,"strat6":True,"strat7":True,"strat8":False,"strat9":True,"trend_1":True}
    return strat_params,activation_params    

#load the strategy parameters for version 22
def generate_strat_params_v8(ground_truth,steps):
    print("################generate_strat_params_v8##################")
    with open(os.path.join(sys.path[0],"exp/strat_param_v20.conf")) as f:
        all_params = json.load(f)
    strat_params = all_params[ground_truth.split("_")[1]][str(steps)+"d"]
    activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":True,"strat5":True,"strat6":True,"strat7":True,"strat8":True,"strat9":True,"trend_1":True}
    return strat_params,activation_params    

#the function is to deal with the abnormal data
def deal_with_abnormal_value_v1(data):
    #deal with the big value
    column_list = []
    for column in data.columns:
        if "_OI" in column:
            column_list.append(column)
    year_list = list(range(int(data.index[0].split("-")[0]),int(data.index[-1].split("-")[0])+1))
    month_list = ['01','02','03','04','05','06','07','08','09','10','11','12']
    for column_name in column_list:   
        for year in year_list:
            for month in month_list:
                start_time = str(year)+'-'+month+'-'+'01'
                end_time = str(year)+'-'+month+'-'+'31'
                value_dict = {}
                value_list=[]
                temp = copy(data.loc[(data.index >= start_time)&(data.index <= end_time)])
                if len(temp) == 0 or len(temp[column_name].dropna()) == 0:
                    continue
                average = np.mean(temp[column_name].dropna())
                data.at[temp[column_name].idxmax(),column_name] = average
                
    #deal with the minor value in OI
    for column_name in column_list:
        temp = data[column_name]
        nsmallest = temp.nsmallest(n = 20).index
        for ind in nsmallest:
            start_time = ind[:-2]+'01'
            end_time = ind[:-2]+'31'
            data.at[ind,column_name] = np.mean(data.loc[(data.index >= start_time)&(data.index <= end_time)][column_name])
    #missing value interpolate
    data = data.interpolate(axis = 0)

    return data

def deal_with_abnormal_value_v2(data):
    #deal with the big value
    column_list = []
    for column in data.columns:
        if "_OI" in column:
            column_list.append(column)
    year_list = list(range(int(data.index[0].split("-")[0]),int(data.index[-1].split("-")[0])+1))
    month_list = ['01','02','03','04','05','06','07','08','09','10','11','12']
    for column_name in column_list:   
        for year in year_list:
            for month in month_list:
                start_time = str(year)+'-'+month+'-'+'01'
                end_time = str(year)+'-'+month+'-'+'31'
                value_dict = {}
                value_list=[]
                temp = copy(data.loc[(data.index >= start_time)&(data.index <= end_time)])
                if len(temp) == 0 or len(temp[column_name].dropna()) == 0:
                    continue
                average = np.mean(temp[column_name].dropna())
                data.at[temp[column_name].idxmax(),column_name] = average
                
    #missing value interpolate
    data = data.interpolate(axis = 0)

    return data

#this function is to build the time_feature into the data
def insert_date_into_feature_v1(time_series):
    time_series['month']=[item[1] for item in time_series.index.str.split('-').to_list()]
    time_series['day']=[item[2] for item in time_series.index.str.split('-').to_list()]
    #print(pd.Series([item[1] for item in time_series.index.str.split('-').to_list()]))
    #print(time_series['day'])
    return time_series


#the function is to label the target and rename the result
def labelling_v1(X,horizon, ground_truth_columns):
    """
    X: which equals the timeseries
    horizon: the time horizon
    ground_truth_columns: the column we predict
    """
    assert ground_truth_columns != []
    ans = []
    for ground_truth in ground_truth_columns:
        labels = copy(X[ground_truth])
        labels = labels.shift(-horizon) - labels
        labels = labels > 0
        labels = labels.rename("Label")
        ans.append(labels)
    return ans

def labelling_v1_ex1(X,horizon,ground_truth_columns,lag):
    '''
    X: timeseries
    horizon: the time horizon
    ground_truth_columns: the columns we predict
    lag: number of days before current period that the ground truth is based on
    '''
    assert ground_truth_columns != []
    ans = []
    for ground_truth in ground_truth_columns:
        labels = copy(X[ground_truth])
        labels = labels.shift(-horizon) - labels.shift(lag)
        labels = labels[lag:]
        labels = labels > 0
        labels = labels.rename("Label")
        ans.append(labels)
    return ans

#this function is for 3 classification
def labelling_v1_ex2(X,horizon,ground_truth_columns,val_end):
    '''
    X: timeseries
    horizon: the time horizon
    ground_truth_columns: the columns we predict
    val_end: the last index of validation set to prevent data leakage
    '''
    assert ground_truth_columns != []
    ans = []
    for ground_truth in ground_truth_columns:
        labels = copy(X[ground_truth])
        labels = np.log( labels.shift(-horizon) / labels )
        mean = np.mean(labels[:val_end])
        std = np.std(labels[:val_end])
        threshold_1 = sct.norm.ppf(q=0.309,loc=mean,scale=std)
        threshold_2 = sct.norm.ppf(q=0.691,loc=mean,scale=std)
        labels[(labels <= threshold_1)&(labels.index <= labels.index[val_end])] = -1
        labels[(labels <= threshold_2)&(labels > threshold_1)&(labels.index <= labels.index[val_end])] = 0
        labels[(labels > threshold_2)&(labels.index <= labels.index[val_end])] = 1
        labels[(labels >= 0)&(labels.index >= labels.index[val_end])] = 1
        labels[(labels < 0)&(labels.index >= labels.index[val_end])] = -1

        labels = labels.rename("Label")
        ans.append(labels)
    
    return ans


#we use this function to make the data normalization
def normalize_without_1d_return_v1(timeseries,train_end, params):
    """
    timeseries: the dataframe we get from the data source
    train_end: string which we use to define the range we use to train
    params: A dictionary we use to feed the parameter
    """
    ans = {"nVol":False,"nSpread":False,"nEx":False}
    
    cols = timeseries.columns.values.tolist()
    ex = False
    if "CNYUSD" in cols:
        print("Considering Exchange Rate")
        ex = True
    #normalize the data based on the specific column
    for col in cols:
        #use the normalize_OI function to deal with the OI
        if "OI" in col:
            print("Normalizing OI:"+"=>".join((col,col[:-2]+"nOI")))
            timeseries[col[:-2]+"nOI"] = normalize_OI(copy(timeseries[col]),train_end,strength = params['strength'], both = params['both'])
        #use the normalize_volume function to deal with the volume
        if "Volume" in col:
            setting = col[:-6]
            if setting+"OI" in cols:
                ans["nVol"] = True
                print("Normalizing Volume:"+"=>".join((col,setting+"OI")))
                timeseries[setting+"nVolume"] = normalize_volume(copy(timeseries[col]), train_end = train_end, OI = copy(timeseries[setting+"OI"]),
                                                        len_ma = params['len_ma'],version = params['vol_norm'], 
                                                        strength = params['strength'], both = params['both'])
        #use the normalize_3mspot_spread function to create 3 month close to spot spread
        if "Close" in col:
            setting = col[:-5]
            if setting+"Spot" in cols:
                ans["nSpread"] = True
                print("Normalizing Spread:"+"=>".join((col,setting+"Spot")))
                timeseries[setting+"n3MSpread"] = normalize_3mspot_spread(copy(timeseries[col]),copy(timeseries[setting+"Spot"]),
                                                                len_update=params['len_update'],
                                                                version = params['spot_spread_norm'])
        #use the normalize_3mspot_spread_ex function to create cross exchange spread
        if "SHFE" in col and "Close" in col and ex:
            metal = col.split("_")[1]
            if "_".join(("LME",metal,"Spot")) in cols:
                ans["nEx"] = True
                print("+".join((col,"_".join(("LME",metal,"Spot"))))+"=>"+"_".join(("SHFE",metal,"nEx3MSpread")))
                timeseries["_".join(("SHFE",metal,"nEx3MSpread"))] = normalize_3mspot_spread_ex(copy(timeseries["_".join(("LME",metal,"Spot"))]),
                                                                                    copy(timeseries[col]),copy(timeseries["CNYUSD"]),
                                                                                    len_update=params['len_update'],
                                                                                    version = params['ex_spread_norm'])
            if "_".join(("LME",metal,"Close")) in cols:
                ans["nEx"] = True
                print("+".join((col,"_".join(("LME",metal,"Close"))))+"=>"+"_".join(("SHFE",metal,"nEx3MSpread")))
                timeseries["_".join(("SHFE",metal,"nExSpread"))] = normalize_3mspot_spread_ex(copy(timeseries["_".join(("LME",metal,"Close"))]),
                                                                                    copy(timeseries[col]),copy(timeseries["CNYUSD"]),
                                                                                    len_update=params['len_update'],
                                                                                    version = params['ex_spread_norm'])
            
    return timeseries, ans


#This function is for one-hot encoding.
def one_hot(dataframe):
    output = pd.DataFrame(index = dataframe.index)
    for col in dataframe.columns:
        output[col+'_positive'] = pd.Series(index = dataframe.index,data = [0]*len(dataframe))
        output[col+'_negative'] = pd.Series(index = dataframe.index,data = [0]*len(dataframe))
        output[col+'_positive'].loc[dataframe[col]==1] = 1
        output[col+'_negative'].loc[dataframe[col]==-1] = 1
        
    return output

#this function is to build the technical indicator which is called PVT and DIV_PVT
def technical_indication_v1(X,train_end,params,ground_truth):
    print('====================================technical indicator_v1========================================')
    cols = X.columns.values.tolist()
    for col in cols:
        setting = col[:-5]
        if 'Close' in col:
            if setting+'Volume' in cols:
                 print("+".join((col,setting+"Volume"))+"=>"+(setting+"divPVT"))
                 X[setting+"divPVT"] = divergence_pvt(copy(X[col]),copy(X[setting+"Volume"]),train_end, 
                                                                params = params)
    return X

#this function is to build the technical indicator which is called PVT and DIV_PVT for LME Ground Truth only
def technical_indication_v1_ex3(X,train_end,params,ground_truth):
    print('====================================technical indicator_v1_ex3========================================')
    cols = X.columns.values.tolist()
    for col in cols:
        setting = col[:-5]
        if 'Close' in col and setting in ground_truth[0]:
            if setting+'Volume' in cols:
                 print("+".join((col,setting+"Volume"))+"=>"+(setting+"divPVT"))
                 X[setting+"divPVT"] = divergence_pvt(copy(X[col]),copy(X[setting+"Volume"]),train_end, 
                                                                params = params)
    return X

def technical_indication_v2(X,train_end,params,ground_truth_columns):
    """
    X: which equals the timeseries
    train_end: string which we use to define the range we use to train
    params: A dictionary we use to feed the parameter
    """
    print('====================================technical indicator_v2========================================')
    cols = X.columns.values.tolist()
    for col in cols:
        setting = col[:-5]
        ground_truth = ground_truth_columns[0][4:6]
        if "Close" in col or 'Spot' in col:
            X[col+"_EMA"] = ema(copy(X[col]),params['Win_EMA'])
            X[col+"_bollinger"] = bollinger(copy(X[col]),params['Win_Bollinger'])
            X[col+"_PPO"] = ppo(copy(X[col]),params['Fast'],params['Slow'])
            X[col+"_RSI"] = rsi(copy(X[col]))
                
            if "Close" in col and setting+"Volume" in cols:
                
                print("+".join((col,setting+"Volume"))+"=>"+"+".join((setting+"PVT",setting+"divPVT")))
                X[setting+"PVT"] = pvt(copy(X.index),copy(X[col]),copy(X[setting+"Volume"]))
                X[setting+"divPVT"] = divergence_pvt(copy(X[col]),copy(X[setting+"Volume"]),train_end, 
                                                            params = params)
            
            if 'Close' in col and setting+'High' in cols and setting+'Low' in cols:
                X[setting+'NATR'] = natr(X[setting+"High"],X[setting+"Low"],X[col],params['Win_NATR'])
                X[setting+'VBM'] = vbm(X[setting+"High"],X[setting+"Low"],X[col],params['Win_VBM'])
                X[setting+'SAR'] = sar(X[setting+"High"],X[setting+"Low"],X[col],params['acc_initial'],params['acc_maximum'])
        
        if setting+"Open" in cols:
            if setting+'High' in cols and setting+'Low' in cols:
                for i in range(len(params['Win_VSD'])):
                    X[setting+"VSD"+str(params['Win_VSD'][i])] = vsd(X[setting+"High"],X[setting+"Low"],X[col],params['Win_VSD'][i])
            
                
    return X

#this function is to build the technical indicator which is called PVT,DIV_PVT,NATR,VSD,VBM,BSD BOLLINGER,
#EMA,SAR,PPO only for LME Ground Truth
def technical_indication_v2_ex3(X,train_end,params,ground_truth_columns):
    """
    X: which equals the timeseries
    train_end: string which we use to define the range we use to train
    params: A dictionary we use to feed the parameter
    """
    print('====================================technical indicator_v2_ex3========================================')
    cols = X.columns.values.tolist()
    for col in cols:
        setting = col[:-5]
        ground_truth = ground_truth_columns[0][4:6]
        if 'LME' in col and ground_truth in col:
            if "Close" in col or 'Spot' in col:
                X[col+"_EMA"] = ema(copy(X[col]),params['Win_EMA'])
                X[col+"_bollinger"] = bollinger(copy(X[col]),params['Win_Bollinger'])
                X[col+"_PPO"] = ppo(copy(X[col]),params['Fast'],params['Slow'])
                X[col+"_RSI"] = rsi(copy(X[col]))
                    
                if "Close" in col and setting+"Volume" in cols:
                    X[setting+"PVT"] = pvt(copy(X.index),copy(X[col]),copy(X[setting+"Volume"]))
                    X[setting+"divPVT"] = divergence_pvt(copy(X[col]),copy(X[setting+"Volume"]),train_end, 
                                                                params = params)
                
                if 'Close' in col and setting+'High' in cols and setting+'Low' in cols:
                    X[setting+'NATR'] = natr(X[setting+"High"],X[setting+"Low"],X[col],params['Win_NATR'])
                    X[setting+'VBM'] = vbm(X[setting+"High"],X[setting+"Low"],X[col],params['Win_VBM'])
                    X[setting+'SAR'] = sar(X[setting+"High"],X[setting+"Low"],X[col],params['acc_initial'],params['acc_maximum'])
            
            if setting+"Open" in cols:
               
                print("+".join((col,setting+"Open"))+"=>"+"+".join((setting+"PVT",setting+"divPVT")))
                if setting+'High' in cols and setting+'Low' in cols:
                    for i in range(len(params['Win_VSD'])):
                        X[setting+"VSD"+str(params['Win_VSD'][i])] = vsd(X[setting+"High"],X[setting+"Low"],X[col],params['Win_VSD'][i])
                
                
    return X
#generate strategy signals
def strategy_signal_v1(X,split_dates,ground_truth_columns,strategy_params,activation_params,cov_inc,mnm):
    
    strat_results = {'sar':{'initial':[],'maximum':[]},'rsi':{'window':[],'upper':[],'lower':[]},'strat1':{'short window':[],"med window":[]},'strat2':{'window':[]},'strat3_high':{'window':[]}, 'strat3_close':{'window':[]},'strat6':{'window':[],'limiting_factor':[]},'strat7':{'window':[],'limiting_factor':[]}, 'strat9':{'SlowLength':[],'FastLength':[],'MACDLength':[]}}
    cols = X.columns.values.tolist()
    ground_truth = ground_truth_columns[0]
    gt = ground_truth[:-5]
    tmp_pd = pd.DataFrame(index = X.index)
    output = pd.DataFrame(index = X.index)
    temp_act = copy(activation_params)
    for key in temp_act.keys():
        temp_act[key] = False
    for col in cols:

        #generate strategy 3 for High 
        if gt+"_High" == col and activation_params["strat3_high"]:
            act = copy(temp_act)
            act['strat3_high'] = True
            comb = list(range(5,51,2))
            comb = [[com] for com in comb]
            tmp_pd = parallel_process(X, split_dates, "strat3_high",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
            output_strat3 = one_hot(tmp_pd)
            output = pd.concat([output,output_strat3],sort = True, axis = 1)
            tmp_pd = pd.DataFrame(index = X.index)
        #generate strategy 8
        if gt+"_Spread" == col and activation_params["strat8"]:
            act = copy(temp_act)
            act['strat8'] = True
            limiting_factor = np.arange(1.8,2.45,0.1)
            window = list(range(10,51,2))
            comb = product(window,limiting_factor)
            tmp_pd = parallel_process(X, split_dates, "strat8",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
            output_strat8 = one_hot(tmp_pd)
            print(output_strat8.columns)
            output = pd.concat([output,output_strat8],sort = True, axis = 1)
            tmp_pd = pd.DataFrame(index = X.index)

        if gt+"_Close" == col:
            setting = col[:-5]
            #generate SAR
            if setting+"High" in cols and setting+"Low" in cols and activation_params['sar']:
                act = copy(temp_act)
                act['sar'] = True
                initial = np.arange(0.01,0.051,0.002)
                mx = np.arange(0.1,0.51,0.02)
                comb = product(initial,mx)
                tmp_pd = parallel_process(X, split_dates, "sar",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_sar = one_hot(tmp_pd)
                output = pd.concat([output,output_sar],sort = True, axis = 1)
                tmp_pd = pd.DataFrame(index = X.index)
            #generate RSI
            if activation_params['rsi']:
                act = copy(temp_act)
                act['rsi'] = True
                window = list(range(5,51,2))
                upper = list(range(60,91,10))
                lower = list(range(20,51,10))
                comb = product(window,upper,lower)
                tmp = parallel_process(X, split_dates, "rsi",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_rsi = one_hot(tmp)
                output = pd.concat([output,output_rsi],sort = True, axis = 1)
            #generate Strat 1
            if activation_params["strat1"]:
                act = copy(temp_act)
                act['strat1'] = True
                short_window = list(range(20,35,2))
                med_window = list(range(50,71,2))
                comb = product(short_window,med_window)
                tmp = parallel_process(X, split_dates, "strat1",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_strat1 = one_hot(tmp)
                output = pd.concat([output,output_strat1],sort = True, axis = 1)
            #generate strat2
            if activation_params["strat2"]:
                act = copy(temp_act)
                act['strat2'] = True
                comb = list(range(45,61,2))
                comb = [[com] for com in comb]
                tmp_pd = parallel_process(X, split_dates, "strat2",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_strat2 = one_hot(tmp_pd)
                output = pd.concat([output,output_strat2],sort = True, axis = 1)
                tmp_pd = pd.DataFrame(index = X.index)
            #generate strat3 Close
            if activation_params["strat3_close"]:
                act = copy(temp_act)
                act['strat3_close'] = True
                comb = list(range(5,51,2))
                comb = [[com] for com in comb]
                tmp_pd = parallel_process(X, split_dates, "strat3_close",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_strat3 = one_hot(tmp_pd)
                output = pd.concat([output,output_strat3],sort = True, axis = 1)
                tmp_pd = pd.DataFrame(index = X.index)
            #generate strat5
            if activation_params["strat5"]:
                print("**********strat5********")
                act = copy(temp_act)
                act['strat5'] = True
                comb = list(range(5,51,2))
                comb = [[com] for com in comb]
                tmp_pd = parallel_process(X, split_dates, "strat5",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                #output_strat3 = one_hot(tmp_pd)
                output = pd.concat([output,tmp_pd],sort = True, axis = 1)
                tmp_pd = pd.DataFrame(index = X.index)
            #generate strat7
            if activation_params["strat7"]:
                act = copy(temp_act)
                act['strat7'] = True
                limiting_factor = np.arange(1.8,2.45,0.1)
                window = list(range(10,51,2))
                comb = product(window,limiting_factor)
                tmp_pd = parallel_process(X, split_dates, "strat7",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_strat7 = one_hot(tmp_pd)
                output = pd.concat([output,output_strat7],sort = True, axis = 1)
                tmp_pd = pd.DataFrame(index = X.index)
            #generate strat9
            if activation_params["strat9"]:
                act = copy(temp_act)
                act['strat9'] = True
                comb = list(permutations(list(range(10,51,2)),3))
                tmp = parallel_process(X, split_dates, "strat9",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_strat9 = one_hot(tmp)
                output = pd.concat([output,output_strat9],sort = True, axis = 1)
                
            #generate strat trend_! 
            if activation_params["trend_1"]:
                print("**********trend_1********")
                act = copy(temp_act)
                act['trend_1'] = True
                comb = [[1],[3],[6]]
                tmp = parallel_process(X, split_dates, "trend_1",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_trend1 = one_hot(tmp)
                output = pd.concat([output,output_trend1],sort = True, axis = 1)
                
                

            #generate strat6
            if gt+"_High" in cols and gt+"_Low" in cols and activation_params["strat6"]:
                act = copy(temp_act)
                act['strat6'] = True
                limiting_factor = np.arange(1.8,2.45,0.1)
                window = list(range(10,51,2))
                comb = product(window,limiting_factor)
                tmp = parallel_process(X, split_dates, "strat6",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_strat6 = one_hot(tmp)
                output = pd.concat([output,output_strat6],sort = True, axis = 1)
            #generate strat4
            if gt+"_High" in cols and gt+"_Low" in cols and activation_params["strat4"]:
                print("*********strat4********")
                act = copy(temp_act)
                act['strat4'] = True
                limiting_factor = np.arange(1.8,2.45,0.1)
                window = list(range(10,51,2))
                comb = product(window,limiting_factor)
                tmp = parallel_process(X, split_dates, "strat4",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_strat4 = one_hot(tmp)
                output = pd.concat([output,output_strat4],sort = True, axis = 1)
            
    X = pd.concat([X,output],axis = 1, sort = True)
            
    return X

# remove columns that hold the original values of Volume, OI, exchange rate and PVT
def remove_unused_columns_v1(time_series,org_cols):
    for col in copy(time_series.columns):
        if "_Volume" in col or "_OI" in col or "CNYUSD" in col or "_PVT" in col:
            time_series = time_series.drop(col, axis = 1)
            if col in org_cols:
                org_cols.remove(col)
    return time_series, org_cols

# remove a list of columns with an additional label
def remove_unused_columns_v2(time_series,org_cols):
    org_cols.append("Label")
    for col in copy(time_series.columns):
        if col in org_cols:
            time_series = time_series.drop(col, axis = 1)
    return time_series,org_cols


def remove_unused_columns_v3(time_series,org_cols,ground_truth):
    print("#####################remove_unused_columns_v3#####################")
    org_cols.append("Label")
    target = ground_truth[0].split('_')[-2]
    for col in copy(time_series.columns):

        if col in org_cols:
            if col!="DXY":
                time_series = time_series.drop(col, axis = 1)
        else:
            if "Spread" in col or ("nVol" in col and ("LME" not in col or target not in col)) or ("nOI" in col and ("LME" not in col or  target not in col)):
                time_series = time_series.drop(col,axis = 1)
    print(time_series.columns)       
    return time_series,org_cols

#this function is to scale the data use the standardscaler
def scaling_v1(X,train_end):
    """
    X:which equals the timeseries
    train_end:string which we choose to define the time range
    """
    scaler = preprocessing.StandardScaler()
    scaler.fit(X.iloc[:train_end,].values)
    X = pd.DataFrame(scaler.transform(X), index = X.index, columns = X.columns)
    return X

# scale all columns except for those in cat_cols
def scaling_v2(X,train_end, cat_cols):
    """
    X:which equals the timeseries
    train_end:string which we choose to define the time range
    cat_cols:columns that are not to be scaled
    """
    scaler = preprocessing.StandardScaler()
    cols = list(set(X.columns)-set(cat_cols))

    data = X[cols]
    scaler.fit(data.iloc[:train_end].values)
    
    data = pd.DataFrame(scaler.transform(data), index = data.index, columns = cols)
    X[cols] = data
    return X




def construct_v1(time_series, ground_truth, start_ind, end_ind, T, h):
    num = 0
    '''
        convert 2d numpy array of time series data into 3d numpy array, with extra dimension for lags, i.e.
        input of (n_samples, n_features) becomes (n_samples, T, n_features)
        time_series (2d np.array): financial time series data
        ground_truth (1d np.array): column which is used as ground truth
        start_index (string): string which is the date that we wish to begin including from.
        end_index (string): string which is the date that we wish to include last.
        T (int): number of lags
    '''
    for ind in range(start_ind, end_ind):
        if not time_series.iloc[ind - T + 1: ind + 1].isnull().values.any():
            num += 1
    X = np.zeros([num, T, time_series.shape[1]], dtype=np.float32)
    y = np.zeros([num, 1], dtype=np.float32)
    #construct the data by the time index
    sample_ind = 0
    for ind in range(start_ind, end_ind):
        if not time_series.iloc[ind - T + 1 : ind + 1].isnull().values.any():
            X[sample_ind] = time_series.values[ind - T + 1: ind + 1, :]
            y[sample_ind, 0] = ground_truth.values[ind]
            sample_ind += 1
    

    return X,y

def construct_v2(time_series, ground_truth, start_ind, end_ind, T, h):
    print("#########################construct_v2############################")
    num = 0
    '''
        convert 2d numpy array of time series data into 3d numpy array, with extra dimension for lags, i.e.
        input of (n_samples, n_features) becomes (n_samples, T, n_features)
        time_series (2d np.array): financial time series data
        ground_truth (1d np.array): column which is used as ground truth
        start_index (string): string which is the date that we wish to begin including from.
        end_index (string): string which is the date that we wish to include last.
        T (int): number of lags
        Considering the auto-correlaiton between features will weaken the power of XGBoost Model,
        lag will be set as discrete time points,like lag1,lag5,lag10, rather that consecutive period,
        like lag1-lag10.
    '''
    lags = [x for x in range(0,T+1) if x%5==0]
    for ind in range(start_ind, end_ind):
        if not time_series.iloc[[ind-x for x in lags]].isnull().values.any():
            num += 1
    X = np.zeros([num, len(lags), time_series.shape[1]], dtype=np.float32)
    y = np.zeros([num, 1], dtype=np.float32)
    #construct the data by the time index
    sample_ind = 0
    for ind in range(start_ind, end_ind):
        if not time_series.iloc[[ind-x for x in lags]].isnull().values.any():
            X[sample_ind] = time_series.values[[ind-x for x in lags], :]
            y[sample_ind, 0] = ground_truth.values[ind]
            sample_ind += 1
    

    return X,y

def construct_v1_ex2(time_series, ground_truth, start_ind, end_ind, T, h):
    num = 0
    '''
        convert 2d numpy array of time series data into 3d numpy array, with extra dimension for lags, i.e.
        input of (n_samples, n_features) becomes (n_samples, T, n_features)
        time_series (2d np.array): financial time series data
        ground_truth (1d np.array): column which is used as ground truth
        start_index (string): string which is the date that we wish to begin including from.
        end_index (string): string which is the date that we wish to include last.
        T (int): number of lags
    '''
    for ind in range(start_ind, end_ind):
        if not time_series.iloc[ind - T + 1: ind + 1].isnull().values.any():
            num += 1
    X = np.zeros([num, T, time_series.shape[1]], dtype=np.float32)
    y = np.zeros([num, 1], dtype=np.float32)
    #construct the data by the time index
    sample_ind = 0
    for ind in range(start_ind, end_ind):
        if not time_series.iloc[ind - T + 1 : ind + 1].isnull().values.any():
            X[sample_ind] = time_series.values[ind - T + 1: ind + 1, :]
            sample_ind += 1
    
    indices_to_keep = y.nonzero()[0]
    y = y[indices_to_keep,:]
    X = X[indices_to_keep,:]

    
    return X,y





def rolling_half_year(start_date,end_date,length):
    '''
        creates the split dates array which rolls forward every 6 months
        Input
        start_date(str) : date to start creating split_dates
        end_date(str)   : date to stop creating split_dates
        length(int)     : number of years for training set
        Output
        split_dates(list)   : list of list. Each list holds 3 dates, train start, val start, and test start.
    '''
    start_year = start_date.split("-")[0]
    end_year = end_date.split("-")[0]
    split_dates = []

    for year in range(int(start_year),int(end_year)+1):
        split_dates.append([str(year)+"-01-01",str(int(year)+length)+"-01-01",str(int(year)+length)+"-07-01"])
        split_dates.append([str(year)+"-07-01",str(int(year)+length)+"-07-01",str(int(year)+length+1)+"-01-01"])
    
    while split_dates[0][0] < start_date:
        del split_dates[0]
    
    while split_dates[-1][2] > end_date:
        del split_dates[-1]
    
    return split_dates
    
    
    
    

def reset_split_dates(time_series, split_dates):
    '''
    change split_dates so that they are within the time_series index list
    '''
    split_dates[0] = time_series.index[time_series.index.get_loc(split_dates[0], method = 'bfill')]
    split_dates[1] = time_series.index[time_series.index.get_loc(split_dates[1], method = 'bfill')]
    split_dates[2] = time_series.index[time_series.index.get_loc(split_dates[2], method = 'ffill')]  
    return split_dates
    
    
    

def construct_keras_data(time_series, ground_truth_index, sequence_length):
    """
    data process
    
    Arguments:
    time_series -- DataFrame of raw data
    ground_truth_index -- index of ground truth in the dataframe, use to form ground truth label
    sequence_length -- An integer of how many days should be looked at in a row
    
    Returns:
    X_train -- A tensor of shape (N, S, F) that will be inputed into the model to train it
    Y_train -- A tensor of shape (N,) that will be inputed into the model to train it--spot price
    X_test -- A tensor of shape (N, S, F) that will be used to test the model's proficiency
    Y_test -- A tensor of shape (N,) that will be used to check the model's predictions
    Y_daybefore -- A tensor of shape (267,) that represents the spot price ,the day before each Y_test value
    unnormalized_bases -- A tensor of shape (267,) that will be used to get the true prices from the normalized ones
    window_size -- An integer that represents how many days of X values the model can look at at once
    """

    #raw_data
    val_date = '2015-01-02'
    tes_date = '2016-01-04'
    val_ind = time_series.index.get_loc(val_date)
    tes_ind = time_series.index.get_loc(tes_date)
    raw_data = time_series.values
    #Convert the file to a list
    data = raw_data.tolist()
    
    #Convert the data to a 3D array (a x b x c) 
    #Where a is the number of days, b is the window size, and c is the number of features in the data file
    result = []
    for index in range(len(data) - sequence_length + 1):
        result.append(data[index: index + sequence_length])

    #Normalizing data by going through each window
    #Every value in the window is divided by the first value in the window, and then 1 is subtracted
    d0 = np.array(result)
    dr = np.zeros_like(d0)
    dr[:,1:,:] = d0[:,1:,:] / d0[:,0:1,:] - 1
    #Keeping the unnormalized prices for Y_test
    #Useful when graphing spot price over time later
    #The first value in the window
    end = int(dr.shape[0] + 1)
    unnormalized_bases_val = d0[val_ind:tes_ind, 0:1, ground_truth_index]
    unnormalized_bases_tes = d0[tes_ind:end, 0:1, ground_truth_index]
    #Splitting data set into training validating and testing data
    split_line = val_ind
    training_data = dr[:int(split_line), :]

    
    #Training Data
    X_train = training_data[:, :-1]
    Y_train = training_data[:, -1,:]
    Y_train = Y_train[:, ground_truth_index]
    X_val = dr[val_ind:tes_ind,:-1]
    Y_val = dr[val_ind:tes_ind,-1]
    Y_val = Y_val[:, ground_truth_index]

    #Testing data
    X_test = dr[tes_ind:, :-1]
    Y_test = dr[tes_ind:, -1]
    Y_test = Y_test[:, ground_truth_index]

    #Get the day before Y_test's price
    Y_daybefore_val = dr[val_ind:tes_ind, -2, :]
    Y_daybefore_val = Y_daybefore_val[:, ground_truth_index]
    Y_daybefore_tes = dr[tes_ind:, -2, :]
    Y_daybefore_tes = Y_daybefore_tes[:, ground_truth_index]
    
    #Get window size and sequence length
    sequence_length = sequence_length
    window_size = sequence_length - 1 #because the last value is reserved as the y value
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_daybefore_val, Y_daybefore_tes, unnormalized_bases_val, unnormalized_bases_tes, window_size
