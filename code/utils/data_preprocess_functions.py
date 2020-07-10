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
from utils.supply_and_demand import *
from sklearn import preprocessing
import json
import scipy.stats as sct


#the function is to deal with the abnormal time_series
def deal_with_abnormal_value(time_series, toggle_dic):
    '''
    input:  toggle_dic  (dict)  :dictionary that dictates whether process are toggled on or off
    '''
    if toggle_dic['max']:
    #deal with the outlier value that is LARGE in OI
        column_list = []
        for column in time_series.columns:
            if "_OI" in column:
                column_list.append(column)
            if "1GQ" in column and toggle_dic["3rd party"]:
                time_series[col] = time_series[column].ffill()

        year_list = list(range(int(time_series.index[0].split("-")[0]),int(time_series.index[-1].split("-")[0])+1))
        month_list = ['01','02','03','04','05','06','07','08','09','10','11','12']
        for column_name in column_list:   
            for year in year_list:
                for month in month_list:
                    start_time = str(year)+'-'+month+'-'+'01'
                    end_time = str(year)+'-'+month+'-'+'31'
                    value_dict = {}
                    value_list=[]
                    temp = copy(time_series.loc[(time_series.index >= start_time)&(time_series.index <= end_time)])
                    if len(temp) == 0 or len(temp[column_name].dropna()) == 0:
                        continue
                    average = np.mean(temp[column_name].dropna())
                    time_series.at[temp[column_name].idxmax(),column_name] = average
                
    if toggle_dic['min']:
    #deal with the minor value in OI
        for column_name in column_list:
            temp = time_series[column_name]
            nsmallest = temp.nsmallest(n = 20).index
            for ind in nsmallest:
                start_time = ind[:-2]+'01'
                end_time = ind[:-2]+'31'
                time_series.at[ind,column_name] = np.mean(time_series.loc[(time_series.index >= start_time)&(time_series.index <= end_time)][column_name])

    if toggle_dic['interpolate']:
        #missing value interpolate
        time_series = time_series.interpolate(axis = 0)

    return time_series

#spot price normalization of 3 classifier
def spot_price_normalization_v1(time_series):
    ans=[]
    spot_price = copy(time_series['spot_price'])
    if type(spot_price)== np.ndarray:
        spot_price = np.log(np.true_divide(spot_price[1:], spot_price[:-1]))
        # scale the data
        spot_price = spot_price * (1.0 / 3.0 / np.nanstd(spot_price))
    else:
        spot_price.values[1:] = np.log(np.true_divide(spot_price.values[1:],
                                                spot_price.values[:-1]))
        # scale the data
        spot_price = spot_price.div(3 * np.nanstd(spot_price.values[1:]))

    spot_price = spot_price.rename("Spot_price")
    ans.append(spot_price)
    return ans    


#this function is to remove missing values from the head of the dataframe
def process_missing_value(time_series,cons_data=1):
    '''
    input:  cons_data  (int)  :integer that dictates how many consecutive acceptable data points are required
    '''
    count = 0
    sta_ind = 0
    for i in range(time_series.shape[0]):
        if not time_series.iloc[i].isnull().values.any():
            count= count + 1
            if sta_ind==0:
                sta_ind = i
        else:
            count = 0
            sta_ind = 0
        if count == cons_data:
            break
    return time_series[sta_ind:].dropna()


#this function is to build the time_feature into the data
def insert_date_into_feature_v1(time_series):
    time_series['month']=[item[1] for item in time_series.index.str.split('-').to_list()]
    time_series['day']=[item[2] for item in time_series.index.str.split('-').to_list()]

    return time_series


#the function is to label the target and rename the result
def labelling_v1(time_series,horizon, ground_truth_columns):
    '''
    input:  horizon  (int)  :horizon to be predicted
            ground_truth_columns (list) :list of columns to be labelled
    '''
    """
    horizon: the time horizon
    ground_truth_columns: the column we predict
    """
    assert ground_truth_columns != []
    ans = []
    for ground_truth in ground_truth_columns:
        labels = copy(time_series[ground_truth])
        labels = labels.shift(-horizon) - labels
        labels = labels > 0
        labels = labels.rename("Label")
        ans.append(labels)
    return ans

def labelling_v2(time_series,horizon, ground_truth_columns):
    """
    horizon: the time horizon
    ground_truth_columns: the column we predict
    """
    assert ground_truth_columns != []
    ans=[]
    for ground in ground_truth_columns:
        labels = copy(time_series[ground])

        price_changes = labels.shift(-horizon) - labels
        labels = price_changes.divide(labels)

        # scaling the label with standard division
        print(np.nanstd(labels.values))
        labels = labels.div(3 * np.nanstd(labels.values))

        labels = labels.rename("Label")
        ans.append(labels)
    return ans

def labelling_v3(time_series,horizon, ground_truth_columns):
    """
    horizon: the time horizon
    ground_truth_columns: the column we predict
    """
    assert ground_truth_columns != []
    ans=[]
    for ground in ground_truth_columns:
        labels = copy(time_series[ground].to_frame())
        labels.columns = ["Regression Label"]
        price_changes = labels.shift(-horizon) - labels
        labels['Label'] = price_changes.divide(labels)
        print(labels)
        ans.append(labels)

    return ans

#we use this function to make the data normalization
def normalize_without_1d_return(time_series,train_end, params):
    """
    input   train_end: string which we use to define the range we use to train
            params: A dictionary we use to feed the parameter
    """
    ans = {"nVol":False,"nSpread":False,"nEx":False}
    
    cols = time_series.columns.values.tolist()
    ex = False
    if "CNYUSD" in cols:
        print("Considering Exchange Rate")
        ex = True

    for col in cols:

        #use the normalize_OI function to deal with the OI
        if col[:-2]+"OI" == col:
            print("Normalizing OI:"+"=>".join((col,col[:-2]+"nOI")))
            time_series[col[:-2]+"nOI"] = normalize_OI(copy(time_series[col]),train_end,strength = params['strength'], both = params['both'])

        #use the normalize_volume function to deal with the volume
        if col[:-6]+"Volume" == col:
            setting = col[:-6]
            if setting+"OI" in cols:
                ans["nVol"] = True
                print("Normalizing Volume:"+"=>".join((col,setting+"OI")))
                time_series[setting+"nVolume"] = normalize_volume(copy(time_series[col]), train_end = train_end, OI = copy(time_series[setting+"OI"]),
                                                        len_ma = params['len_ma'],version = params['vol_norm'], 
                                                        strength = params['strength'], both = params['both'])

        #use the normalize_3mspot_spread function to create 3 month close to spot spread
        if col[:-5]+"Close" == col:
            setting = col[:-5]
            if setting+"Spot" in cols:
                ans["nSpread"] = True
                print("Normalizing Spread:"+"=>".join((col,setting+"Spot")))
                time_series[setting+"n3MSpread"] = normalize_3mspot_spread(copy(time_series[col]),copy(time_series[setting+"Spot"]),
                                                                len_update=params['len_update'],
                                                                version = params['spot_spread_norm'])

        #Cross Exchange Spread
        if "SHFE" == col[:4] and "Close" == col[-5:] and ex:
            metal = col.split("_")[1]
            if "_".join(("LME",metal,"Spot")) in cols:
                ans["nEx"] = True
                print("+".join((col,"_".join(("LME",metal,"Spot"))))+"=>"+"_".join(("SHFE",metal,"nEx3MSpread")))
                time_series["_".join(("SHFE",metal,"nEx3MSpread"))] = normalize_3mspot_spread_ex(copy(time_series["_".join(("LME",metal,"Spot"))]),
                                                                                    copy(time_series[col]),copy(time_series["CNYUSD"]),
                                                                                    len_update=params['len_update'],
                                                                                    version = params['ex_spread_norm'])
            if "_".join(("LME",metal,"Close")) in cols:
                ans["nEx"] = True
                print("+".join((col,"_".join(("LME",metal,"Close"))))+"=>"+"_".join(("SHFE",metal,"nEx3MSpread")))
                time_series["_".join(("SHFE",metal,"nExSpread"))] = normalize_3mspot_spread_ex(copy(time_series["_".join(("LME",metal,"Close"))]),
                                                                                    copy(time_series[col]),copy(time_series["CNYUSD"]),
                                                                                    len_update=params['len_update'],
                                                                                    version = params['ex_spread_norm'])

        if "Spot" in col:
            if "TPSpread" in params.keys():
                #Third Party Prediction Normalization
                if params["TPSpread"] == "v1":
                    time_series["_".join([col[:-5],"TPDirection"])] = normalize_prediction(copy(time_series[col]),copy(time_series['METF'+col[4]+"3 1GQ"]), version = params['TPSpread'])
                elif params["TPSpread"] == "v2":
                    time_series["_".join([col[:-5],"TPSpread"])],time_series["_".join([col[:-5],"TPDirection"])] = normalize_prediction(copy(time_series[col]),copy(time_series['METF'+col[4]+"3 1GQ"]), version = params['TPSpread'])
                elif params["TPSpread"] == "v3":
                    time_series["_".join([col[:-5],"TPSpread"])],time_series["_".join([col[:-5],"TPDirection"])] = normalize_prediction(copy(time_series[col]),copy(time_series['METF'+col[4]+"3 1GQ"]), version = params['TPSpread'])
            
            if "STL" in params.keys():
                #STL Decomposition
                time_series[col[:-4]+"Trend"] = STL_decomposition_trend(time_series[col])
                time_series[col[:-4]+"Season"] = STL_decomposition_seasonal(time_series[col])
    
    return time_series, ans

#This function is for one-hot encoding.
def one_hot(dataframe):
    output = pd.DataFrame(index = dataframe.index)
    for col in dataframe.columns:
        output[col+'_positive'] = pd.Series(index = dataframe.index,data = [0]*len(dataframe))
        output[col+'_negative'] = pd.Series(index = dataframe.index,data = [0]*len(dataframe))
        output[col+'_positive'].loc[dataframe[col]==1] = 1
        output[col+'_negative'].loc[dataframe[col]==-1] = 1
        
    return output

def technical_indication(time_series,train_end,params,ground_truth_columns, activation):
    """
    input   train_end: string which we use to define the range we use to train
            params: A dictionary we use to feed the parameter
            activation: A dictionary which dictates which indicators are active
    """
    print('====================================technical indicator========================================')
    cols = time_series.columns.values.tolist()
    if activation['Index']:
        if 'COMEX_GC_lag1_Close' in cols and 'COMEX_SI_lag1_Close' in cols and 'COMEX_PA_lag1_Close' in cols and 'COMEX_PL_lag1_Close' in cols:
            time_series["COMEX"] = time_series.loc[:,["COMEX_GC_lag1_Close","COMEX_SI_lag1_Close","COMEX_PA_lag1_Close","COMEX_PL_lag1_Close"]].mean(axis = 1)
            cols = list(set(cols) - set(['COMEX_GC_lag1_Close','COMEX_SI_lag1_Close','COMEX_PA_lag1_Close','COMEX_PL_lag1_Close'])) +["COMEX"]
        if 'SHFE_RT_Close' in cols and 'SHFE_Al_Close' in cols:
            time_series["SHFE"] = time_series.loc[:,["SHFE_RT_Close","SHFE_Al_Close"]].mean(axis = 1)
            cols = list(set(cols) - set(['SHFE_RT_Close','SHFE_Al_Close']))+["SHFE"]
        if 'DCE_AC_Close' in cols and 'DCE_AK_Close' in cols and 'DCE_AE_Close' in cols:
            time_series["DCE"] = time_series.loc[:,["DCE_AC_Close","DCE_AK_Close","DCE_AE_Close"]].mean(axis = 1)
            cols = list(set(cols) - set(['DCE_AC_Close','DCE_AK_Close','DCE_AE_Close']))+['DCE']
        if 'SHCOMP' in cols and 'SHSZ300' in cols and 'HSI' in cols:
            time_series["China"] = time_series.loc[:,["SHCOMP","SHSZ300","HSI"]].mean(axis = 1)
            cols = list(set(cols) - set(['SHCOMP',"SHSZ300","HSI"]))+["China"]

    for col in cols:
        setting = col[:-5]
        ground_truth = ground_truth_columns[0][4:6]
        org_col_trigger_condition = setting+"Close" == col or setting+'_Spot' == col 
        col_trigger_condition = org_col_trigger_condition if not activation['Index'] else \
                             (setting+"Close" == col and ground_truth in setting)  or (len(col.split('_')) == 1 and col != "CNYUSD")
        if col_trigger_condition:
            if activation["EMA"] and "Win_EMA" in params.keys():
                for i in range(len(params['Win_EMA'])):
                    col_name = col+"_EMA"+str(params['Win_EMA'][i]) if len(params["Win_EMA"]) > 1 else col+"_EMA"
                    time_series[col_name] = ema(copy(time_series[col]),params['Win_EMA'][i])
                    if activation["EMA"] > 1:
                        time_series[col+"_WMA"+str(params['Win_EMA'][i])] = wma(copy(time_series[col]),params['Win_EMA'][i])
            
            if activation["Bollinger"] and "Win_Bollinger" in params.keys():
                for i in range(len(params['Win_Bollinger'])):
                    col_name = col+"_bollinger"+str(params['Win_Bollinger'][i]) if len(params["Win_Bollinger"]) > 1 else col+"_bollinger"
                    if activation["Bollinger"] == 1:
                        time_series[col_name] = bollinger(copy(time_series[col]),params['Win_Bollinger'][i])
                    elif activation["Bollinger"] == 2 and  org_col_trigger_condition:
                        time_series[col_name] = rrange = time_series[col].rolling(params['Win_Bollinger'][i]).std()
            
            if activation["MOM"] and "Win_MOM" in params.keys():
                for i in range(len(params['Win_MOM'])):
                    col_name = col+"_Mom"+str(params['Win_MOM'][i]) if len(params["Win_MOM"]) > 1 else col+"_PPO"
                    time_series[col_name] = mom(copy(time_series[col]),params['Win_MOM'][i])
            
            if activation["PPO"] and "PPO_Fast" in params.keys():
                for i in range(len(params['PPO_Fast'])):
                    col_name = col+"_PPO"+str(params['PPO_Fast'][i]) if len(params["PPO_Fast"]) > 1 else col+"_PPO"
                    time_series[col_name] = ppo(copy(time_series[col]),params['PPO_Fast'][i],params['PPO_Slow'][i])
            
            if activation["RSI"]:
                if "Win_RSI" not in params.keys():
                     time_series[col+"_RSI"] = rsi(copy(time_series[col]))
                else:
                    for i in range(len(params['Win_RSI'])):
                        time_series[col+"_RSI"+str(params['Win_RSI'][i])] = rsi(copy(time_series[col]),params['Win_RSI'][i])
                
            if setting+"Close" == col and setting+"Volume" in cols:
                if activation["PVT"]:
                    print("+".join((col,setting+"Volume"))+"=>"+"+".join((setting+"PVT",setting+"divPVT")))
                    time_series[setting+"PVT"] = pvt(copy(time_series.index),copy(time_series[col]),copy(time_series[setting+"Volume"]))
                    time_series[setting+"divPVT"] = divergence_pvt(copy(time_series[col]),copy(time_series[setting+"Volume"]),train_end, 
                                                                params = params)
            
            if setting + 'Close' == col and setting+'High' in cols and setting+'Low' in cols:
                if activation["NATR"] and "Win_NATR" in params.keys():
                    for i in range(len(params['Win_NATR'])):    
                        col_name = setting+"NATR"+str(params['Win_NATR'][i]) if len(params["Win_NATR"]) > 1 else setting+"NATR"
                        time_series[col_name] = natr(time_series[setting+"High"],time_series[setting+"Low"],time_series[col],params['Win_NATR'][i])
                
                if activation["CCI"] and "Win_CCI" in params.keys():
                    for i in range(len(params['Win_CCI'])):
                        time_series[setting+'CCI'+str(params['Win_CCI'][i])] = cci(time_series[setting+"High"],time_series[setting+"Low"],time_series[col],params['Win_CCI'][i])
                    
                if activation["VBM"] and "Win_VBM" in params.keys():
                    if "v_VBM" in params.keys():
                        for i in range(len(params['Win_VBM'])):    
                            time_series[setting+'VBM'+str(params['Win_VBM'][i])] = VBM(time_series[setting+"High"],time_series[setting+"Low"],time_series[col],params['Win_VBM'][i],params['v_VBM'][i])
                    else:
                        for i in range(len(params['Win_VBM'])): 
                            time_series[setting+'VBM'] = vbm(time_series[setting+"High"],time_series[setting+"Low"],time_series[col],params['Win_VBM'][i])
                
                if activation["ADX"] and "Win_ADX" in params.keys():
                    for i in range(len(params['Win_ADX'])):
                        time_series[setting+'ADX'+str(params['Win_ADX'][i])] = ADX(time_series[setting+"High"],time_series[setting+"Low"],time_series[col],params['Win_ADX'][i])
                    
                if activation["sar"]:
                    time_series[setting+'SAR'] = sar(time_series[setting+"High"],time_series[setting+"Low"],time_series[col],params['acc_initial'],params['acc_maximum'])
        
                if activation["SAR"]:
                    time_series[setting+'SAR'] = SAR(time_series[setting+"High"],time_series[setting+"Low"],time_series[col],params['acc_initial'],params['acc_maximum'])
        
        if setting+"_Open" == col:
            if setting+'High' in cols and setting+'Low' in cols:
                if activation["VSD"] and "Win_VSD" in params.keys():
                    for i in range(len(params['Win_VSD'])):
                        time_series[setting+"VSD"+str(params['Win_VSD'][i])] = vsd(time_series[setting+"High"],time_series[setting+"Low"],time_series[col],params['Win_VSD'][i])

                
    return time_series

#process supply and demand data
def process_supply_and_demand(time_series,params):
    cols = time_series.columns.values.tolist()
    for gt in ["LME_Al","LME_Co","LME_Le","LME_Ni","LME_Ti","LME_Zi"]:
        if gt+"_Supply" in cols and gt+"_Demand" in cols:
            time_series[gt+"_SD_Spread"] = SupplyDemandSpread(time_series[gt+"_Supply"],time_series[gt+"_Demand"],params["Spread"])
    return time_series


#generate strategy signals
def strategy_signal_v1(time_series,split_dates,ground_truth_columns,strategy_params,activation_params,cov_inc,mnm):
    '''
    cov_inc (float) :coverage increment
    mnm     (float) :minimum coverage
    '''
    strat_results = {'sar':{'initial':[],'maximum':[]},'rsi':{'window':[],'upper':[],'lower':[]},'strat1':{'short window':[],"med window":[]},'strat2':{'window':[]},'strat3_high':{'window':[]}, 'strat3_close':{'window':[]},'strat6':{'window':[],'limiting_factor':[]},'strat7':{'window':[],'limiting_factor':[]}, 'strat9':{'SlowLength':[],'FastLength':[],'MACDLength':[]}}
    cols = time_series.columns.values.tolist()
    ground_truth = ground_truth_columns[0]
    gt = ground_truth[:-5]
    tmp_pd = pd.DataFrame(index = time_series.index)
    output = pd.DataFrame(index = time_series.index)
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
            tmp_pd = parallel_process(time_series, split_dates, "strat3_high",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
            output_strat3 = one_hot(tmp_pd)
            output = pd.concat([output,output_strat3],sort = True, axis = 1)
            tmp_pd = pd.DataFrame(index = time_series.index)
        #generate strategy 8
        if gt+"_Spread" == col and activation_params["strat8"]:
            act = copy(temp_act)
            act['strat8'] = True
            limiting_factor = np.arange(1.8,2.45,0.1)
            window = list(range(10,51,2))
            comb = product(window,limiting_factor)
            tmp_pd = parallel_process(time_series, split_dates, "strat8",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
            output_strat8 = one_hot(tmp_pd)
            print(output_strat8.columns)
            output = pd.concat([output,output_strat8],sort = True, axis = 1)
            tmp_pd = pd.DataFrame(index = time_series.index)

        if gt+"_Close" == col:
            setting = col[:-5]
            #generate SAR
            if setting+"High" in cols and setting+"Low" in cols and activation_params['sar']:
                act = copy(temp_act)
                act['sar'] = True
                initial = np.arange(0.01,0.051,0.002)
                mx = np.arange(0.1,0.51,0.02)
                comb = product(initial,mx)
                tmp_pd = parallel_process(time_series, split_dates, "sar",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_sar = one_hot(tmp_pd)
                output = pd.concat([output,output_sar],sort = True, axis = 1)
                tmp_pd = pd.DataFrame(index = time_series.index)
            #generate RSI
            if activation_params['rsi']:
                act = copy(temp_act)
                act['rsi'] = True
                window = list(range(5,51,2))
                upper = list(range(60,91,10))
                lower = list(range(20,51,10))
                comb = product(window,upper,lower)
                tmp = parallel_process(time_series, split_dates, "rsi",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_rsi = one_hot(tmp)
                output = pd.concat([output,output_rsi],sort = True, axis = 1)
            #generate Strat 1
            if activation_params["strat1"]:
                act = copy(temp_act)
                act['strat1'] = True
                short_window = list(range(20,35,2))
                med_window = list(range(50,71,2))
                comb = product(short_window,med_window)
                tmp = parallel_process(time_series, split_dates, "strat1",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_strat1 = one_hot(tmp)
                output = pd.concat([output,output_strat1],sort = True, axis = 1)
            #generate strat2
            if activation_params["strat2"]:
                act = copy(temp_act)
                act['strat2'] = True
                comb = list(range(45,61,2))
                comb = [[com] for com in comb]
                tmp_pd = parallel_process(time_series, split_dates, "strat2",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_strat2 = one_hot(tmp_pd)
                output = pd.concat([output,output_strat2],sort = True, axis = 1)
                tmp_pd = pd.DataFrame(index = time_series.index)
            #generate strat3 Close
            if activation_params["strat3_close"]:
                act = copy(temp_act)
                act['strat3_close'] = True
                comb = list(range(5,51,2))
                comb = [[com] for com in comb]
                tmp_pd = parallel_process(time_series, split_dates, "strat3_close",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_strat3 = one_hot(tmp_pd)
                output = pd.concat([output,output_strat3],sort = True, axis = 1)
                tmp_pd = pd.DataFrame(index = time_series.index)
            #generate strat5
            if activation_params["strat5"]:
                print("**********strat5********")
                act = copy(temp_act)
                act['strat5'] = True
                comb = list(range(5,51,2))
                comb = [[com] for com in comb]
                tmp_pd = parallel_process(time_series, split_dates, "strat5",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                #output_strat3 = one_hot(tmp_pd)
                output = pd.concat([output,tmp_pd],sort = True, axis = 1)
                tmp_pd = pd.DataFrame(index = time_series.index)
            #generate strat7
            if activation_params["strat7"]:
                act = copy(temp_act)
                act['strat7'] = True
                limiting_factor = np.arange(1.8,2.45,0.1)
                window = list(range(10,51,2))
                comb = product(window,limiting_factor)
                tmp_pd = parallel_process(time_series, split_dates, "strat7",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_strat7 = one_hot(tmp_pd)
                output = pd.concat([output,output_strat7],sort = True, axis = 1)
                tmp_pd = pd.DataFrame(index = time_series.index)
            #generate strat9
            if activation_params["strat9"]:
                act = copy(temp_act)
                act['strat9'] = True
                comb = list(permutations(list(range(10,51,2)),3))
                tmp = parallel_process(time_series, split_dates, "strat9",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_strat9 = one_hot(tmp)
                output = pd.concat([output,output_strat9],sort = True, axis = 1)
                
            #generate strat trend_1 
            if activation_params["trend_1"]:
                print("**********trend_1********")
                act = copy(temp_act)
                act['trend_1'] = True
                comb = [[1],[3],[6]]
                tmp = parallel_process(time_series, split_dates, "trend_1",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_trend1 = one_hot(tmp)
                output = pd.concat([output,output_trend1],sort = True, axis = 1)
                
            #generate strat6
            if gt+"_High" in cols and gt+"_Low" in cols and activation_params["strat6"]:
                act = copy(temp_act)
                act['strat6'] = True
                limiting_factor = np.arange(1.8,2.45,0.1)
                window = list(range(10,51,2))
                comb = product(window,limiting_factor)
                tmp = parallel_process(time_series, split_dates, "strat6",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
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
                tmp = parallel_process(time_series, split_dates, "strat4",strat_results,ground_truth,strategy_params,act,cov_inc,comb,mnm)
                output_strat4 = one_hot(tmp)
                output = pd.concat([output,output_strat4],sort = True, axis = 1)
            
    time_series = pd.concat([time_series,output],axis = 1, sort = True)
            
    return time_series

# remove columns that hold the original values of Volume, OI, exchange rate and PVT
def remove_unused_columns(time_series, org_cols, condition):
    '''
        org_cols:   original columns from raw data
        condition:  condition for a column to be deleted
    '''
    for col in copy(time_series.columns):
        if condition(col):
            time_series = time_series.drop(col, axis = 1)
            if col in org_cols:
                org_cols.remove(col)
    return time_series, org_cols

#this function is to scale the data use the standardscaler
def scaling_v1(time_series,train_end):
    """
    train_end:string which we choose to define the time range
    """
    scaler = preprocessing.StandardScaler()
    scaler.fit(time_series.iloc[:train_end,].values)
    time_series = pd.DataFrame(scaler.transform(time_series), index = time_series.index, columns = time_series.columns)
    return time_series

# scale all columns except for those in cat_cols
def scaling_v2(time_series,train_end, cat_cols):
    """
    train_end:string which we choose to define the time range
    cat_cols:categorical columns that are not to be scaled
    """
    scaler = preprocessing.StandardScaler()
    cols = list(set(time_series.columns)-set(cat_cols))

    data = time_series[cols]
    scaler.fit(data.iloc[:train_end].values)
    
    data = pd.DataFrame(scaler.transform(data), index = data.index, columns = cols)
    time_series[cols] = data
    return time_series

#

def construct(time_series, ground_truth, start_ind, end_ind, lags, h):
    num = 0
    '''
        convert 2d numpy array of time series data into 3d numpy array, with extra dimension for lags, i.e.
        input of (n_samples, n_features) becomes (n_samples, T, n_features)
        time_series (2d np.array): financial time series data
        ground_truth (1d np.array): column which is used as ground truth
        start_index (string): string which is the date that we wish to begin including from.
        end_index (string): string which is the date that we wish to include last.
        lags (array): list of lags
        Considering the auto-correlaiton between features will weaken the power of XGBoost Model,
        lag will be set as discrete time points,like lag1,lag5,lag10, rather that consecutive period,
        like lag1-lag10.
    '''
    for ind in range(start_ind, end_ind):
        if not time_series.iloc[[ind-x for x in lags]].isnull().values.any():
            num += 1
    X = np.zeros([num, len(lags), time_series.shape[1]], dtype=np.float32)
    y = np.zeros([num, len(ground_truth.columns.values.tolist())], dtype=np.float32)
    #construct the data by the time index
    sample_ind = 0
    for ind in range(start_ind, end_ind):
        if not time_series.iloc[[ind-x for x in lags]].isnull().values.any():
            X[sample_ind] = time_series.values[[ind-x for x in lags], :]
            if len(ground_truth.columns.values.tolist()) > 1:
                y[sample_ind, 0] = ground_truth.values[ind][0]
                y[sample_ind, 1] = ground_truth.values[ind][1]
            else:
                y[sample_ind, 0] = ground_truth.values[ind]
            sample_ind += 1
    

    return X,y

#change split_dates so that they are within the time_series index list
def reset_split_dates(time_series, split_dates):
    split_dates[0] = time_series.index[time_series.index.get_loc(split_dates[0], method = 'bfill')]
    split_dates[1] = time_series.index[time_series.index.get_loc(split_dates[1], method = 'bfill')]
    split_dates[2] = time_series.index[time_series.index.get_loc(split_dates[2], method = 'ffill')]  
    return split_dates
    


