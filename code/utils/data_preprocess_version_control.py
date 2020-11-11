'''
    This file controls the preprocess
    Each feature version has a unique preprocess setup, which is controlled by version for each individual step of the process
'''
import os
import sys
import json
import utils.data_preprocess_functions as dpf
from utils.normalize_feature import log_1d_return, fractional_diff, rel_to_open
from copy import copy

'''
    This file contains the functions which identify the versions of preprocess subfunctions that are to be used for a particular feature version
'''


#Generates the corresponding version of each step based on the feature version
def generate_version_params(version):
    '''
        input:  version : a string which refers to the feature version of data preprocessing required
        output: ans     : a dictionary that holds the required version for each process within process data
    '''
    #when experimenting with tweaks under a certain version, we add a ex(num) to denote their differences
    ver = version.split("_")
    v = ver[0]
    ex = ver[1] if len(ver) > 1 else None

    #for understanding the difference between feature versions, please go to https://wiki.alphien.com/ALwiki/Feature_Version
    if v == "v3":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":None, "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":None,
                "normalize_without_1d_return": "v1", "technical_indication":"v2", "supply_and_demand":None,
                "remove_unused_columns":"v4", "price_normalization":"v1", "scaling":"v1",
                "construct":"v1"}
    elif v == "v5":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":None, "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":None,
                "normalize_without_1d_return": "v1", "technical_indication":"v1", "supply_and_demand":None,
                "remove_unused_columns":"v1", "price_normalization":"v1", "scaling":"v1",
                "construct":"v1"}
    elif v == "v7":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":None, "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":None,
                "normalize_without_1d_return": "v1", "technical_indication":"v2", "supply_and_demand":None,
                "remove_unused_columns":"v1", "price_normalization":"v1", "scaling":"v1",
                "construct":"v1"}
    elif v == "v9":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":"v1", "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":"v1",
                "normalize_without_1d_return": None, "technical_indication":None, "supply_and_demand":None,
                "remove_unused_columns":"v2", "price_normalization":None, "scaling":None,
                "construct":"v1"}
    elif v == "v10":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":"v2", "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":"v1",
                "normalize_without_1d_return": None, "technical_indication":None, "supply_and_demand":None,
                "remove_unused_columns":"v2", "price_normalization":None, "scaling":None,
                "construct":"v1"}
    elif v == "v11":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":"v3", "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":"v1",
                "normalize_without_1d_return": None, "technical_indication":None, "supply_and_demand":None,
                "remove_unused_columns":"v2", "price_normalization":None, "scaling":None,
                "construct":"v1"}
    elif v == "v12":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":"v4", "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":"v1",
                "normalize_without_1d_return": None, "technical_indication":None, "supply_and_demand":None,
                "remove_unused_columns":"v2", "price_normalization":None, "scaling":None,
                "construct":"v1"}
    elif v == "v14":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":"v5", "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":"v1",
                "normalize_without_1d_return": "v1", "technical_indication":None, "supply_and_demand":None,
                "remove_unused_columns":"v3", "price_normalization":"v2", "scaling":None,
                "construct":"v2"}
    elif v == "v16":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":"v2", "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v2", "process_missing_value":"v1", "strategy_signal":"v1",
                "normalize_without_1d_return": None, "technical_indication":None, "supply_and_demand":None,
                "remove_unused_columns":"v2", "price_normalization":None, "scaling":None,
                "construct":"v1"}
    elif v == "v18":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":"v6", "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":"v1",
                "normalize_without_1d_return": None, "technical_indication":None, "supply_and_demand":None,
                "remove_unused_columns":"v2", "price_normalization":None, "scaling":None,
                "construct":"v1"}
    elif v == "v20":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":"v7", "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":"v1",
                "normalize_without_1d_return": None, "technical_indication":None, "supply_and_demand":None,
                "remove_unused_columns":"v2", "price_normalization":None, "scaling":None,
                "construct":"v1"}
    elif v == "v22":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":"v8", "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":"v1",
                "normalize_without_1d_return": "v1", "technical_indication":None, "supply_and_demand":None,
                "remove_unused_columns":"v3", "price_normalization":"v2", "scaling":None,
                "construct":"v1"}
    elif v == "v23":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v2","generate_strat_params":None, "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":None,
                "normalize_without_1d_return": None, "technical_indication":"v3", "supply_and_demand":None,
                "remove_unused_columns":"v4", "price_normalization":None, "scaling":"v2",
                "construct":"v3"}
    elif v == "v24":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v2","generate_strat_params":None, "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":None,
                "normalize_without_1d_return": None, "technical_indication":"v3", "supply_and_demand":None,
                "remove_unused_columns":"v4", "price_normalization":None, "scaling":"v2",
                "construct":"v3"}
    elif v == "v28":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v2","generate_strat_params":"v10", "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":"v1",
                "normalize_without_1d_return":None, "technical_indication":"v3", "supply_and_demand":None,
                "remove_unused_columns":"v4", "price_normalization":None, "scaling":"v2",
                "construct":"v3"}
    elif v == "v30":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v2","generate_strat_params":"v11", "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":"v1",
                "normalize_without_1d_return":None, "technical_indication":"v3", "supply_and_demand":None,
                "remove_unused_columns":"v4", "price_normalization":None, "scaling":"v2",
                "construct":"v3"}
    elif v == "v26":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v2","generate_strat_params":None, "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v2", "process_missing_value":"v1", "strategy_signal":None,
                "normalize_without_1d_return":None, "technical_indication":"v3", "supply_and_demand":None,
                "remove_unused_columns":"v4", "price_normalization":None, "scaling":"v2",
                "construct":"v1"}
    elif v == "v31":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":None, "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":None,
                "normalize_without_1d_return": "v1", "technical_indication":"v2", "supply_and_demand":"v1",
                "remove_unused_columns":"v4", "price_normalization":"v1", "scaling":"v1",
                "construct":"v1"}
    elif v == "v32":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":None, "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":None,
                "normalize_without_1d_return": "v1", "technical_indication":"v2", "supply_and_demand":"v1",
                "remove_unused_columns":"v4", "price_normalization":"v1", "scaling":"v1",
                "construct":"v1"}
    elif v == "v33":
        ans = { "generate_norm_params":"v2","generate_tech_params":"v1","generate_strat_params":None, "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v3", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":None,
                "normalize_without_1d_return": "v2", "technical_indication":"v2", "supply_and_demand":None,
                "remove_unused_columns":"v4", "price_normalization":"v1", "scaling":"v1",
                "construct":"v1"}
    elif v == "v35":
        ans = { "generate_norm_params":"v3","generate_tech_params":"v1","generate_strat_params":None, "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v3", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":None,
                "normalize_without_1d_return": "v2", "technical_indication":"v2", "supply_and_demand":None,
                "remove_unused_columns":"v4", "price_normalization":"v1", "scaling":"v1",
                "construct":"v1"}
    elif v == "v37":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":None, "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":None,
                "normalize_without_1d_return": "v3", "technical_indication":"v2", "supply_and_demand":None,
                "remove_unused_columns":"v5", "price_normalization":"v1", "scaling":"v1",
                "construct":"v1"}
    elif v == "v43":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":None, "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":None,
                "normalize_without_1d_return": "v1", "technical_indication":"v2", "supply_and_demand":None,
                "remove_unused_columns":"v4", "price_normalization":"v2", "scaling":"v1",
                "construct":"v1"}
    elif v == "r2":
        ans = { "generate_norm_params":"v1","generate_tech_params":"v3","generate_strat_params":None, "generate_SD_params":"v1",
                "deal_with_abnormal_value":"v2", "labelling":"v3", "process_missing_value":"v1", "strategy_signal":None,
                "normalize_without_1d_return": "v1", "technical_indication":"v4", "supply_and_demand":None,
                "remove_unused_columns":"v6", "price_normalization":"v3", "scaling":None,
                "construct":"v4"}
    
    elif v == "c1":
        ans = { "generate_norm_params":"v1","generate_tech_params":None,"generate_strat_params":None, "generate_SD_params":None,
                "deal_with_abnormal_value":"v4", "labelling":"v4", "process_missing_value":"v1", "strategy_signal":None,
                "normalize_without_1d_return": None, "technical_indication":None, "supply_and_demand":None,
                "remove_unused_columns":"v7", "price_normalization":None, "scaling":"v3",
                "construct":"v4"}
    print(ans)
    return ans

def generate_norm_params(version, date):
    '''
        generates parameter values in normalization
        input:  version: (string) version of normalization parameters to generate
                date: (int) whether date is a data column
    '''
    if version == "v1":
        # default normalization parameters
        return {'vol_norm':'v1','ex_spread_norm':'v1','spot_spread_norm':'v1','len_ma':5,'len_update':30,'both':3,'strength':0.01,'date':date > 0}
    elif version == "v2":
        # only used for third party feature versions (normalize third party data)
        return {'vol_norm':'v1','ex_spread_norm':'v1','spot_spread_norm':'v1','len_ma':5,'len_update':30,'both':3,'strength':0.01,'date':date > 0, "TPSpread":"v"+str(int(version[1])-1)}
    elif version == "v3":
        # only used for STL decomposition experiment
        return {'vol_norm':'v1','ex_spread_norm':'v1','spot_spread_norm':'v1','len_ma':5,'len_update':30,'both':3,'strength':0.01,'date':date > 0, "STL":True}
    else:
        return None
        
def generate_tech_params(version):
    '''
        generates parameter values in technical indicator generation
        input:  version: (string) version of technical indicator parameters to generate
    '''
    if version == "v1":
        return {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':[12],'Win_Bollinger':[22],'PPO_Fast':[12],'PPO_Slow':[26],'Win_NATR':[10],'Win_VBM':[22],'acc_initial':0.02,'acc_maximum':0.2}
    elif version == "v2":
        return {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':[12,26,40,65,125],'Win_Bollinger':[5,10,15,20,30,65],
                               'Win_MOM':[5,10,15,26,40,65,125],'PPO_Fast':[12,22],'PPO_Slow':[26,65],'Win_NATR':[14,26,65,125],'Win_VBM':[12,22],'v_VBM':[26,65],
                               'acc_initial':0.02,'acc_maximum':0.2,'Win_CCI':[12,26,40,65,125],'Win_ADX':[14,26,40,54,125],'Win_RSI':[14,26,40,54,125]}
    elif version == "v3":
        return {'strength':0.01,'both':3,'Win_EMA':[12,26,40,65,125],'Win_Bollinger':[12,26,40,65,125]}
    else:
        return None

def generate_SD_params(version):
    '''
        generates parameter values in supply and demand indicator generation
        input:  version: (string) version of supply and demand indicator parameters to generate
    '''
    if version == "v1":
        return {"Spread":"v2"}
    else:
        return None

def generate_strat_params(ground_truth,horizon,version):
    '''
        generate strategy parameters as previously stored
        input:  ground_truth :  (string)    metal that is being predicted
                horizon   :       (int)       prediction horizon
                version :       (string)    version of strategy parameters to read
    '''
    if version is None:
        return None,None
    
    elif  version == "v1":
        f = "strat_param_v9.conf"
        activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":False,"strat5":False,"strat6":True,"strat7":True,"strat8":False,"strat9":True,"trend_1":False}
    
    elif version == "v2":
        f = "strat_param_v10.conf"
        activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":False,"strat5":False,"strat6":True,"strat7":True,"strat8":False,"strat9":True,"trend_1":False}
    
    elif version == "v3":
        f = "strat_param_v11.conf"
        activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":False,"strat5":False,"strat6":True,"strat7":True,"strat8":False,"strat9":True,"trend_1":False}
    
    elif version == "v4":
        f = "strat_param_v12.conf"
        activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":False,"strat5":False,"strat6":True,"strat7":True,"strat8":False,"strat9":True,"trend_1":False}
    
    elif version == "v5":
        f = "strat_param_v14.conf"
        activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":False,"strat5":False,"strat6":True,"strat7":True,"strat8":True,"strat9":True,"trend_1":False}
    
    elif version == "v6":
        f = "strat_param_v18.conf"
        activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":True,"strat5":True,"strat6":True,"strat7":True,"strat8":False,"strat9":True,"trend_1":False}
    
    elif version == "v7":
        f = "strat_param_v20.conf"
        activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":False,"strat5":False,"strat6":True,"strat7":True,"strat8":False,"strat9":True,"trend_1":True}
    
    elif version == "v8":
        f = "strat_param_v20.conf"
        activation_params = {"sar":True,"rsi":True,"strat1":True,"strat2":True,"strat3_high":True,"strat3_close":True,"strat4":True,"strat5":True,"strat6":True,"strat7":True,"strat8":True,"strat9":True,"trend_1":True}
    
    elif version == "v9":
        f = "strat_param_v14.conf"
        activation_params = {"sar":False,"rsi":False,"strat1":False,"strat2":False,"strat3_high":False,"strat3_close":False,"strat4":False,"strat5":False,"strat6":False,"strat7":False,"strat8":True,"strat9":False,"trend_1":False}
    
    elif version == "v10":
        f = "strat_param_v18.conf"
        activation_params = {"sar":False,"rsi":False,"strat1":False,"strat2":False,"strat3_high":False,"strat3_close":False,"strat4":True,"strat5":True,"strat6":False,"strat7":False,"strat8":False,"strat9":False,"trend_1":False}
    
    elif version == "v11":
        f = "strat_param_v20.conf"
        activation_params = {"sar":False,"rsi":False,"strat1":False,"strat2":False,"strat3_high":False,"strat3_close":False,"strat4":False,"strat5":False,"strat6":False,"strat7":False,"strat8":False,"strat9":False,"trend_1":True}
    
    elif version == "v12":
        f = "strat_param_v22.conf"
        activation_params = {"sar":False,"rsi":False,"strat1":False,"strat2":False,"strat3_high":False,"strat3_close":False,"strat4":False,"strat5":False,"strat6":False,"strat7":False,"strat8":True,"strat9":False,"trend_1":False}
    
    with open(os.path.join(sys.path[0],'exp',f)) as f:
        all_params = json.load(f)
    strat_params = all_params[ground_truth.split("_")[1]][str(horizon)+"d"]
    return strat_params,activation_params
    

def deal_with_abnormal_value(arguments, version):
    time_series = arguments['time_series']
    if version is None:
        return time_series
    elif version == "v1":
        #process abnormally large and small values and interpolate existing data
        toggle_dic = {'max':True,'3rd party':False, 'min':True, 'interpolate':True, 'ffill':False}
    elif version == "v2":
        #process abnormally large values and interpolate existing data
        toggle_dic = {'max':True,'3rd party':False, 'min':False, 'interpolate':True, 'ffill':False}
    elif version == "v3":
        #process abnormally large values and interpolate existing data (3rd party predictions are filled with last known value)
        toggle_dic = {'max':True,'3rd party':True, 'min':False, 'interpolate':True, 'ffill':False}
    elif version == "v4":
        toggle_dic = {'max':False, '3rd party':False, 'min':False, 'interpolate':False, 'ffill':True}
    return dpf.deal_with_abnormal_value(time_series, toggle_dic)

def labelling(arguments, version):
    time_series = arguments['time_series']
    if version == "v1":
        #standard labelling
        return dpf.labelling_v1(time_series, arguments['horizon'], arguments['ground_truth_columns'])
    
    elif version == "v2":
        #construct the torch version
        return dpf.labelling_v2(time_series, arguments['horizon'], arguments['ground_truth_columns'],arguments['split_dates'])

    elif version == "v3":
        #regression
        return dpf.labelling_v3(time_series, arguments['horizon'],
                                arguments['ground_truth_columns'])
    elif version == "v4":
        #competition
        return dpf.labelling_v4(time_series, arguments['horizon'],
                                arguments['ground_truth_columns'], arguments['split_dates'])
    else:
        return None
    

def process_missing_value(arguments, version):
    time_series = arguments['time_series']
    if version == "v1":
        #drop NA rows from head
        return dpf.process_missing_value(time_series)
    else:
        return time_series

def strategy_signal(arguments,version):
    time_series = arguments['time_series']
    if version is None:
        return time_series
    elif version == "v1":
        ts = copy(time_series)
        ts['Label'] = arguments['labels'][0]
        return dpf.strategy_signal_v1(ts,  arguments['split_dates'], arguments['ground_truth_columns'], arguments['strat_params'],arguments['activation_params'],0.1,0.1)

def normalize_without_1d_return(arguments, version):
    time_series = arguments['time_series']
    if version is None:
        return time_series, None
    else:
        #automated normalization for all possible combinations
        return dpf.normalize_without_1d_return(time_series,time_series.index.get_loc(arguments['split_dates'][1]),
                                                arguments['norm_params'])

def technical_indication(arguments, version):
    time_series = arguments['time_series']

    activation = {"Index":False, 'EMA':0,'PVT':False, 'Bollinger':False,'MOM':False,'PPO':False,'RSI':False,'NATR':False,'CCI':False,'VBM':False,'ADX':False,'sar':False,'SAR':False,'VSD':False}
    if version is None:
        return time_series
    if version == "v1":
        # automated generation of divPVT for all possible combinations (only PVT)
        activation['PVT'] = True

    elif version in ["v2","v3"]:
        
        #automated generation of the below technical indicators
        
        #EMA has 3 values, 0 for unactivated, 1 for only EMA, 2 for EMA and WMA activation
        activation['EMA'] = 1 if version == "v2" else 2

        activation['PVT'] = True
        activation['Bollinger'] = 1
        activation["MOM"] = True
        activation['PPO'] = True
        activation['RSI'] = True
        activation['NATR'] = True
        activation['CCI'] = True
        activation['VBM'] = True
        activation['ADX'] = True
        if version == "v2":
            activation['sar'] = True
        elif version == "v3":
            activation['SAR'] = True
        activation['VSD'] = True
    
    elif version in ["v4"]:
        activation["Index"] = True
        activation['EMA'] = 1
        activation["Bollinger"] = 2
    return dpf.technical_indication(time_series,time_series.index.get_loc(arguments['split_dates'][1]),
                                    arguments['tech_params'],arguments['ground_truth_columns'],activation)

def supply_and_demand(arguments, version):
    if version is None:
        return arguments['time_series']
    elif version == "v1":
        return dpf.process_supply_and_demand(arguments['time_series'],arguments['SD_params'])


def remove_unused_columns(arguments, version):
    time_series = arguments['time_series']
    if version is None:
        return time_series
    if version == "v1":
        # remove raw Volume, OI, exchnage rate and PVT
        condition = lambda x : "_Volume" in x or "_OI" in x or "CNYUSD" in x or "_PVT" in x or "_Supply" in x or "_Demand" in x

    elif version == "v2":
        # remove all original columns AND Label column (does not affect Label used for gauging performance)
        arguments['org_cols'].append("Label")
        condition = lambda x : x in arguments['org_cols']

    elif version == "v3":
        # NExT experiment (unrelated to 4E Base Metal Prediction)
        arguments['org_cols'].append("Label")
        target = arguments['ground_truth_columns'][0].split("_")[-2]
        arguments['org_cols'].remove("DXY")
        condition = lambda x: "Spread" in x or ("nVol" in x and ("LME" not in x or target not in x)) or ("nOI" in x and ("LME" not in x or target not in x))
    
    elif version == "v4":
        # remove all data columns not of LME and not of specified metal (ground truth) and raw Volume and OI
        target = arguments['ground_truth_columns'][0][:-5]
        condition = lambda x : (target not in x or "_Volume" in x or "_OI" in x) and (x != "day" or x != "month")
    
    elif version == "v5":
        # remove all data columns not of LME and not of specified metal (ground truth) and raw Volume, OI and Spot Price
        target = arguments['ground_truth_columns'][0][:-5]
        condition = lambda x : (target not in x or "_Volume" in x or "_OI" in x) and (x != "day" or x != "month")
    
    elif version == "v6":
        # remove all data columns not of LME and not of specified metal (ground truth) and raw Volume, OI and Spot Price
        target = arguments['ground_truth_columns'][0][:-5]
        condition = lambda x : ((target not in x and len(x.split('_'))>2) or (target not in x and "LME" in x) or x==target+'_OI' or x==target+'_Volume' or x == target+"_Spot" or 'Spread' in x or 'CNYUSD' == x) and (x != "day" or x != "month")
    
    elif version == "v7":
        # remove all data columns not of LME and not of specified metal (ground truth) and raw Volume, OI and Spot Price
        target = arguments['ground_truth_columns'][0][:-5]
        condition = lambda x : ((target not in x and len(x.split('_'))>2) or (target not in x and "LME" in x) or x==target+'_OI' or x == target+"_Spot" or 'Spread' in x or 'CNYUSD' == x) and (x != "day" or x != "month")
    
    return dpf.remove_unused_columns(time_series,arguments['org_cols'],condition)
    
def price_normalization(arguments, version):
    time_series = arguments['time_series']
    if version is None:
        return time_series
    if version == "v1":
        #daily log returns
        return log_1d_return(time_series,arguments['org_cols'])
    if version == "v2":
        # fractional difference
        return fractional_diff(time_series,arguments['org_cols'])
    if version == "v3":
        #log difference between Close,High,Low to Open
        return rel_to_open(time_series,arguments['org_cols'])

def spot_price_normalization(arguments):
    time_series = arguments['time_series']
    return dpf.spot_price_normalization_v1(time_series)

    
def insert_date_into_feature(arguments):
    time_series = arguments['time_series']
    return dpf.insert_date_into_feature_v1(time_series)

def scaling(arguments, version):
    time_series = arguments['time_series']
    if version is None:
        return time_series
    if version == "v1":
        #standard scaling for all variables
        return dpf.scaling_v1(time_series,time_series.index.get_loc(arguments['split_dates'][1]))
    if version == "v2":
        #scaling without affecting categorical variables
        return dpf.scaling_v2(time_series,time_series.index.get_loc(arguments['split_dates'][1]),arguments['cat_cols'])
    if version == "v3":
        #minmax scaler
        return dpf.scaling_v3(time_series, time_series.index.get_loc(arguments['split_dates'][1]))


def construct(ind, arguments, version):
    time_series = arguments['time_series']
    if version in ["v1","v2","v3"]:
        ground_truth = time_series[ind]["Label"].to_frame()
        if version == "v1":
            lag =range(arguments['lags'])[::-1]
        elif version == "v2":
            lag = [x for x in range(0,arguments['lags']+1) if x%5==0]
        elif version == "v3":
            lag = [0]
    else:
        lag = range(arguments['lags'])[::-1]
        ground_truth = time_series[ind][["Label", "Regression Label"]]
    return dpf.construct(time_series[ind][arguments['all_cols'][ind]], ground_truth, 
                            arguments['start_ind'], arguments['end_ind'], 
                            lag, arguments['horizon'])

