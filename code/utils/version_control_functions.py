from utils.construct_data import *
from utils.read_data import process_missing_value_v3
from utils.normalize_feature import log_1d_return

def generate_version_params(version):
    '''
        input:  version : a string which refers to the version of data preprocessing required
        output: ans     : a dictionary that holds the required version for each process within load data
    '''
    ans = { "generate_norm_params":"v1","generate_tech_params":"v1","generate_strat_params":None,
            "deal_with_abnormal_value":"v2", "labelling":"v1", "process_missing_value":"v1", "strategy_signal":None,
            "normalize_without_1d_return": "v1", "technical_indication":"v1",
            "remove_unused_columns":"v1", "price_normalization":"v1", "scaling":"v1",
            "construct":"v1"}
    ver = version.split("_")
    v = ver[0]
    ex = ver[1] if len(ver) > 1 else None
    if v == "v7" or v=="v3":
        ans['technical_indication'] = "v2"
        if v=="v3":
            ans['remove_unused_columns'] = "v4"
    
    if v in ["v9","v10","v11","v12","v14","v16","v18","v20","v22","v28","v30","v32"]:
        if v == "v9":
            ans["generate_strat_params"]="v1"
        elif v== "v10":
            ans["generate_strat_params"]="v2"
        elif v== 'v11':
            ans["generate_strat_params"]="v3"
        elif v== 'v12':
            ans["generate_strat_params"]="v4"
        elif v== 'v14':
            ans["generate_strat_params"]="v5"
            ans["construct"]="v2"
        elif v=='v16':
            ans["generate_strat_params"]="v2"
            ans['labelling'] = "v2"
        elif v== 'v18':
            ans["generate_strat_params"]="v6"
        elif v== 'v20':
            ans["generate_strat_params"]="v7"
        elif v== 'v22':
            ans["generate_strat_params"]="v8"
        elif v == "v28":
            ans["generate_strat_params"]="v10"
        elif v == "v30":
            ans["generate_strat_params"]="v11"
        elif v == "v32":
            ans["generate_strat_params"]="v12"
        ans['strategy_signal'] = "v1"
        ans["technical_indication"] = None
        
        if v=='v14' or v=='v22':
            ans["remove_unused_columns"] = "v3"
            ans["price_normalization"] = "v2"
        else: 
            ans["remove_unused_columns"] = "v2"
            ans["normalize_without_1d_return"] = None
            ans["price_normalization"] = None
            
        ans["scaling"] = None
    
    if v in["v23","v24","v28","v30","v32"]:
        ans["generate_tech_params"]="v2"
        ans['technical_indication'] = "v3"
        ans['remove_unused_columns'] = "v4"
        ans["construct"]="v3"
        ans["normalize_without_1d_return"] = None
        ans["price_normalization"] = None
        ans["scaling"] = "v2"
    
    if v in ["v26"]:
        ans["generate_tech_params"]="v2"
        ans['technical_indication'] = "v3"
        ans['remove_unused_columns'] = "v4"
        ans["construct"]="v1"
        ans["normalize_without_1d_return"] = None
        ans["price_normalization"] = None
        ans["scaling"] = "v2"
        ans['labelling'] = "v2"        
    
    if ex == "ex1":
        ans['labelling'] = "v1_ex1"
    if ex == "ex2":
        ans['labelling'] = "v1_ex2"
        ans['construct'] = "v1_ex2"
    if ex == "ex3":
        ans['technical_indication'] = ans['technical_indication']+"_ex3"
    print(ans)
    return ans

def generate_norm_params(version,xgboost):
    if version == "v1":
        if xgboost == 1:
            return {'vol_norm':'v1','ex_spread_norm':'v1','spot_spread_norm':'v1','len_ma':5,'len_update':30,'both':3,'strength':0.01,'xgboost':True}
        else:
            return {'vol_norm':'v1','ex_spread_norm':'v1','spot_spread_norm':'v1','len_ma':5,'len_update':30,'both':3,'strength':0.01,'xgboost':False}
        
def generate_tech_params(version):
    if version == "v1":
        return {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
    elif version == "v2":
        return {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':[12,26,40,65,125],'Win_Bollinger':[5,10,15,20,30,65],
                               'Win_MOM':[5,10,15,26,40,65,125],'PPO_Fast':[12,22],'PPO_Slow':[26,65],'Win_NATR':[14,26,65,125],'Win_VBM':[12,22],'v_VBM':[26,65],
                               'acc_initial':0.02,'acc_maximum':0.2,'Win_CCI':[12,26,40,65,125],'Win_ADX':[14,26,40,54,125],'Win_RSI':[14,26,40,54,125]}


def generate_strat_params(ground_truth,steps,version):
    if version is None:
        return None,None
    if version == "v1":
        '''
        generate strategy parameters for v9, which is single metal for coverage increment 0.1, minimum 0.1
        '''
        return generate_strat_params_v1(ground_truth,steps)
    if version == "v2":
        '''
        generate strategy parameters for v10, which is multiple metals for coverage increment 0.1, minimum 0.1
        '''
        return generate_strat_params_v2(ground_truth,steps)
    if version == "v3":
        '''
        generate strategy parameters for v11, which is single metal for coverage increment 1.0, minimum 0.1
        '''
        return generate_strat_params_v3(ground_truth,steps)
    if version == "v4":
        '''
        generate strategy parameters for v12, which is multiple metals for coverage increment 1.0, minimum 0.1
        '''  
        return generate_strat_params_v4(ground_truth,steps)
    if version == "v5":
        '''
            load strategy parameters for preprocessing version 14
        '''
        return generate_strat_params_v5(ground_truth,steps)
    if version == "v6":
        
        '''
            load strategy parameters for preprocessing version 18
        '''
        return generate_strat_params_v6(ground_truth,steps)
    if version == "v7":
        '''
            load strategy parameters for preprocessing version 20
        '''
        return generate_strat_params_v7(ground_truth,steps)
    
    if version == "v8":
        '''
            load strategy parameters for preprocessing version 22
        '''
        return generate_strat_params_v8(ground_truth,steps)
    
    if version == "v9":
        '''
            load strategy_parameters for preprocessing version 26
        '''
        return generate_strat_params_v9(ground_truth,steps)
    if version == "v10":
        '''
            load strategy_parameters for preprocessing version 28
        '''
        return generate_strat_params_v10(ground_truth,steps)
    if version == "v11":
        '''
            load strategy_parameters for preprocessing version 26
        '''
        return generate_strat_params_v11(ground_truth,steps)
    if version == "v12":
        '''
            load strategy_parameters for preprocessing version 26
        '''
        return generate_strat_params_v12(ground_truth,steps)
    

def deal_with_abnormal_value(arguments, version):
    time_series = arguments['time_series']
    if version is None:
        return time_series
    elif version == "v1":
        '''
        includes processing abnormally large and small values and interpolation for missing values
        '''
        return deal_with_abnormal_value_v1(time_series)
    elif version == "v2":
        '''
        includes processing abnormally large values and interpolation for missing values
        '''
        return deal_with_abnormal_value_v2(time_series)

def labelling(arguments, version):
    time_series = arguments['time_series']
    if version == "v1":
        '''
        standard labelling
        '''
        return labelling_v1(time_series, arguments['horizon'], arguments['ground_truth_columns'])
    
    elif version == "v2":
        """
        construct the torch version
        """
        return labelling_v2(time_series, arguments['horizon'], arguments['ground_truth_columns'])
    
    elif version == "v1_ex1":
        '''
        labelling increases and decreases respective to some price before current time point.
        '''
        return labelling_v1_ex1(time_series, arguments['horizon'], 
                                arguments['ground_truth_columns'], arguments['lags'])

    elif version == "v1_ex2":
        '''
        three classifier
        '''
        return labelling_v1_ex2(time_series, arguments['horizon'],
                                arguments['ground_truth_columns'], arguments['split_dates'][2])


def process_missing_value(arguments, version):
    time_series = arguments['time_series']
    if version == "v1":
        '''
        drop NA rows
        '''
        return process_missing_value_v3(time_series)

def strategy_signal(arguments,version):
    time_series = arguments['time_series']
    if version is None:
        return time_series
    elif version == "v1":
        ts = copy(time_series)
        ts['Label'] = arguments['labels'][0]
        return strategy_signal_v1(ts,  arguments['split_dates'], arguments['ground_truth_columns'], arguments['strat_params'],arguments['activation_params'],0.1,0.1)

def normalize_without_1d_return(arguments, version):
    time_series = arguments['time_series']
    if version is None:
        return time_series, None
    if version == "v1":
        '''
        automated normalization for all possible combinations
        '''
        return normalize_without_1d_return_v1(time_series,time_series.index.get_loc(arguments['split_dates'][1]),
                                                arguments['norm_params'])


def technical_indication(arguments, version):
    time_series = arguments['time_series']
    if version is None:
        return time_series
    if version == "v1":
        '''
        automated generation of divPVT for all possible combinations (only PVT)
        '''
        return technical_indication_v1(time_series,time_series.index.get_loc(arguments['split_dates'][1]),
                                        arguments['tech_params'],arguments['ground_truth_columns'])
                                        
    elif version == "v1_ex3":
        '''
        automated generation of technical indicators for all possible combinations (only LME Ground Truth)
        '''
        return technical_indication_v1_ex3(time_series,time_series.index.get_loc(arguments['split_dates'][1]),
                                        arguments['tech_params'],arguments['ground_truth_columns'])
    
    elif version == "v2":
        '''
        automated generation of divPVT for all possible combinations
        '''
        return technical_indication_v2(time_series,time_series.index.get_loc(arguments['split_dates'][1]),
                                        arguments['tech_params'],arguments['ground_truth_columns'])
    
    elif version == "v2_ex3":
        '''
        automated generation of technical indicators for all possible combinations (only LME Ground Truth)
        '''
        return technical_indication_v2_ex3(time_series,time_series.index.get_loc(arguments['split_dates'][1]),
                                        arguments['tech_params'],arguments['ground_truth_columns'])
    elif version == "v3":
        '''
        automated generation of technical indicators for all possible combinations (only LME Ground Truth)
        '''
        return technical_indication_v3(time_series,time_series.index.get_loc(arguments['split_dates'][1]),
                                        arguments['tech_params'],arguments['ground_truth_columns'])
    


def remove_unused_columns(arguments, version,ground_truth):
    time_series = arguments['time_series']
    if version is None:
        return time_series
    if version == "v1":
        '''
        remove columns that will not be used in model
        '''
        return remove_unused_columns_v1(time_series,arguments['org_cols'])
    if version == "v2":
        '''
        remove columns that will not be used in model
        '''
        return remove_unused_columns_v2(time_series,arguments['org_cols'])
    
    if version == "v3":
        '''
        remove columns that will not be used in model
        '''
        print("Remove Columns Version3")
        return remove_unused_columns_v3(time_series,arguments['org_cols'],ground_truth)
    if version == "v4":
        '''
        remove columns that will not be used in model
        '''
        print("Remove Columns Version4")
        print("ground_truth",ground_truth)
    
        return remove_unused_columns_v4(time_series,arguments['org_cols'],ground_truth)



def price_normalization(arguments, version):
    time_series = arguments['time_series']
    if version is None:
        return time_series
    if version == "v1":
        '''
        daily log returns
        '''
        return log_1d_return(time_series,arguments['org_cols'])
    if version == "v2":
        '''
        DXY log returns
        '''
        return log_1d_return(time_series,["DXY"])
def spot_price_normalization(arguments):
    time_series = arguments['time_series']
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

    
def insert_date_into_feature(arguments):
    time_series = arguments['time_series']

    return insert_date_into_feature_v1(time_series)

def scaling(arguments, version):
    time_series = arguments['time_series']
    if version is None:
        return time_series
    if version == "v1":
        '''
        standard scaling
        '''
        return scaling_v1(time_series,time_series.index.get_loc(arguments['split_dates'][1]))
    if version == "v2":
        '''
        scaling without affecting categorical variables
        '''
        return scaling_v2(time_series,time_series.index.get_loc(arguments['split_dates'][1]),arguments['cat_cols'])


def construct(ind, arguments, version):
    time_series = arguments['time_series']
    if version == "v1":
        '''
        construct ndarray for standard labelling
        '''
        return construct_v1(time_series[ind][arguments['all_cols'][ind]], time_series[ind]["Label"], 
                            arguments['start_ind'], arguments['end_ind'], 
                            arguments['lags'], arguments['horizon'])
    elif version =="v1_ex2":
        '''
        construct ndarray for three classifier  
        '''
        return construct_v1_ex2(time_series[ind][arguments['all_cols'][ind]], time_series[ind]["Label"], 
                            arguments['start_ind'], arguments['end_ind'], 
                            arguments['lags'], arguments['horizon'])
    elif version == "v2":
        '''
        construct ndarray for discrete lags
        '''
        return construct_v2(time_series[ind][arguments['all_cols'][ind]], time_series[ind]["Label"], 
                    arguments['start_ind'], arguments['end_ind'], 
                    arguments['lags'], arguments['horizon'])
    elif version == "v3":
        '''
        construct ndarray for no lag
        '''
        return construct_v3(time_series[ind][arguments['all_cols'][ind]], time_series[ind]["Label"], 
                    arguments['start_ind'], arguments['end_ind'], 
                    arguments['lags'], arguments['horizon'])

