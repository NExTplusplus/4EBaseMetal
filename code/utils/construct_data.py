import numpy as np
import pandas as pd
from copy import copy
import os
import sys
from datetime import datetime
sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],"..")))
from utils.normalize_feature import *
from utils.Technical_indicator import *
from sklearn import preprocessing
import scipy.stats as sct

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

def labelling_v1_ex2(X,horizon,ground_truth_columns,val_end):
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

def strategy_testing(X,strategy_params,activation_params):
    cols = X.columns.values.tolist()
    for col in cols:
        if "High" in col and activation_params["strat3"]:
            X[col+"_strat3"] = strategy_3(X[col],strategy_params['strat3']['window'])
        if "Close" in col:
            setting = col[:-5]
            if activation_params["strat3"]:
                X[col+"_strat3"] = strategy_3(X[col],strategy_params['strat3']['window'])
            if activation_params["strat7"]:
                X[col+"_strat7"] = strategy_7(X[col],strategy_params['strat7']['window'],strategy_params['strat7']['limiting_factor'])
            if setting+"High" in cols and setting+"Low" in cols and activation_params["strat6"]:
                X[setting+"strat6"] = strategy_6(X[setting+"High"],X[setting+"Low"],X[col],strategy_params['strat6']['window'],strategy_params['strat6']['limiting_factor'])
            
    return X

def remove_unused_columns_v1(time_series,org_cols):
    for col in copy(time_series.columns):
        if "_Volume" in col or "_OI" in col or "CNYUSD" in col or "_PVT" in col:
            time_series = time_series.drop(col,axis = 1)
            if col in org_cols:
                org_cols.remove(col)
    return time_series, org_cols


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
