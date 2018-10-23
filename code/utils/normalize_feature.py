from copy import copy
import statsmodels.api as sm
import pandas as pd
import numpy as np

'''
parameters:
X (2d numpy array): the data to be normalized

returns:
X_norm (2d numpy array): note that the dimension of X_norm is different from
    that of X since it less one row (cannot calculate return for the 1st day).
'''
def log_1d_return(X,cols):
    # assert type(X) == np.ndarray, 'only 2d numpy array is accepted'
    for col in cols:
        if type(X[col]) == np.ndarray:
            X[col] = np.log(np.true_divide(X[col][1:], X[col][:-1]))
        else:
            X[col].values[1:] = np.log(np.true_divide(X[col].values[1:],
                                                X[col].values[:-1]))
    # if type(X) == np.ndarray:
    #     return np.log(np.true_divide(X[1:, :], X[:-1, :]))
    # else:
    #     X.values[1:, :] = np.log(np.true_divide(X.values[1:, :],
    #                                             X.values[:-1, :]))
    return X


# See "Volume normalization methods" in google drive/ data cleaning file/volume normalization for more explanations
# "X" is the dataframe we want to process and "OI_name" is the name of the column contained open interest
# version can be v1,v2,v3 or v4 as stated in the file. v1,v2 and v3 will require Open Interest column ("OI_name")
# and for v3 and v4 length of moving average is required
def normalize_volume (X,OI_name,len_ma,version="v1"):
    df_X = X.copy()
    if version == "v1":
        if OI_name not in X.columns:
            print("Open Interest Column missing in dataframe")
            return
        else:
            df_X["volume_v1"] = X['Volume']/X[OI_name]
    elif version == "v2":
        if OI_name not in X.columns:
            print("Open Interest Column missing in dataframe")
            return
        else:
            turn_over = np.log(X['Volume']/X[OI_name])
            df_X["volume_v2"] = turn_over - turn_over.shift(1)
    elif version =="v3":
        if OI_name not in X.columns:
            print("Open Interest Column missing in dataframe")
            return
        else:
            turn_over = np.log(X['Volume']/X[OI_name])
            turn_over_ma = turn_over.shift(len_ma)
            ma_total = 0
            for i in range (len_ma):
                ma_total += turn_over.iloc[i]
            turn_over_ma.iloc[len_ma] = ma_total/len_ma
            for i in range(len_ma,len(turn_over)-1):
                turn_over_ma.iloc[i+1]= (turn_over.iloc[i]+ (len_ma-1)*turn_over_ma.iloc[i])/len_ma
            df_X["volume_v3"]=turn_over-turn_over_ma
    elif version =="v4":
        volume_col =X['Volume'].copy()
        volume_col_ma = volume_col.shift(len_ma)
        ma_total = 0
        for i in range (len_ma):
            ma_total += volume_col.iloc[i]
        volume_col_ma.iloc[len_ma] = ma_total/len_ma
        for i in range(len_ma,len(volume_col)-1):
            volume_col_ma.iloc[i+1]= (volume_col.iloc[i]+ (len_ma-1)*volume_col_ma.iloc[i])/len_ma
        df_X["volume_v4"]=volume_col/volume_col_ma -1
    else:
        print("wrong version")
        return 
            
    return df_X.dropna()
    
# See "spread normalization methods" in google drive/ data cleaning file/spread normalization for more explanations
# "X" is the dataframe we want to process and spot_col is the name of the spot price column
# len_update is for v2, it is after how many days we should update the relationship between spot price and 3month forward price
# version can be v1 or v2 as stated in the file.
def normalize_3mspot_spread (X,spot_col,len_update = 30 ,version="v1"):
    df_X = X.copy()
    if version == "v1":
        if spot_col not in X.columns:
            print("Spot Price Column missing in dataframe")
            return
        else:
            df_X["spread_v1"] = np.log(X['Close.Price'])- np.log(X[spot_col])
    elif version == "v2":
        if spot_col not in X.columns:
            print("Spot Price Column missing in dataframe")
            return
        else:
            three_m = np.log(X['Close.Price'])
            spot = np.log(X[spot_col])
            relationship = spot.shift(len_update)
            model = sm.OLS(three_m[0:len_update],spot[0:len_update])
            results = model.fit()
            beta = results.params[0]
            for i in range(len_update,len(three_m),len_update):
                last_beta = beta
                index_update = i+len_update
                if index_update>(len(three_m)-1):
                    index_update = len(three_m)-1
                relationship[i:index_update] = three_m[i:index_update] - beta*spot[i:index_update]
                model = sm.OLS(three_m[i:index_update],spot[i:index_update])
                results = model.fit()
                beta = results.params[0]
                last_index = i
            relationship[last_index:len(three_m)] = three_m[last_index:len(three_m)]  - last_beta*spot[last_index:len(three_m)] 
            df_X["spread_v2"]=relationship
            
    else:
        print("wrong version")
        return 
            
    return df_X.dropna()

# This function will normalize OI 
# "X" is the dataframe we want to process and spot_col is the name of the spot price column
# OI_col is the col name of the open interest
def normalize_OI (X,OI_col):
    df_X = X.copy()
    OI = np.log(X[OI_col])
    df_X['normalized_OI'] = OI - OI.shift(1)
    return df_X.dropna()

# See "spread normalization methods" in google drive/ data cleaning file/spread normalization for more explanations
# "X" is the dataframe we want to process and lme_col is the name of the spot price column/ 3month forward contract from lme
# len_update is for v2, it is after how many days we should update the relationship between spot price and 3month forward price
# version can be v1 or v2 as stated in the file.
# shfe_col is the name of the column for shfe contract
# exchange is the name of the column for exchange rate

def normalize_3mspot_spread_ex (X,lme_col,shfe_col,exchange,len_update = 30 ,version="v1"):
    df_X = X.copy()
    shfe_usd = X[shfe_col]*X[exchange]
    if version == "v1":
        if lme_col not in X.columns:
            print("LME Price Column missing in dataframe")
            return
        else:
            df_X["spread_shfe_v1"] = np.log(X[lme_col])- np.log(shfe_usd)
    elif version == "v2":
        if lme_col not in X.columns:
            print("LME Price Column missing in dataframe")
            return
        else:
            lme = np.log(X[lme_col])
            relationship = lme.shift(len_update)
            model = sm.OLS(lme[0:len_update],shfe_usd[0:len_update])
            results = model.fit()
            beta = results.params[0]
            for i in range(len_update,len(lme),len_update):
                last_beta = beta
                index_update = i+len_update
                if index_update>(len(lme)-1):
                    index_update = len(lme)-1
                relationship[i:index_update] = lme[i:index_update] - beta*shfe_usd[i:index_update]
                model = sm.OLS(lme[i:index_update],shfe_usd[i:index_update])
                results = model.fit()
                beta = results.params[0]
                last_index = i
            relationship[last_index:len(lme)] = lme[last_index:len(lme)]  - last_beta*shfe_usd[last_index:len(lme)] 
            df_X["spread_shfe_v2"]=relationship
            
    else:
        print("wrong version")
        return 
            
    return df_X.dropna()
