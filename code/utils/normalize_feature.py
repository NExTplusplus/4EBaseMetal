from copy import copy
import statsmodels.api as sm
import pandas as pd
import numpy as np

'''
    This file contains the functions to normalize the raw financial data
'''


'''
parameters:
X (2d numpy array): the data to be normalized

returns:
X_norm (2d numpy array): note that the dimension of X_norm is different from
    that of X since it less one row (cannot calculate return for the 1st day).
'''
#daily log returns
def log_1d_return(X,cols):
    for col in cols:
        if type(X[col]) == np.ndarray:
            X[col] = np.log(np.true_divide(X[col][1:], X[col][:-1]))
        else:
            X[col].values[1:] = np.log(np.true_divide(X[col].values[1:],
                                                X[col].values[:-1]))
    return X

#daily log difference between Close, High, and Low to Open
def rel_to_open(X,org_cols):
    for col in X.columns.values.tolist():
        if "High" in col:
            X[col] = (X[col] - X[col[:-4]+"Open"]) / X[col[:-4]+"Open"]
        if "Low" in col:
            X[col] = (X[col] - X[col[:-3]+"Open"]) / X[col[:-3]+"Open"]
        if "Close" in col or len(col.split('_')) == 1:
            X[col] = (X[col] - X[col].shift(1))/X[col].shift(1)
    for col in X.columns.values.tolist():
        if "Open" in col:
            X = X.drop(col,axis = 1)
    return X
    
# See "Volume normalization methods" in google drive/ data cleaning file/volume normalization for more explanations
# "volume" is the volume colume we want to process and "OI" is the OI column contained open interest
# version can be v1,v2,v3 or v4 as stated in the file. v1,v2 and v3 will require Open Interest column ("OI_name")
# and for v3 and v4 length of moving average is required
def normalize_volume(volume,OI=None,len_ma=None,version="v1", train_end=0 ,strength = 0.01,both = 0):

    if version == "v1":
        nVol = volume/OI
    elif version == "v2":
            turn_over = np.log(volume/OI)
            nVol = turn_over - turn_over.shift(1)
    elif version =="v3":
            turn_over = np.log(volume/OI)
            turn_over_ma = turn_over.shift(len_ma)
            ma_total = 0
            for i in range (len_ma):
                ma_total += turn_over.iloc[i]
            turn_over_ma.iloc[len_ma] = ma_total/len_ma
            for i in range(len_ma,len(turn_over)-1):
                turn_over_ma.iloc[i+1]= (turn_over.iloc[i]+ (len_ma-1)*turn_over_ma.iloc[i])/len_ma
            nVol = turn_over-turn_over_ma
    elif version =="v4":
        volume_ma = volume.shift(len_ma)
        ma_total = 0
        for i in range (len_ma):
            ma_total += volume.iloc[i]
        volume_ma.iloc[len_ma] = ma_total/len_ma
        for i in range(len_ma,len(volume)-1):
            volume_ma.iloc[i+1]= (volume.iloc[i]+ (len_ma-1)*volume_ma.iloc[i])/len_ma
        nVol = volume/volume_ma -1
    else:
        print("wrong version")
        return 
    temp = sorted(copy(nVol[:train_end]))    
    mx = temp[-1]
    mn = temp[0]
    if both == 1:
        mx = temp[int(np.floor((1-strength)*len(temp)))]
    elif both == 2:
        mn = temp[int(np.ceil(strength*len(temp)))]
    elif both == 3:
        mx = temp[int(np.floor((1-strength)*len(temp)))]
        mn = temp[int(np.ceil(strength*len(temp)))]
    for i in range(len(nVol)):
        if nVol[i] > mx:
            nVol[i] = mx
        elif nVol[i] < mn:
            nVol[i] = mn
    return nVol
    
# See "spread normalization methods" in google drive/ data cleaning file/spread normalization for more explanations
# "close is the close price column and spot_col is the spot price column
# len_update is for v2, it is after how many days we should update the relationship between spot price and 3month forward price
# version can be v1 or v2 as stated in the file.
def normalize_3mspot_spread (close,spot_col,len_update = 30 ,version="v1"):
    if version == "v1":
            return np.log(close)- np.log(spot_col)
    elif version == "v2":
            three_m = np.log(close)
            spot = np.log(spot_col)
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
            return relationship
            
    else:
        print("wrong version")
        return 

# See "spread normalization methods" in google drive/ data cleaning file/spread normalization for more explanations
# lme_col is the spot price column/ 3month forward contract from lme
# len_update is for v2, it is after how many days we should update the relationship between spot price and 3month forward price
# version can be v1 or v2 as stated in the file.
# shfe_col is the column for shfe contract
# exchange is the column for exchange rate

def normalize_3mspot_spread_ex (lme_col,shfe_col,exchange,len_update = 30 ,version="v1"):
    shfe_usd = shfe_col*exchange
    if version == "v1":
        return np.log(lme_col) - np.log(shfe_usd)
    elif version == "v2":
        lme = np.log(lme_col)
        shfe_usd = np.log(shfe_usd)
        relationship = lme.shift(len_update)
        model = sm.OLS(lme[0:len_update],shfe_usd[0:len_update])
        results = model.fit()
        beta = results.params[0]
        for i in range(len_update,len(lme),len_update):
            last_beta = beta
            index_update = i+len_update
            if index_update>(len(lme)-1):
                index_update = len(lme)-1
            if (lme[i:index_update].empty) or shfe_usd[i:index_update].empty:
                break
            relationship[i:index_update] = lme[i:index_update] - beta*shfe_usd[i:index_update]
            model = sm.OLS(lme[i:index_update],shfe_usd[i:index_update])
            results = model.fit()
            beta = results.params[0]
            last_index = i
        relationship[last_index:len(lme)] = lme[last_index:len(lme)]  - last_beta*shfe_usd[last_index:len(lme)] 
        return relationship

    else:
        print("wrong version")
        return 

# This function will normalize OI 
# OI_col is the col the open interest
def normalize_OI (OI_col, train_end, strength, both):

    OI = np.log(OI_col)
    nOI = OI-OI.shift(1)
    temp = sorted(copy(nOI[:train_end]))    
    mx = temp[-1]
    mn = temp[0]
    if both == 1:
        mx = temp[int(np.floor((1-strength)*len(temp)))]
    elif both == 2:
        mn = temp[int(np.ceil(strength*len(temp)))]
    elif both == 3:
        mx = temp[int(np.floor((1-strength)*len(temp)))]
        mn = temp[int(np.ceil(strength*len(temp)))]
    for i in range(len(nOI)):
        if nOI[i] > mx:
            nOI[i] = mx
        elif nOI[i] < mn:
            nOI[i] = mn
    return nOI


#this function will normalize predictions with spot price
def normalize_prediction(spot,prediction,version="v1"):
    if version == "v1":
        return 1*(prediction > spot)
    elif version == "v2":
        return np.log(prediction) - np.log(spot), 1*(prediction > spot)
    elif version == "v3":
        spread = prediction - spot
        spread = spread/spread.shift(1) - 1
        return spread, 1*(prediction > spot)

# #this function will fractionally difference the selected columns
# def fractional_diff(X,cols):
#     for col in cols:
#         X[col] = frac_diff_ffd(X[col].to_frame(),0.3,0.01)
#     return X

# #this function will return the log return of the STL decomposition trend of column
# def STL_decomposition_trend(col):
#     res = seasonal_decompose(col, freq = 5, two_sided= False)
#     trend = np.log(res.trend) - np.log(res.trend.shift(1))
#     return res.trend

# #this function will return the log return of the STL decomposition seasonal component of column
# def STL_decomposition_seasonal(col):
#     res = seasonal_decompose(col, freq = 5, two_sided= False)
#     return res.seasonal


# def custom_seasonal_decompose(col,train_end):
#     res = custom_seasonal_decomp(col,train_end, period = 5,two_sided= False)
#     return res.seasonal
