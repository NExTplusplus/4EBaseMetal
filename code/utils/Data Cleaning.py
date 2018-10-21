
# coding: utf-8

# In[23]:


DATA_PATH_OHLC = "C:/Users/User/Desktop/Next++/Data/LME/LMECopper3M.csv"
DATA_PATH_SPOT = "C:/Users/User/Desktop/Next++/Data/LME/LMCADY.csv"    
DATA_PATH_OI = "C:/Users/User/Desktop/Next++/Data/LME/LMCADS03_OI.csv" 

import statsmodels.api as sm
import pandas as pd
import numpy as np
ohlc = pd.read_csv(DATA_PATH_OHLC,index_col=0)
spot = pd.read_csv(DATA_PATH_SPOT,index_col=0)
oi = pd.read_csv(DATA_PATH_OI,index_col=0)
all_data = pd.concat([ohlc, spot,oi], axis=1, sort=True)


# In[24]:


# See "Deal with NA value" in google drive/ data cleaning file for more explanations
# "X" is the dataframe we want to process and "cons_data" is number of consecutive complede data we need to have 
def process_missing_value_v3(X,cons_data):
    count = 0
    sta_ind = 0
    for i in range(X.shape[0]):
        if not X.iloc[i].isnull().values.any():
            count= count +1
            if sta_ind!=0:
                sta_ind = i
        else:
            count = 0
            sta_ind =0
        if count == cons_data:
            break
        
    return X[sta_ind:].dropna()


# In[83]:


# See "Volume normalization methods" in google drive/ data cleaning file/volume normalization for more explanations
# "X" is the dataframe we want to process and "OI_name" is the name of the column contained open interest
# version can be v1,v2,v3 or v4 as stated in the file. v1,v2 and v3 will require Open Interest column ("OI_name")
# and for v3 and v4 length of moving average is required
def normalze_volume (X,OI_name,len_ma,version="v1"):
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
    


# In[101]:


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


# In[153]:


# This function will normalize OI 
# "X" is the dataframe we want to process and spot_col is the name of the spot price column
# OI_col is the col name of the open interest
def normalize_OI (X,OI_col):
    df_X = X.copy()
    OI = np.log(X[OI_col])
    df_X['normalized_OI'] = OI - OI.shift(1)
    return df_X.dropna()


# In[149]:


# This function will calculate Price Volume Trend as mentioned in google drive/ technical indicator for more explanations
# "X" is the dataframe we want to process
# version can be v1,v2,v3,v4 or v5 as stated in the file.
def pvt (X,version = "v1"):
    df_X = X.copy()
    current_close = X['Close.Price']
    pvt = current_close.shift(1)
    if version == "v1":
        volume = X['Volume']
        pvt.iloc[1] = ((current_close.iloc[1]/current_close.iloc[0])-1)*volume.iloc[1]
        print(pvt.iloc[1])
        for i in range(2,len(current_close)):
            pvt.iloc[i] = ((current_close.iloc[i]/current_close.iloc[i-1])-1)*volume.iloc[i] + pvt.iloc[i-1]
        df_X["pvt_v1"] = pvt
        
    elif version == "v2":
        if "volume_v1" not in X.columns:
            print("volume_v1 missing use normalze_volume and choose v1 first")
            return
        else:
            volume = X['volume_v1']
            pvt.iloc[1] = ((current_close.iloc[1]/current_close.iloc[0])-1)*volume.iloc[1]
            for i in range(2,len(current_close)):
                pvt.iloc[i] = ((current_close.iloc[i]/current_close.iloc[i-1])-1)*volume.iloc[i]+ pvt.iloc[i-1]
            df_X["pvt_v2"] = pvt
    elif version == "v3":
        if "volume_v2" not in X.columns:
            print("volume_v2 missing use normalze_volume and choose v2 first")
            return
        else:
            volume = X['volume_v2']
            pvt.iloc[1] = ((current_close.iloc[1]/current_close.iloc[0])-1)*volume.iloc[1]
            for i in range(2,len(current_close)):
                pvt.iloc[i] = ((current_close.iloc[i]/current_close.iloc[i-1])-1)*volume.iloc[i]+ pvt.iloc[i-1]
            df_X["pvt_v3"] = pvt
    elif version == "v4":
        if "volume_v3" not in X.columns:
            print("volume_v3 missing use normalze_volume and choose v3 first")
            return
        else:
            volume = X['volume_v3']
            pvt.iloc[1] = ((current_close.iloc[1]/current_close.iloc[0])-1)*volume.iloc[1]
            for i in range(2,len(current_close)):
                pvt.iloc[i] = ((current_close.iloc[i]/current_close.iloc[i-1])-1)*volume.iloc[i]+ pvt.iloc[i-1]
            df_X["pvt_v4"] = pvt
    elif version == "v5":
        if "volume_v4" not in X.columns:
            print("volume_v4 missing use normalze_volume and choose v4 first")
            return
        else:
            volume = X['volume_v4']
            pvt.iloc[1] = ((current_close.iloc[1]/current_close.iloc[0])-1)*volume.iloc[1]
            for i in range(2,len(current_close)):
                pvt.iloc[i] = ((current_close.iloc[i]/current_close.iloc[i-1])-1)*volume.iloc[i]+ pvt.iloc[i-1]
            df_X["pvt_v5"] = pvt
    else:
        print("wrong version")
        return 
    return df_X.dropna()


# In[154]:


# This function will calculate divergence between Price Volume Trend and percentage price change 
# as mentioned in google drive/ technical indicator for more explanations
# "X" is the dataframe we want to process
# version can be v1,v2,v3,v4 or v5 as stated in the file.
def divergence_pvt (X,version = "v1"):
    df_X = X.copy()
    current_close = X['Close.Price']
    percentage_change = (current_close/current_close.shift(1))-1
    if version == "v1":
        if "pvt_v1" not in X.columns:
            print("pvt_v1 missing use pvt and choose v1 first")
            return
        else:
            pvt = X['pvt_v1']
            df_X["div_pvt_v1"] = percentage_change-pvt
            
    elif version == "v2":
        if "pvt_v2" not in X.columns:
            print("pvt_v2 missing use pvt and choose v2 first")
            return
        else:
            pvt = X['pvt_v2']
            df_X["div_pvt_v2"] = percentage_change-pvt
    
    elif version == "v3":
        if "pvt_v3" not in X.columns:
            print("pvt_v3 missing use pvt and choose v3 first")
            return
        else:
            pvt = X['pvt_v3']
            df_X["div_pvt_v3"] = percentage_change-pvt
    elif version == "v4":
        if "pvt_v4" not in X.columns:
            print("pvt_v4 missing use pvt and choose v4 first")
            return
        else:
            pvt = X['pvt_v4']
            df_X["div_pvt_v4"] = percentage_change-pvt
    elif version == "v5":
        if "pvt_v5" not in X.columns:
            print("pvt_v5 missing use pvt and choose v5 first")
            return
        else:
            pvt = X['pvt_v5']
            df_X["div_pvt_v5"] = percentage_change-pvt
    else:
        print("wrong version")
        return 
    return df_X.dropna()


# In[157]:


# This function will calculate accumulation/distribution as mentioned in google drive/ technical indicator for more explanations
# "X" is the dataframe we want to process
# version can be v1,v2,v3,v4 or v5 as stated in the file.
def ad (X,version = "v1"):
    df_X = X.copy()
    close_p = X['Close.Price']
    low_p =  X['Low.Price']
    open_p =  X['Open.Price']
    high_p =  X['High.Price']
    money_flow = ((close_p - low_p)-(high_p-close_p))/(high_p-low_p)
    if version == "v1":
        volume = X['Volume']
        df_X["ad_v1"] = money_flow*volume
    elif version == "v2":
        if "volume_v1" not in X.columns:
            print("volume_v1 missing use normalze_volume and choose v1 first")
            return
        else:
            volume = X['volume_v1']
            df_X["ad_v2"] = money_flow*volume
    elif version == "v3":
        if "volume_v2" not in X.columns:
            print("volume_v2 missing use normalze_volume and choose v2 first")
            return
        else:
            volume = X['volume_v2']
            df_X["ad_v3"] = money_flow*volume

    elif version == "v4":
        if "volume_v3" not in X.columns:
            print("volume_v3 missing use normalze_volume and choose v3 first")
            return
        else:
            volume = X['volume_v3']
            df_X["ad_v4"] = money_flow*volume
    elif version == "v5":
        if "volume_v4" not in X.columns:
            print("volume_v4 missing use normalze_volume and choose v4 first")
            return
        else:
            volume = X['volume_v4']
            df_X["ad_v5"] = money_flow*volume
    else:
        print("wrong version")
        return 
    
    
    return df_X.dropna()


# In[158]:


# This function will calculate divergence between accumulation/distribution and percentage price change 
# as mentioned in google drive/ technical indicator for more explanations
# "X" is the dataframe we want to process
# version can be v1,v2,v3,v4 or v5 as stated in the file.
def divergence_ad (X,version = "v1"):
    df_X = X.copy()
    current_close = X['Close.Price']
    percentage_change = (current_close/current_close.shift(1))-1
    if version == "v1":
        if "ad_v1" not in X.columns:
            print("ad_v1 missing use ad and choose v1 first")
            return
        else:
            ad = X['ad_v1']
            df_X["div_ad_v1"] = percentage_change-ad
            
    elif version == "v2":
        if "ad_v2" not in X.columns:
            print("ad_v2 missing use ad and choose v2 first")
            return
        else:
            ad = X['ad_v2']
            df_X["div_ad_v2"] = percentage_change-ad
    
    if version == "v3":
        if "ad_v3" not in X.columns:
            print("ad_v3 missing use ad and choose v3 first")
            return
        else:
            ad = X['ad_v3']
            df_X["div_ad_v3"] = percentage_change-ad
    if version == "v4":
        if "ad_v4" not in X.columns:
            print("ad_v4 missing use ad and choose v4 first")
            return
        else:
            ad = X['ad_v4']
            df_X["div_ad_v4"] = percentage_change-ad
    if version == "v5":
        if "ad_v5" not in X.columns:
            print("ad_v5 missing use ad and choose v5 first")
            return
        else:
            ad = X['ad_v5']
            df_X["div_ad_v5"] = percentage_change-ad
    else:
        print("wrong version")
        return 
    return df_X.dropna()

