
import pandas as pd
import numpy as np

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

