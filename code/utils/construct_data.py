import numpy as np
import pandas as pd
from utils.normalize_feature import normalize_3mspot_spread,normalize_3mspot_spread_ex,normalize_OI,normalize_volume
from utils.Technical indicator import ad, divergence_ad, pvt, divergence_pvt

def construct(time_series, ground_truth, start_ind, end_ind, T, norm_method):
    num = 0
    '''
        convert 2d numpy array of time series data into 3d numpy array, with extra dimension for lags, i.e.
        input of (n_samples, n_features) becomes (n_samples, T, n_features)
        time_series (2d np.array): financial time series data
        ground_truth (1d np.array): column which is used as ground truth
        start_index (string): string which is the date that we wish to begin including from.
        end_index (string): string which is the date that we wish to include last.
        T (int): number of lags
        norm_method (string): normalization method
    '''
    for ind in range(start_ind + 1, end_ind + 1):
        if not time_series.iloc[ind - T: ind].isnull().values.any():
            num += 1
    X = np.zeros([num, T, time_series.shape[1]], dtype=np.float32)
    y = np.zeros([num, 1], dtype=np.float32)

    sample_ind = 0
    for ind in range(start_ind + 1, end_ind + 1):
        if not time_series.iloc[ind - T: ind].isnull().values.any():
            if norm_method == "log_1d_return":
                X[sample_ind] = time_series.values[ind - T: ind, :]
            elif norm_method == "log_nd_return":
                X[sample_ind] = np.flipud(np.add.accumulate(np.flipud(time_series.values[ind - T: ind, :])))
            y[sample_ind, 0] = ground_truth.values[ind - 1]
            sample_ind += 1

    return X,y

def normalize(X,vol_norm ="v1", vol_len = None, spot_spread_norm = "v1", 
                spot_spread_len = 30, ex_spread_norm = "v1",ex_spread_len = 30):
    ans = {"val":None, "nVol":False,"nSpread":False,"nEx":False}
    cols = X.columns.values.tolist()
    ex = False
    if "CNYUSD" in cols:
        print("Considering Exchange Rate")
        ex = True
    
    for col in cols:
        if "OI" in col:
            print("Normalizing OI:"+"=>".join((col,col[:-2]+"nOI")))
            X[col[:-2]+"nOI"] = normalize_OI(X[col])
        if "Volume" in col:
            setting = col[:-6]
            if setting+"OI" in cols:
                ans["nVol"] = True
                print("Normalizing Volume:"+"=>".join((col,setting+"OI")))
                X[setting+"nVolume"] = normalize_volume(X[col],OI=X[setting+"OI"],len_ma = vol_len,version = vol_norm)
        if "Close" in col:
            setting = col[:-5]
            if setting+"Spot" in cols:
                ans["nSpread"] = True
                print("Normalizing Spread:"+"=>".join((col,setting+"Spot")))
                X[setting+"n3MSpread"] = normalize_3mspot_spread(X[col],X[setting+"Spot"],
                                                                len_update=spot_spread_len,version = spot_spread_norm)
        if "SHFE" in col and "Close" in col and ex:
            metal = col.split("_")[1]
            if "_".join(("LME",metal,"Spot")) in cols:
                ans["nEx"] = True
                print("+".join((col,"_".join(("LME",metal,"Spot"))))+"=>"+"_".join(("SHFE",metal,"nEx3MSpread")))
                X["_".join(("SHFE",metal,"nEx3MSpread"))] = normalize_3mspot_spread_ex(X["_".join(("LME",metal,"Spot"))],X[col],X["CNYUSD"],
                                                                                    len_update=ex_spread_len,version = ex_spread_norm)
            if "_".join(("LME",metal,"Close")) in cols:
                ans["nEx"] = True
                print("+".join((col,"_".join(("LME",metal,"Close"))))+"=>"+"_".join(("SHFE",metal,"nEx3MSpread")))
                X["_".join(("SHFE",metal,"nExSpread"))] = normalize_3mspot_spread_ex(X["_".join(("LME",metal,"Close"))],X[col],X["CNYUSD"],
                                                                                    len_update=ex_spread_len,version = ex_spread_norm)
            
    ans["val"] = X
    return ans

def technical_indication(X):
    cols = X.columns.values.tolist()
    for col in cols:
        if "Close" in col:
            setting = col[:5]
            if setting+"Volume" in cols:
                print("+".join((col,setting+"Volume"))+"=>"+"+".join((setting+"PVT",setting+"divPVT")))
                X[setting+"PVT"] = pvt(X[col],X[setting+"Volume"])
                X[setting+"divPVT"] = divergence_pvt(X[col],X[setting+"PVT"])
            if set([setting+"Volume",setting+"Open",setting+"High",setting+"Low"]).issubset(cols):
                print("+".join((col,setting+"Volume",setting+"Open",setting+"High",setting+"Low"))+"=>"+"+".join((setting+"AD",setting+"divAD")))
                X[setting+"AD"] = ad(X[col],X[setting+"Low"],X[setting+"Open"],X[setting+"High"],X[setting+"Volume"])
                X[setting+"divAD"] = divergence_ad(X[col],X[setting+"AD"])
            
    return X

        