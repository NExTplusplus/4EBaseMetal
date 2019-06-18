import joblib
import json
from copy import copy
import importlib
import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '../utils')))
read_data = importlib.import_module("read_data")
construct_data = importlib.import_module("construct_data")
transform_data = importlib.import_module("transform_data")
normalize_feature = importlib.import_module("normalize_feature")
#Horizon to predict
h = 1


#path to config file
with open("../../exp/test_config.conf") as f:
    fl = json.load(f)

if h == 1:
    lag = 10
    norm_ex = "v1"
    norm_volume = "v1"
    norm_3m_spread = "v1"

elif h == 3:
    lag = 5
    norm_ex = "v1"
    norm_volume = "v1"
    norm_3m_spread = "v1"
    
elif h == 5:
    lag = 5
    norm_ex = "v1"
    norm_volume = "v1"
    norm_3m_spread = "v1"
#load model
model = joblib.load("../../exp/log_reg/LME_Ni_Spot_h"+str(h)+"_n1.joblib")

time_series = None

#Read columns from json file
for fname_columns in fl:
    for fname in fname_columns:
        print('read columns:', fname_columns[fname], 'from:', fname)
        if time_series is None:
            time_series = read_data.read_single_csv(fname, fname_columns[fname])
        else:
            time_series = read_data.merge_data_frame(
                time_series, read_data.read_single_csv(fname, fname_columns[fname])
            )
        
#Processing data
time_series = read_data.process_missing_value_v3(time_series,10)
org_cols = time_series.columns.values.tolist()
print("Normalizing")

#Normalize and generate technical indications
norm_params = construct_data.normalize(time_series,vol_norm = norm_volume, spot_spread_norm=norm_3m_spread,ex_spread_norm = norm_ex)
time_series = copy(norm_params["val"])
del norm_params["val"]
time_series = construct_data.technical_indication(time_series)
cols = time_series.columns.values.tolist()
for col in cols:
    if "_Volume" in col or "_OI" in col or "CNYUSD" in col:
        time_series = time_series.drop(col,axis = 1)
        org_cols.remove(col)
    
ground_truth = copy(time_series["LME_Ni_Spot"])

for ind in range(time_series.shape[0] - h):
    if ground_truth.iloc[ind + h] - ground_truth.iloc[ind] > 0:
        ground_truth.iloc[ind] = 1
    else:
        ground_truth.iloc[ind] = 0

norm_data = copy(normalize_feature.log_1d_return(time_series,org_cols))

norm_data = read_data.process_missing_value_v3(norm_data,10)

X_te, y_te = construct_data.construct(norm_data, ground_truth, 10, norm_data.shape[0] - h - 1, lag, "log_1d_return")

X_te = transform_data.flatten(X_te)

prediction = model.predict(X_te).reshape(X_te.shape[0],1)
y_te = y_te*2-1

with open("Nickel h"+str(h)+".csv","w") as out:
    for i in range(X_te.shape[0]):
        out.write(str(prediction[i]).strip("[ ").strip("]") + ","+ str(y_te[i]).strip("[ ").strip("]") + "\n")
        

total_no = prediction.shape[0]
no_true = sum(np.equal(prediction,y_te))
no_TT = sum(np.multiply(prediction,y_te))
no_FF = sum(np.multiply(prediction - 1,y_te - 1))
no_TF = -sum(np.multiply(prediction,y_te - 1))
no_FT = -sum(np.multiply(prediction - 1,y_te))


print("Overall Accuracy:%d",no_true/total_no )
print("TT:%d", no_TT)
print("TF:%d", no_TF)
print("FT:%d", no_FT)
print("FF:%d", no_FF)
