#encoding:utf-8
import pandas as pd
import numpy as np
import time
import sys
import os
import time
import json
sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],'..')))
from data.load_rnn import load_pure_log_reg
from model.logistic_regression import LogReg
from utils.log_reg_functions import objective_function, loss_function
import xgboost as xgb

#file = '~/jiangeng/4EBaseMetal-master/data/Financial Data/LME/LMCADY.csv'
#Copper_file = pd.read_csv(file)
horizon = 3
#f = file
data_configure_file = 'exp/3d/Co/logistic_regression/v1/LMCADY_v1.conf'
with open(os.path.join(sys.path[0],data_configure_file)) as fin:
    fname_columns = json.load(fin)
print(fname_columns)
tra_date = '2005-01-10'
val_date = '2016-06-01'
tes_date = '2016-12-16'
split_date = [tra_date, val_date, tes_date]
lag=5
norm_volume="v1"
norm_3m_spread = "v1"
norm_ex = "v2"
len_ma=5
len_update = 30
version=1
for f in fname_columns: 
    X_tr, y_tr,X_va,y_va,X_te,y_te,norm_params=load_pure_log_reg(
        f,'log_1d_return', split_date, gt_column ='LME_Co_Spot', T=lag,S = horizon,
    vol_norm=norm_volume, ex_spread_norm=norm_ex, spot_spread_norm = norm_3m_spread,
len_ma = len_ma, len_update = len_update, version = version
)
    
