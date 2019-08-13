# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:11:31 2019

@author: Kwoks
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

metal = '../big/NExT/Data/Version 1/LME/LMAHDY.csv'
price = pd.read_csv(metal)
price['Index'] =  pd.to_datetime(price['Index'])
price['return_1d'] = (price['LMAHDY']/price['LMAHDY'].shift(1)-1)*100
price.dropna(inplace = True)

price['return_1d'] = ((price['LMAHDY'] / price['LMAHDY'].shift(1)) -1)*100
price['return_3d'] = ((price['LMAHDY'] / price['LMAHDY'].shift(3)) -1)*100
price['return_5d'] = ((price['LMAHDY'] / price['LMAHDY'].shift(5)) -1)*100
price['return_7d'] = ((price['LMAHDY'] / price['LMAHDY'].shift(7)) -1)*100
price['return_10d'] = ((price['LMAHDY'] / price['LMAHDY'].shift(10)) -1)*100
price['return_15d'] = ((price['LMAHDY'] / price['LMAHDY'].shift(15)) -1)*100
price['return_20d'] = ((price['LMAHDY'] / price['LMAHDY'].shift(20)) -1)*100

period = 100
price['Std1d_20'] = (price[u'return_1d'].shift(1).rolling(period).std())
price['Std3d_20'] = (price[u'return_3d'].shift(1).rolling(period).std())
price['Std5d_20'] = (price[u'return_5d'].shift(1).rolling(period).std())
price['Std7d_20'] = (price[u'return_7d'].shift(1).rolling(period).std())
price['Std10d_20'] = (price[u'return_10d'].shift(1).rolling(period).std())
price['Std15d_20'] = (price[u'return_15d'].shift(1).rolling(period).std())
price['Std20d_20'] = (price[u'return_20d'].shift(1).rolling(period).std())

price_forward = price.copy()
price_forward['return_1d'] = price_forward['return_1d'].shift(-1)
price_forward['return_3d'] = price_forward['return_3d'].shift(-3)
price_forward['return_5d'] = price_forward['return_5d'].shift(-5)
price_forward['return_7d'] = price_forward['return_7d'].shift(-7)
price_forward['return_10d'] = price_forward['return_10d'].shift(-10)
price_forward['return_15d'] = price_forward['return_15d'].shift(-15)
price_forward['return_20d'] = price_forward['return_20d'].shift(-20)


