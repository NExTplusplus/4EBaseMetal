# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:35:33 2019

"""

import os

import glob
import pandas as pd
import re
from copy import copy
path1 = [os.getcwd()+"\\Indices\\",os.getcwd()+"\\DCE\\Generic\\"]
files = []
target_file = os.getcwd()+"\\LME\\LMCADY.csv"
target = re.split('\\\\|\.',target_file)[-2]
ground_truth = pd.read_csv(target_file)
ground_truth['Return'] = (ground_truth.iloc[:,-1] - ground_truth.iloc[:,-1].shift(1))/ground_truth.iloc[:,-1].shift(1)
ground_truth['label'] = 2*(ground_truth['Return']>0)-1
lags = list(range(5,121,5))
date = list(range(2007,2016))

for path in path1:
    for file in glob.glob(path+"*.csv"):
        files.append(file)

for file in files:
    tag = re.split('\\\\|\.',file)[-2]
    tmp = pd.read_csv(file)
#    tmp.index = tmp.Index
    correlation = pd.DataFrame()
    if "Close" in tmp.columns:
        tmp["Return"] = (tmp.Close-tmp.Close.shift(1))/tmp.Close.shift(1)
        for lag in lags:
            tmp['lag'+str(lag)] = tmp.Return.shift(lag)
    else:
        for lag in lags:
            tmp["Return"] = (tmp.iloc[:,-1]-tmp.iloc[:,-1].shift(1))/tmp.iloc[:,-1].shift(1)
            tmp['lag'+str(lag)] = tmp.Return.shift(lag)
        
    for year in date:
        indice = split_date(copy(tmp),year)
        time_series = split_date(copy(ground_truth),year)
#    tmp = tmp.loc[tmp["Index"]>"2016-01-01"]
#    time_series = ground_truth.loc[ground_truth["Index"]>"2016-01-01"]
        indice.index = indice.Index
        time_series.index = time_series.Index
        time_series = time_series.drop("Index",axis = 1)
        col_list = []
        for col in indice.columns:
            if "lag" not in col:
                col_list.append(col)
        indice = indice.drop(columns = col_list)
        
        time_series = pd.concat([time_series,indice],axis = 1)
        time_series = time_series.dropna(axis = 0)
        corr = pd.DataFrame(time_series.corr())
        corr[str(year)] = corr['Return']
        correlation = pd.concat([correlation,corr[str(year)]],axis = 1)
    correlation = correlation.drop([x for x in correlation.index if "lag" not in x])
    correlation.to_csv(target+'_'+tag+'.csv')

def split_date(time_series,year):
    time_series = time_series.loc[(time_series.Index>str(year)+'-01-01') & (time_series.Index<str(year+1)+'-01-01')]
    return time_series
