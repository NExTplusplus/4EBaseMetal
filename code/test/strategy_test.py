'''
    
'''
import os
import sys


import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from utils.read_data import read_data_NExT, process_missing_value_v3
from utils.construct_data import labelling_v1, deal_with_abnormal_value_v1, strategy_testing

os.chdir(os.path.abspath(sys.path[0]))
def save_data(fname,time_series,columns, ground_truth = None):
    col_name = ""
    for col in columns:
        col_name = col_name + " " + col
    with open("../../"+fname+".csv","w") as out:
        out.write(col_name.replace(" ",","))
        out.write(",\n")
        for i in [5]:
            row = time_series.iloc[i]
            out.write(time_series.index[i]+",")
            for v in row:
                out.write(str(v)+",")
            if ground_truth is not None:
                out.write(str(ground_truth[i]))
            out.write("\n")
# read data configure file
with open(os.path.join(sys.path[0],"exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf")) as fin:
    fname_columns = json.load(fin)[0]

with open("test1.csv","w") as out:
    data_list, LME_dates = read_data_NExT(fname_columns, "2003-11-12")
    time_series = pd.concat(data_list, axis = 1, sort = True)
    n=0
    time_series = deal_with_abnormal_value_v1(time_series)
    LME_dates = sorted(set(LME_dates).intersection(time_series.index.values.tolist()))
    time_series = time_series.loc[LME_dates]
    org_cols = set(time_series.columns.values.tolist())
    labels = labelling_v1(time_series,3,['LME_Co_Spot'])
    time_series = process_missing_value_v3(time_series)
    time_series = pd.concat([time_series, labels[0]], axis = 1)
    
    strategy_params = {'strat3':{'window':0},'strat6':{'window':0,'limiting_factor':0},'strat7':{'window':0,'limiting_factor':0}}
    activation_params = {'strat3':True, 'strat6':False, 'strat7':False}
    out.write("strat3_window,")
    for strat3_window in [20]:
        strategy_params['strat3']['window'] = strat3_window
        ts = strategy_testing(copy(time_series),strategy_params, activation_params)
        ts = ts[list(set(ts.columns.values.tolist()) - org_cols)]
        temp_list = [strat3_window]
        for col in ts.columns.values.tolist():
            if col == "Label":
                continue
            if n == 0:
                out.write(col+","+col+"_length,")
            labels = copy(ts['Label'])
            length = len(labels)
            column = copy(ts[col])
            column = column.replace(0,np.nan)
            column = column.dropna()
            labels = labels.loc[column.index]
            labels = np.array(labels)*2-1
            column = np.array(column)
            compared = sum(labels == column)/len(labels)
            temp_list.append(compared)
            temp_list.append(len(labels)/length)
        if n == 0:
            out.write("\n")
            n+=1
        temp_list = [str(e) for e in temp_list]
        out.write(",".join(temp_list))
        out.write("\n")

    out.write("\n\nstrat6_window,strat6_limiting_factor,")
    n = 0
    activation_params['strat3'] = False
    activation_params['strat6'] = True
    for strat6_window in [10]:
        for strat6_limiting_factor in [0.3]:
            strategy_params['strat6']['window'] = strat6_window
            strategy_params['strat6']['limiting_factor'] = strat6_limiting_factor
            ts = strategy_testing(copy(time_series),strategy_params, activation_params)
            ts = ts[list(set(ts.columns.values.tolist()) - org_cols)]
            temp_list = [strat6_window,strat6_limiting_factor]
            for col in ts.columns.values.tolist():
                if col == "Label":
                    continue
                if n == 0:
                    out.write(col+","+col+"_length,")
                labels = copy(ts['Label'])
                length = len(labels)
                column = copy(ts[col])
                column = column.replace(0,np.nan)
                column = column.dropna()
                labels = labels.loc[column.index]
                labels = np.array(labels)*2-1
                column = np.array(column)
                compared = sum(labels == column)/len(labels)
                temp_list.append(compared)
                temp_list.append(len(labels)/length)
            if n == 0:
                out.write("\n")
                n+=1
            temp_list = [str(e) for e in temp_list]
            out.write(",".join(temp_list))
            out.write("\n")


    out.write("\n\nstrat7_window,strat7_limiting_factor,")
    n = 0
    activation_params['strat6'] = False
    activation_params['strat7'] = True
    for strat7_window in [6]:
        for strat7_limiting_factor in [1.8]:
            strategy_params['strat7']['window'] = strat6_window
            strategy_params['strat7']['limiting_factor'] = strat6_limiting_factor
            ts = strategy_testing(copy(time_series),strategy_params, activation_params)
            ts = ts[list(set(ts.columns.values.tolist()) - org_cols)]
            temp_list = [strat7_window,strat7_limiting_factor]
            for col in ts.columns.values.tolist():
                if col == "Label":
                    continue
                if n == 0:
                    out.write(col+","+col+"_length,")
                labels = copy(ts['Label'])
                length = len(labels)
                column = copy(ts[col])
                column = column.replace(0,np.nan)
                column = column.dropna()
                labels = labels.loc[column.index]
                labels = np.array(labels)*2-1
                column = np.array(column)
                compared = sum(labels == column)/len(labels)
                temp_list.append(compared)
                temp_list.append(len(labels)/length)
            if n == 0:
                out.write("\n")
                n+=1
            temp_list = [str(e) for e in temp_list]
            out.write(",".join(temp_list))
            out.write("\n")
    



    out.close()
    

            



