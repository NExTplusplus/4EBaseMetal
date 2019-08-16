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

with open("test.csv","w") as out:
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
    
    out.write("Strat3_Window,Strat6_Window,Strat7_Window,Strat6_Limiting_Factor,Strat7_Limiting_Factor,")
    for strat3_window in range(5,50,1):
        for strat6_window in range(5,50,1):
            for strat7_window in range(5,50,1):
                for strat6_limiting_factor in range(0.3,1.0,0.1):
                    for strat7_limiting_factor in range(1.8,2.4,0.1):
                        strategy_params = {'strat3':{'window':strat3_window},'strat6':{'window':strat6_window,'limiting_factor':strat6_limiting_factor},'strat7':{'window':strat7_window,'limiting_factor':strat7_limiting_factor}}
                        time_series = strategy_testing(time_series,strategy_params)
                        time_series = time_series[list(set(time_series.columns.values.tolist()) - org_cols)]
                        temp_list = [strat3_window,strat6_window,strat7_window,strat6_limiting_factor, strat7_limiting_factor]
                        for col in time_series.columns.values.tolist():
                            if col == "Label":
                                continue
                            if n == 0:
                                out.write(col+","+col+"_length,")
                            labels = copy(time_series['Label'])
                            length = len(labels)
                            column = copy(time_series[col])
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
    

            



