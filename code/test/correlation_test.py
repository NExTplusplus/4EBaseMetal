'''
    
'''
import os
import sys


import json
import argparse
import numpy as np
import pandas as pd
from copy import copy,deepcopy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from utils.read_data import read_data_NExT, process_missing_value_v3,read_data_v5_4E
from utils.construct_data import labelling_v1, deal_with_abnormal_value_v1, rolling_half_year
from sklearn.metrics import accuracy_score

os.chdir(os.path.abspath(sys.path[0]))
def save_data(fname,time_series,columns, ground_truth = None):
    col_name = ""
    for col in columns:
        col_name = col_name + " " + col
    with open(fname+".csv","w") as out:
        out.write(col_name.replace(" ",","))
        out.write(",\n")
        for i in [5]:
            row = time_series.iloc[i]
            out.write(time_series.index[i]+",")
            for v in row:
                out.write(str(v)+",")
            if args.ground_truth is not None:
                out.write(str(args.ground_truth[i]))
            out.write("\n")

if __name__ == '__main__':
    desc = 'strategy testing'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--data_configure_file', '-c', type=str,
        help='configure file of the features to be read',
        default='exp/5d/Co/logistic_regression/v5/LMCADY_v5.conf'
    )
    parser.add_argument('-s','--steps',type=str,default='1,1',
                        help='steps in the future to be predicted')
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                        type=str, default="LME_Co_Spot,LME_Zi_Spot")
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "NExT"
    )
    parser.add_argument(
        '-l','--lag', help='lag', type = int, default = 0
    )

    args = parser.parse_args()
    ground_truth = args.ground_truth.split(",")
    steps = [int(i) for i in args.steps.split(",")]
    with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)[0]

    if args.source == "NExT":
        data_list, LME_dates = read_data_NExT(fname_columns, "2000-11-12")
        time_series = pd.concat(data_list, axis = 1, sort = True)
    elif args.source == "4E":
        time_series, LME_dates = read_data_v5_4E("2000-11-12")
    
    time_series = deal_with_abnormal_value_v1(time_series)
    LME_dates = sorted(set(LME_dates).intersection(time_series.index.values.tolist()))
    time_series = time_series.loc[LME_dates]
    org_cols = set(time_series.columns.values.tolist())
    ttarget = labelling_v1(time_series,steps[0],[ground_truth[0]])[0]
    ttbc = labelling_v1(time_series,steps[1],[ground_truth[1]])[0]
    ans = {"start":[],"end":[],"acc":[]}
    start_date = "2002-01-01"
    last_date =  "2017-01-01"
    ttbc = ttbc.shift(args.lag)
    ttbc.dropna(inplace = True)
    ttarget = ttarget.loc[ttbc.index]
    

    ans['start'].append(start_date)
    ans['end'].append(last_date)
    ans['acc'].append(sum(ttarget.loc[(ttarget.index>=start_date)&(ttarget.index< last_date)] == ttbc.loc[(ttbc.index>=start_date)&(ttbc.index< last_date)])/len(ttarget.loc[(ttarget.index>=start_date)&(ttarget.index< last_date)]))

    while len(ttarget.loc[ttarget.index > start_date])>0 :
        end_date = str(int(start_date.split("-")[0])+1)+"-01-01"
        target = copy(ttarget)
        tbc = copy(ttbc)
        target = target[(target.index >= start_date)&(target.index <end_date)]
        tbc = tbc[(tbc.index >= start_date)&(tbc.index <end_date)]
        ans['start'].append(start_date)
        ans['end'].append(end_date)
        ans['acc'].append(sum(target == tbc)/len(target))
        start_date = end_date
    ans = pd.DataFrame(ans)
    ans.to_csv(args.ground_truth+args.steps+".csv")


    

    


    