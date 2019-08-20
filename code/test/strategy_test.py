'''
    
'''
import os
import sys


import json
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool as pl
from itertools import permutations, product
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

def output(time_series,ground_truth,labels,strategy_params,activation_params,array):
    org_cols = set(time_series.columns.values.tolist()) - set(["Label"])
    strat = None
    sp = copy(strategy_params)
    for key in activation_params.keys():
        if activation_params[key]:
            strat = key
    if strat is None:
        return None
    n = 0
    for key in sp[strat]:
        sp[strat][key] = array[n]
        n+=1
    if strat =='strat9' and sp[strat]['SlowLength'] < sp[strat]['FastLength']:
        return [""]
    ts = strategy_testing(copy(time_series),ground_truth,sp, activation_params)
    ts = ts[list(set(ts.columns.values.tolist()) - org_cols)]
    temp_list = array
    for col in ts.columns.values.tolist():
        if col == "Label":
            continue
        temp_list.append(col)
        labels = copy(ts['Label'])
        length = len(labels)
        column = copy(ts[col])
        column = column.replace(0,np.nan)
        column = column.dropna()
        labels = labels.loc[column.index]
        labels = np.array(labels)*2-1
        column = np.array(column)
        compared = abs(sum(labels == column)/len(labels)-0.5)
        if compared < 0.025:
            return [""]
        temp_list.append(compared)
        temp_list.append(len(labels)/length)
    temp_list = [str(e) for e in temp_list]
    
    return temp_list



if __name__ == '__main__':
    desc = 'strategy testing'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--data_configure_file', '-c', type=str,
        help='configure file of the features to be read',
        default='log_reg_data.conf'
    )
    parser.add_argument('-s','--steps',type=int,default=1,
                        help='steps in the future to be predicted')
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                        type=str, default="LMCADY")
    parser.add_argument('-out','--output',type = str, default= "",help ="output name")

    args = parser.parse_args()
    ground_truth = args.ground_truth

    with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)[0]

    with open(args.output,"w") as out:
        data_list, LME_dates = read_data_NExT(fname_columns, "2003-11-12")
        time_series = pd.concat(data_list, axis = 1, sort = True)
        n=0
        time_series = deal_with_abnormal_value_v1(time_series)
        LME_dates = sorted(set(LME_dates).intersection(time_series.index.values.tolist()))
        time_series = time_series.loc[LME_dates]
        org_cols = set(time_series.columns.values.tolist())
        labels = labelling_v1(time_series,args.steps,[args.ground_truth])
        time_series = process_missing_value_v3(time_series)
        time_series = pd.concat([time_series, labels[0]], axis = 1)
        
        strategy_params = {'strat3':{'window':0},'strat6':{'window':0,'limiting_factor':0},'strat7':{'window':0,'limiting_factor':0}, 'strat9':{'SlowLength':0,'FastLength':0,'MACDLength':0}}
        activation_params = {'strat3':True, 'strat6':False, 'strat7':False, 'strat9': False}
        comb = range(5,51)
        ls = [list([time_series,ground_truth[:-5],labels,strategy_params,activation_params,[com]]) for com in comb]
        pool = pl()
        results = pool.starmap_async(output,ls)
        pool.close()
        pool.join()
        results = [','.join(res) for res in results.get()]
        [out.write(res+"\n") if len(res.split(",")) > 1 else out.write("") for res in results]

        out.write("\n\n")
        activation_params['strat3'] = False
        activation_params['strat6'] = True
        limiting_factor = np.arange(0.3,1.05,0.05)
        print("a")
        window = range(6,51)
        comb = product(window,limiting_factor)
        ls = [list([time_series,ground_truth[:-5],labels,strategy_params,activation_params,list(com)]) for com in comb]
        pool = pl()
        results = pool.starmap_async(output,ls)
        pool.close()
        pool.join()
        results = [','.join(res) for res in results.get()]
        [out.write(res+"\n") if len(res.split(",")) > 2 else out.write("") for res in results]
        
        out.write("\n\n")
        activation_params['strat6'] = False
        activation_params['strat7'] = True
        limiting_factor = np.arange(1.8,2.44,0.05)
        window = range(8,51)
        print("b")
        comb = product(window,limiting_factor)
        ls = [list([time_series,ground_truth[:-5],labels,strategy_params,activation_params,list(com)]) for com in comb]
        pool = pl()
        results = pool.starmap_async(output,ls)
        pool.close()
        pool.join()
        results = [','.join(res) for res in results.get()]
        [out.write(res+"\n") if len(res.split(",")) > 2 else out.write("") for res in results]

        out.write("\n\n")
        activation_params['strat7'] = False
        activation_params['strat9'] = True
        print("c")
        comb = list(permutations(range(5,50),3))
        ls = [list([time_series,ground_truth[:-5],labels,strategy_params,activation_params,list(com)]) for com in comb]
        print(len(comb))
        pool = pl()
        results = pool.starmap_async(output,ls)
        pool.close()
        pool.join()
        results = [','.join(res) for res in results.get()]
        [out.write(res+"\n") if len(res.split(",")) > 3 else out.write("") for res in results]
        
        out.close()
        

                



