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
from utils.read_data import read_data_NExT, process_missing_value_v3,read_data_v5_4E
from utils.construct_data import labelling_v1, deal_with_abnormal_value_v1, strategy_testing, rolling_half_year

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

def output(time_series,ground_truth,strategy_params,activation_params,array,check = True):
    org_cols = set(time_series.columns.values.tolist()) - set(["Label"])
    strat = None
    sp = copy(strategy_params)
    for key in activation_params.keys():
        if activation_params[key]:
            strat = key
    if strat is None:
        return 
    n = 0
    for key in sp[strat]:
        sp[strat][key] = array[n]
        n+=1
    if strat =='strat9' and sp[strat]['SlowLength'] < sp[strat]['FastLength']:
        return 
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
        # if compared < 0.025 and check:
        #     return [""]
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
        default='exp/5d/Co/logistic_regression/v5/LMCADY_v5.conf'
    )
    parser.add_argument('-s','--steps',type=int,default=5,
                        help='steps in the future to be predicted')
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                        type=str, default="LME_Co_Spot")
    parser.add_argument('-out','--output',type = str, default= "test1.csv",help ="output name")
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "NExT"
    )

    args = parser.parse_args()
    ground_truth = args.ground_truth

    with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)[0]

    with open(args.output,"w") as out:
        if args.source == "NExT":
            data_list, LME_dates = read_data_NExT(fname_columns, "2003-11-12")
            time_series = pd.concat(data_list, axis = 1, sort = True)
        elif args.source == "4E":
            time_series, LME_dates = read_data_v5_4E("2003-11-12")
        
        n=0
        time_series = deal_with_abnormal_value_v1(time_series)
        LME_dates = sorted(set(LME_dates).intersection(time_series.index.values.tolist()))
        time_series = time_series.loc[LME_dates]
        org_cols = set(time_series.columns.values.tolist())
        labels = labelling_v1(time_series,args.steps,[args.ground_truth])
        time_series = process_missing_value_v3(time_series)
        time_series = pd.concat([time_series, labels[0]], axis = 1)
        if args.source == "NExT":
            split_dates = rolling_half_year("2003-01-01","2017-01-01",5)
            split_dates = split_dates[-5:]
        elif args.source == "4E":
            split_dates = rolling_half_year("2003-01-01","2019-01-01",5)
            split_dates = split_dates[-9:]
        for split_date in split_dates:
            ts = time_series.loc[(time_series.index >= split_date[0])&(time_series.index <= split_date[1])]
            strategy_params = {'strat3':{'window':0},'strat6':{'window':0,'limiting_factor':0},'strat7':{'window':0,'limiting_factor':0}, 'strat9':{'SlowLength':0,'FastLength':0,'MACDLength':0}}
            activation_params = {'strat3':True, 'strat6':False, 'strat7':False, 'strat9': False}
            strat_results = {'strat3':{'high window':0, 'close window':0},'strat6':{'window':0,'limiting_factor':0},'strat7':{'window':0,'limiting_factor':0}, 'strat9':{'SlowLength':0,'FastLength':0,'MACDLength':0}}
            comb = range(5,51,2)
            ls = [list([ts,ground_truth[:-5],strategy_params,activation_params,[com]]) for com in comb]
            pool = pl()
            results = pool.starmap_async(output,ls)
            pool.close()
            pool.join()
            results,idx = list(np.unique(results.get(),return_inverse = True,axis = 1))
            results = [list(res[idx]) for res in results]
            results = [[int(res[0]),float(res[2]),float(res[3]),float(res[5]),float(res[6])] for res in results]
            results = pd.DataFrame(data = results, columns = ["Window","High Acc","High Cov","Close Acc","Close Cov"])
            results = results.loc[(results["High Cov"]>0.05)&(results["Close Cov"]>0.05)]
            strat_results['strat3']['high window'] = results.loc[results["High Acc"].idxmax(),"Window"]
            strat_results['strat3']['close window'] = results.loc[results["Close Acc"].idxmax(),"Window"]
            
            activation_params['strat3'] = False
            activation_params['strat6'] = True
            limiting_factor = np.arange(0.3,1.05,0.1)
            window = range(10,51,2)
            comb = product(window,limiting_factor)
            ls = [list([ts,ground_truth[:-5],strategy_params,activation_params,list(com)]) for com in comb]
            pool = pl()
            results = pool.starmap_async(output,ls)
            pool.close()
            pool.join()
            results,idx = list(np.unique(results.get(),return_inverse = True,axis = 1))
            results = [list(res[idx]) for res in results]
            results = [[int(res[0]),float(res[1]),float(res[3]),float(res[4])] for res in results]
            results = pd.DataFrame(data = results, columns = ["Window","Limiting Factor","Acc","Cov"])
            results = results.loc[(results["Cov"]>0.05)]
            strat_results['strat6']['window'] = results.loc[results["Acc"].idxmax(),"Window"]
            strat_results['strat6']['limiting_factor'] = results.loc[results["Acc"].idxmax(),"Limiting Factor"]
            
            activation_params['strat6'] = False
            activation_params['strat7'] = True
            limiting_factor = np.arange(1.8,2.45,0.1)
            window = range(10,51,2)
            comb = product(window,limiting_factor)
            ls = [list([ts,ground_truth[:-5],strategy_params,activation_params,list(com)]) for com in comb]
            pool = pl()
            results = pool.starmap_async(output,ls)
            pool.close()
            pool.join()
            results,idx = list(np.unique(results.get(),return_inverse = True,axis = 1))
            results = [list(res[idx]) for res in results]
            results = [[int(res[0]),float(res[1]),float(res[3]),float(res[4])] for res in results]
            results = pd.DataFrame(data = results, columns = ["Window","Limiting Factor","Acc","Cov"])
            results = results.loc[(results["Cov"]>0.05)]
            strat_results['strat7']['window'] = results.loc[results["Acc"].idxmax(),"Window"]
            strat_results['strat7']['limiting_factor'] = results.loc[results["Acc"].idxmax(),"Limiting Factor"]

            activation_params['strat7'] = False
            activation_params['strat9'] = True
            comb = list(permutations(range(10,51,2),3))
            ls = [list([ts,ground_truth[:-5],strategy_params,activation_params,list(com)]) for com in comb]
            pool = pl()
            results = pool.starmap_async(output,ls)
            pool.close()
            pool.join()
            results,idx = list(np.unique([i for i in results.get() if i],return_inverse = True,axis = 1))
            results = [res[idx] for res in results]
            results = [[int(res[0]),int(res[1]),int(res[2]),float(res[4]),float(res[5])] for res in results]
            results = pd.DataFrame(data = results, columns = ["Slow Window","Fast Window","MACD Length","Acc","Cov"])
            results = results.loc[(results["Cov"]>0.05)]
            strat_results['strat9']['SlowLength'] = results.loc[results["Acc"].idxmax(),"Slow Window"]
            strat_results['strat9']['FastLength'] = results.loc[results["Acc"].idxmax(),"Fast Window"]
            strat_results['strat9']['MACDLength'] = results.loc[results["Acc"].idxmax(),"MACD Length"]

            ts = time_series.loc[(time_series.index >= split_date[1])&(time_series.index < split_date[2])]
            activation_params = {'strat3':True, 'strat6':False, 'strat7':False, 'strat9': False}
            results = output(ts,ground_truth[:-5],strategy_params,activation_params,[strat_results['strat3']['high window']], check = False)
            out.write(','.join(results)+"\n")
            results = output(ts,ground_truth[:-5],strategy_params,activation_params,[strat_results['strat3']['close window']], check = False)
            out.write(','.join(results)+"\n\n\n")
            activation_params = {'strat3':False, 'strat6':True, 'strat7':False, 'strat9': False}
            results = output(ts,ground_truth[:-5],strategy_params,activation_params,[strat_results['strat6']['window'],strat_results['strat6']['limiting_factor']], check = False)
            out.write(','.join(results)+"\n\n\n")
            activation_params = {'strat3':False, 'strat6':False, 'strat7':True, 'strat9': False}
            results = output(ts,ground_truth[:-5],strategy_params,activation_params,[strat_results['strat7']['window'],strat_results['strat7']['limiting_factor']], check = False)
            out.write(','.join(results)+"\n\n\n")
            activation_params = {'strat3':False, 'strat6':False, 'strat7':False, 'strat9': True}
            results = output(ts,ground_truth[:-5],strategy_params,activation_params,[strat_results['strat9']['SlowLength'],strat_results['strat9']['FastLength'],strat_results['strat9']['MACDLength']], check = False)
            out.write(','.join(results)+"\n\n\n")
        
        out.close()
        

                



