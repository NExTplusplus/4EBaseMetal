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
    with open(fname+".csv","w") as out:
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
    ts = ts[sorted(list(set(ts.columns.values.tolist()) - org_cols))]
    temp_list = array
#     if not check:
#         save_data("rolling_"+ground_truth+str(sp[strat]),ts,ts.columns.values.tolist())
    
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
        if len(labels) == 0:
            compared = 0
        else:
            compared = abs(sum(labels == column)/len(labels)-0.5)
        # if compared < 0.025 and check:
        #     return [""]
        temp_list.append(compared)
        temp_list.append(len(labels)/length)
    temp_list = [str(e) for e in temp_list]
    
    return temp_list

def parallel_process(ts,strat,cov,strat_results,ground_truth,strategy_params,activation_params,combination):
    ls = [list([ts,ground_truth,strategy_params,activation_params,list(com)]) for com in comb]
    pool = pl()
    results = pool.starmap_async(output,ls)
    pool.close()
    pool.join()
    results,idx = list(np.unique([i for i in results.get() if i],return_inverse = True,axis = 1))
    results = [list(res[idx]) for res in results]
    if strat == 'sar':
        results = [[float(res[0]),float(res[1]),float(res[3]),float(res[4])] for res in results]
        results = pd.DataFrame(data = results, columns = ["Initial","Maximum","Acc","Cov"])
        results = results.loc[(results["Cov"]>cov)]
        strat_results['sar']['initial'] = results.loc[results["Acc"].idxmax(),"Initial"]
        strat_results['sar']['maximum'] = results.loc[results["Acc"].idxmax(),"Maximum"]

    elif strat =='rsi':
        results = [[int(res[0]),int(res[1]),int(res[2]),float(res[4]),float(res[5])] for res in results]
        results = pd.DataFrame(data = results, columns = ["Window","Upper","Lower","Acc","Cov"])
        results = results.loc[(results["Cov"]>cov)]
        strat_results['rsi']['window'] = results.loc[results["Acc"].idxmax(),"Window"]
        strat_results['rsi']['upper'] = results.loc[results["Acc"].idxmax(),"Upper"]
        strat_results['rsi']['lower'] = results.loc[results["Acc"].idxmax(),"Lower"]
    
    elif strat == 'strat1':
        results = [[int(res[0]),int(res[1]),float(res[3]),float(res[4])] for res in results]
        results = pd.DataFrame(data = results, columns = ["Short Window","Med Window","Acc","Cov"])
        results = results.loc[(results["Cov"]>cov)]
        strat_results['strat1']['short window'] = results.loc[results["Acc"].idxmax(),"Short Window"]
        strat_results['strat1']['med window'] = results.loc[results["Acc"].idxmax(),"Med Window"]

    elif strat == 'strat2':
        results = [[int(res[0]),float(res[2]),float(res[3])] for res in results]
        results = pd.DataFrame(data = results, columns = ["Window","Acc","Cov"])
        results = results.loc[(results["Cov"]>cov)]
        strat_results['strat2']['window'] = results.loc[results["Acc"].idxmax(),"Window"]

    elif strat == "strat3":
        results = [[int(res[0]),float(res[2]),float(res[3]),float(res[5]),float(res[6])] for res in results]
        results = pd.DataFrame(data = results, columns = ["Window","Close Acc","Close Cov","High Acc","High Cov"])
        results = results.loc[(results["High Cov"]>cov)&(results["Close Cov"]>cov)]
        strat_results['strat3']['high window'] = results.loc[results["High Acc"].idxmax(),"Window"]
        strat_results['strat3']['close window'] = results.loc[results["Close Acc"].idxmax(),"Window"]
    
    elif strat == "strat6":
        results = [[int(res[0]),float(res[1]),float(res[3]),float(res[4])] for res in results]
        results = pd.DataFrame(data = results, columns = ["Window","Limiting Factor","Acc","Cov"])
        results = results.loc[(results["Cov"]>cov)]
        strat_results['strat6']['window'] = results.loc[results["Acc"].idxmax(),"Window"]
        strat_results['strat6']['limiting_factor'] = results.loc[results["Acc"].idxmax(),"Limiting Factor"]
    
    elif strat == "strat7":
        results = [[int(res[0]),float(res[1]),float(res[3]),float(res[4])] for res in results]
        results = pd.DataFrame(data = results, columns = ["Window","Limiting Factor","Acc","Cov"])
        results = results.loc[(results["Cov"]>cov)]
        strat_results['strat7']['window'] = results.loc[results["Acc"].idxmax(),"Window"]
        strat_results['strat7']['limiting_factor'] = results.loc[results["Acc"].idxmax(),"Limiting Factor"]
    
    elif strat == "strat9":
        results = [[int(res[0]),int(res[1]),int(res[2]),float(res[4]),float(res[5])] for res in results]
        results = pd.DataFrame(data = results, columns = ["Slow Window","Fast Window","MACD Length","Acc","Cov"])
        results = results.loc[(results["Cov"]>cov)]
        strat_results['strat9']['SlowLength'] = results.loc[results["Acc"].idxmax(),"Slow Window"]
        strat_results['strat9']['FastLength'] = results.loc[results["Acc"].idxmax(),"Fast Window"]
        strat_results['strat9']['MACDLength'] = results.loc[results["Acc"].idxmax(),"MACD Length"]
    return strat_results



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

    if args.source == "NExT":
        data_list, LME_dates = read_data_NExT(fname_columns, "2003-11-12")
        time_series = pd.concat(data_list, axis = 1, sort = True)
        split_dates = rolling_half_year("2003-01-01","2017-01-01",5)
    elif args.source == "4E":
        time_series, LME_dates = read_data_v5_4E("2003-11-12")
        split_dates = rolling_half_year("2003-01-01","2019-01-01",5)
    
    n = 0
    time_series = deal_with_abnormal_value_v1(time_series)
    LME_dates = sorted(set(LME_dates).intersection(time_series.index.values.tolist()))
    time_series = time_series.loc[LME_dates]
    org_cols = set(time_series.columns.values.tolist())
    labels = labelling_v1(time_series,args.steps,[args.ground_truth])
    time_series = process_missing_value_v3(time_series)
    time_series = pd.concat([time_series, labels[0]], axis = 1)
    
    split_dates = split_dates[-4:]
    ans = {'index':['2017-01-01','2017-07-01','2018-01-01','2018-07-01'],
                'sar_initial':[],'sar_maximum':[],'sar_acc':[],'sar_cov':[],
                'rsi_window':[],'rsi_upper':[],'rsi_lower':[],'rsi_acc':[],'rsi_cov':[],
                'strat1_short_window':[],'strat1_med_window':[],'strat1_acc':[],'strat1_cov':[],
                'strat2_window':[],'strat2_acc':[],'strat2_cov':[],
                'strat3_high_window':[],'strat3_high_acc':[],'strat3_high_cov':[],
                'strat3_close_window':[],'strat3_close_acc':[],'strat3_close_cov':[],
                'strat6_window':[],'strat6_limiting_factor':[],'strat6_acc':[],'strat6_cov':[],
                'strat7_window':[],'strat7_limiting_factor':[],'strat7_acc':[],'strat7_cov':[],
                'strat9_slow_length':[],'strat9_fast_length':[],'strat9_macd_length':[],'strat9_acc':[],'strat9_cov':[]
                }
    for split_date in split_dates:
        print(split_date)
        ts = time_series.loc[(time_series.index >= split_date[0])&(time_series.index <= split_date[1])]
        strategy_params = {'sar':{'initial':0,'maximum':0},'rsi':{'window':0,'upper':0,'lower':0},'strat1':{'short window':0,"med window":0},'strat2':{'window':0},'strat3':{'window':0},'strat6':{'window':0,'limiting_factor':0},'strat7':{'window':0,'limiting_factor':0}, 'strat9':{'SlowLength':0,'FastLength':0,'MACDLength':0}}
        activation_params = {'sar':True,'rsi':False,'strat1':False,'strat2':False,'strat3':False, 'strat6':False, 'strat7':False, 'strat9': False}
        strat_results = {'sar':{'initial':0,'maximum':0},'rsi':{'window':0,'upper':0,'lower':0},'strat1':{'short window':0,"med window":0},'strat2':{'window':0},'strat3':{'high window':0, 'close window':0},'strat6':{'window':0,'limiting_factor':0},'strat7':{'window':0,'limiting_factor':0}, 'strat9':{'SlowLength':0,'FastLength':0,'MACDLength':0}}
        
        print("sar")
        initial = np.arange(0.01,0.051,0.002)
        mx = np.arange(0.1,0.51,0.02)
        comb = product(initial,mx)
        strat_results = parallel_process(ts, "sar", 0.05, strat_results, ground_truth, strategy_params,activation_params,comb)
        
        print("rsi")
        activation_params['sar'] = False
        activation_params['rsi'] = True
        window = range(5,51,2)
        upper = range(60,91,2)
        lower = range(20,51,2)
        comb = product(window, upper,lower)
        strat_results = parallel_process(ts, "rsi", 0.05, strat_results, ground_truth, strategy_params,activation_params,comb)

        print("strat1")
        activation_params['rsi'] = False
        activation_params['strat1'] = True
        short = range(20,35,2)
        med = range(50,71,2)
        comb = product(short,med)
        strat_results = parallel_process(ts, "strat1", 0.05, strat_results, ground_truth, strategy_params,activation_params,comb)

        print("strat2")
        activation_params['strat1'] = False
        activation_params['strat2'] = True
        comb = list(range(45,61,2))
        comb = [[com] for com in comb]
        strat_results = parallel_process(ts, "strat2", 0.05, strat_results, ground_truth, strategy_params,activation_params,comb)

        print("strat3")
        activation_params['strat2'] = False
        activation_params['strat3'] = True
        comb = list(range(5,51,2))
        comb = [[com] for com in comb]
        strat_results = parallel_process(ts, "strat3", 0.05, strat_results, ground_truth, strategy_params,activation_params,comb)
        
        print("strat6")
        activation_params['strat3'] = False
        activation_params['strat6'] = True
        limiting_factor = np.arange(0.3,1.05,0.1)
        window = range(10,51,2)
        comb = product(window,limiting_factor)
        strat_results = parallel_process(ts, "strat6", 0.05, strat_results, ground_truth, strategy_params,activation_params,comb)
        
        print("strat7")
        activation_params['strat6'] = False
        activation_params['strat7'] = True
        limiting_factor = np.arange(1.8,2.45,0.1)
        window = range(10,51,2)
        comb = product(window,limiting_factor)
        strat_results = parallel_process(ts, "strat7", 0.05, strat_results, ground_truth, strategy_params,activation_params,comb)

        print('strat9')
        activation_params['strat7'] = False
        activation_params['strat9'] = True
        comb = list(permutations(range(10,51,2),3))
        strat_results = parallel_process(ts, "strat9", 0.05, strat_results, ground_truth, strategy_params,activation_params,comb)
        

        ts = time_series.loc[(time_series.index >= split_date[1])&(time_series.index < split_date[2])]
        activation_params = {'sar':True,'rsi':False,'strat1':False,'strat2':False,'strat3':False, 'strat6':False, 'strat7':False, 'strat9': False}
        results = output(ts,ground_truth,strategy_params,activation_params,[strat_results['sar']['initial'],strat_results['sar']['maximum']], check = False)
        ans['sar_initial'].append(strat_results['sar']['initial'])
        ans['sar_maximum'].append(strat_results['sar']['maximum'])
        ans['sar_acc'].append(results[3])
        ans['sar_cov'].append(results[4])            
        activation_params = {'sar':False,'rsi':True,'strat1':False,'strat2':False,'strat3':False, 'strat6':False, 'strat7':False, 'strat9': False}
        results = output(ts,ground_truth,strategy_params,activation_params,[strat_results['rsi']['window'],strat_results['rsi']['upper'],strat_results['rsi']['lower']], check = False)
        ans['rsi_window'].append(strat_results['rsi']['window'])
        ans['rsi_upper'].append(strat_results['rsi']['upper'])
        ans['rsi_lower'].append(strat_results['rsi']['lower'])
        ans['rsi_acc'].append(results[4])
        ans['rsi_cov'].append(results[5])
        activation_params = {'sar':False,'rsi':False,'strat1':True,'strat2':False,'strat3':False, 'strat6':False, 'strat7':False, 'strat9': False}
        results = output(ts,ground_truth,strategy_params,activation_params,[strat_results['strat1']['short window'],strat_results['strat1']['med window']], check = False)
        ans['strat1_short_window'].append(strat_results['strat1']['short window'])
        ans['strat1_med_window'].append(strat_results['strat1']['med window'])
        ans['strat1_acc'].append(results[3])
        ans['strat1_cov'].append(results[4])    
        activation_params = {'sar':False,'rsi':False,'strat1':False,'strat2':True,'strat3':False, 'strat6':False, 'strat7':False, 'strat9': False}
        results = output(ts,ground_truth,strategy_params,activation_params,[strat_results['strat2']['window']], check = False)
        ans['strat2_window'].append(strat_results['strat2']['window'])
        ans['strat2_acc'].append(results[2])
        ans['strat2_cov'].append(results[3]) 
        
        activation_params = {'sar':False,'rsi':False,'strat1':False,'strat2':False,'strat3':True, 'strat6':False, 'strat7':False, 'strat9': False}
        results = output(ts,ground_truth,strategy_params,activation_params,[strat_results['strat3']['high window']], check = False)
        ans['strat3_high_window'].append(strat_results['strat3']['high window'])
        ans['strat3_high_acc'].append(results[5])
        ans['strat3_high_cov'].append(results[6]) 

        results = output(ts,ground_truth,strategy_params,activation_params,[strat_results['strat3']['close window']], check = False)
        ans['strat3_close_window'].append(strat_results['strat3']['close window'])
        ans['strat3_close_acc'].append(results[2])
        ans['strat3_close_cov'].append(results[3])
        
        activation_params = {'sar':False,'rsi':False,'strat1':False,'strat2':False,'strat3':False, 'strat6':True, 'strat7':False, 'strat9': False}
        results = output(ts,ground_truth,strategy_params,activation_params,[strat_results['strat6']['window'],strat_results['strat6']['limiting_factor']], check = False)
        ans['strat6_window'].append(strat_results['strat6']['window'])
        ans['strat6_limiting_factor'].append(strat_results['strat6']['limiting_factor'])
        ans['strat6_acc'].append(results[3])
        ans['strat6_cov'].append(results[4]) 
        activation_params = {'sar':False,'rsi':False,'strat1':False,'strat2':False,'strat3':False, 'strat6':False, 'strat7':True, 'strat9': False}
        results = output(ts,ground_truth,strategy_params,activation_params,[strat_results['strat7']['window'],strat_results['strat7']['limiting_factor']], check = False)
        ans['strat7_window'].append(strat_results['strat7']['window'])
        ans['strat7_limiting_factor'].append(strat_results['strat7']['limiting_factor'])
        ans['strat7_acc'].append(results[3])
        ans['strat7_cov'].append(results[4])
        activation_params = {'sar':False,'rsi':False,'strat1':False,'strat2':False,'strat3':False, 'strat6':False, 'strat7':False, 'strat9': True}
        results = output(ts,ground_truth,strategy_params,activation_params,[strat_results['strat9']['SlowLength'],strat_results['strat9']['FastLength'],strat_results['strat9']['MACDLength']], check = False)
        ans['strat9_slow_length'].append(strat_results['strat9']['SlowLength'])
        ans['strat9_fast_length'].append(strat_results['strat9']['FastLength'])
        ans['strat9_macd_length'].append(strat_results['strat9']['MACDLength'])
        ans['strat9_acc'].append(results[4])
        ans['strat9_cov'].append(results[5])
    ans = pd.DataFrame(ans)
    ans.to_csv("rolling_"+args.output)


        

                



