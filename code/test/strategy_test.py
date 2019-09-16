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
from copy import copy,deepcopy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from utils.read_data import read_data_NExT, process_missing_value_v3,read_data_v5_4E
from utils.construct_data import labelling_v1, deal_with_abnormal_value_v1, rolling_half_year
from utils.process_strategy import *

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
    parser.add_argument('-s','--steps',type=int,default=5,
                        help='steps in the future to be predicted')
    parser.add_argument('-gt', '--args.ground_truth', help='ground truth column',
                        type=str, default="LME_Co_Spot")
    parser.add_argument('-out','--output',type = str, default= "test1.csv",help ="output name")
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "NExT"
    )
    parser.add_argument(
        '-m','--min', help='minimum coverage', type = float, default = 0.1
    )
    parser.add_argument(
        '-cov_inc', '--cov_increase',help="coverage increment", type = float, default = 0.1
    )

    args = parser.parse_args()
    all_strat_results = {"Co":{"1d":{},"3d":{},"5d":{}},"Al":{"1d":{},"3d":{},"5d":{}},"Zi":{"1d":{},"3d":{},"5d":{}},"Ti":{"1d":{},"3d":{},"5d":{}},"Le":{"1d":{},"3d":{},"5d":{}},"Ni":{"1d":{},"3d":{},"5d":{}}}
    for ground_truth in ["LME_Co_Spot","LME_Al_Spot","LME_Zi_Spot","LME_Ti_Spot","LME_Le_Spot","LME_Ni_Spot"]:
        for horizon in [1,3,5]:
            args.ground_truth = ground_truth
            args.steps = horizon
            with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
                fname_columns = json.load(fin)[0]

            if args.source == "NExT":
                data_list, LME_dates = read_data_NExT(fname_columns, "1995-11-12")
                time_series = pd.concat(data_list, axis = 1, sort = True)
                split_dates = ["1997-01-01","2017-01-01","2019-01-01"]
                test_split_dates = rolling_half_year(split_dates[0],split_dates[2],5)
                test_split_dates = test_split_dates[-5:]
            elif args.source == "4E":
                time_series, LME_dates = read_data_v5_4E("1995-11-12")
                split_dates = ["1997-01-01","2017-01-01","2019-01-01"]
                test_split_dates = rolling_half_year(split_dates[0],split_dates[2],5)
                test_split_dates = test_split_dates[-4:]
            n = 0
            time_series = deal_with_abnormal_value_v1(time_series)
            LME_dates = sorted(set(LME_dates).intersection(time_series.index.values.tolist()))
            time_series = time_series.loc[LME_dates]
            org_cols = set(time_series.columns.values.tolist())
            labels = labelling_v1(time_series,args.steps,[args.ground_truth])
            time_series = process_missing_value_v3(time_series)
            ts = pd.concat([time_series, labels[0]], axis = 1)
            # ts = time_series.loc[(time_series.index >= split_date[0])&(time_series.index <= split_date[1])]
            strategy_params = {'sar':{'initial':[],'maximum':[]},'rsi':{'window':[],'upper':[],'lower':[]},'strat1':{'short window':[],"med window":[]},'strat2':{'window':[]},'strat3_high':{'window':[]},'strat3_close':{'window':[]},'strat6':{'window':[],'limiting_factor':[]},'strat7':{'window':[],'limiting_factor':[]}, 'strat9':{'SlowLength':[],'FastLength':[],'MACDLength':[]}}
            activation_params = {'sar':True,'rsi':False,'strat1':False,'strat2':False,'strat3_high':False,'strat3_close':False, 'strat6':False, 'strat7':False, 'strat9': False}
            strat_results = {'sar':{'initial':[],'maximum':[]},'rsi':{'window':[],'upper':[],'lower':[]},'strat1':{'short window':[],"med window":[]},'strat2':{'window':[]},'strat3_high':{'window':[]}, 'strat3_close':{'window':[]},'strat6':{'window':[],'limiting_factor':[]},'strat7':{'window':[],'limiting_factor':[]}, 'strat9':{'SlowLength':[],'FastLength':[],'MACDLength':[]}}
            
            print("sar")
            initial = np.arange(0.01,0.051,0.002)
            mx = np.arange(0.1,0.51,0.02)
            comb = product(initial,mx)
            sar = parallel_process(copy(ts), split_dates, "sar", strat_results, args.ground_truth, strategy_params,activation_params,args.cov_increase,comb,args.min)
            
            
            print("rsi")
            activation_params['sar'] = False
            activation_params['rsi'] = True
            window = range(5,51,2)
            upper = range(60,91,10) 
            lower = range(20,51,10)
            comb = product(window, upper,lower)
            rsi = parallel_process(copy(ts), split_dates, "rsi", strat_results, args.ground_truth, strategy_params,activation_params,args.cov_increase,comb,args.min)

            print("strat1")
            activation_params['rsi'] = False
            activation_params['strat1'] = True
            short = range(20,35,2)
            med = range(50,71,2)
            comb = product(short,med)
            strat1 = parallel_process(copy(ts), split_dates, "strat1", strat_results, args.ground_truth, strategy_params,activation_params,args.cov_increase,comb,args.min)
            
            print("strat2")
            activation_params['strat1'] = False
            activation_params['strat2'] = True
            comb = list(range(45,61,2))
            comb = [[com] for com in comb]
            strat2 = parallel_process(copy(ts), split_dates, "strat2", strat_results, args.ground_truth, strategy_params,activation_params,args.cov_increase,comb,args.min)

            print("strat3_high")
            activation_params['strat2'] = False
            activation_params['strat3_high'] = True
            comb = list(range(5,51,2))
            comb = [[com] for com in comb]
            strat3 = parallel_process(copy(ts), split_dates, "strat3_high", strat_results, args.ground_truth, strategy_params,activation_params,args.cov_increase,comb,args.min)

            print("strat3_close")
            activation_params['strat3_high'] = False
            activation_params['strat3_close'] = True
            comb = list(range(5,51,2))
            comb = [[com] for com in comb]
            strat3 = parallel_process(copy(ts), split_dates, "strat3_close", strat_results, args.ground_truth, strategy_params,activation_params,args.cov_increase,comb,args.min)

            print("strat6")
            activation_params['strat3_close'] = False
            activation_params['strat6'] = True
            limiting_factor = np.arange(0.3,1.05,0.1)
            window = range(10,51,2)
            comb = product(window,limiting_factor)
            strat6 = parallel_process(copy(ts), split_dates, "strat6", strat_results, args.ground_truth, strategy_params,activation_params,args.cov_increase,comb,args.min)
            
            print("strat7")
            activation_params['strat6'] = False
            activation_params['strat7'] = True
            limiting_factor = np.arange(1.8,2.45,0.1)
            window = range(10,51,2)
            comb = product(window,limiting_factor)
            strat7 = parallel_process(copy(ts), split_dates, "strat7", strat_results, args.ground_truth, strategy_params,activation_params,args.cov_increase,comb,args.min)
            
            print('strat9')
            activation_params['strat7'] = False
            activation_params['strat9'] = True
            comb = list(permutations(range(10,51,5),3))
            strat9 = parallel_process(copy(ts), split_dates, "strat9", strat_results, args.ground_truth, strategy_params,activation_params,args.cov_increase,comb,args.min)
            

            ans = {'index':[],
                    'sar_initial':[],'sar_maximum':[],'sar_train_acc':[],'sar_train_cov':[],'sar_acc':[],'sar_cov':[],
                    'rsi_window':[],'rsi_upper':[],'rsi_lower':[],'rsi_train_acc':[],'rsi_train_cov':[],'rsi_acc':[],'rsi_cov':[],
                    'strat1_short_window':[],'strat1_med_window':[],'strat1_train_acc':[],'strat1_train_cov':[],'strat1_acc':[],'strat1_cov':[],
                    'strat2_window':[],'strat2_train_acc':[],'strat2_train_cov':[],'strat2_acc':[],'strat2_cov':[],
                    'strat3_high_window':[],'strat3_train_high_acc':[],'strat3_train_high_cov':[],'strat3_high_acc':[],'strat3_high_cov':[],
                    'strat3_close_window':[],'strat3_train_close_acc':[],'strat3_train_close_cov':[],'strat3_close_acc':[],'strat3_close_cov':[],
                    'strat6_window':[],'strat6_limiting_factor':[],'strat6_train_acc':[],'strat6_train_cov':[],'strat6_acc':[],'strat6_cov':[],
                    'strat7_window':[],'strat7_limiting_factor':[],'strat7_train_acc':[],'strat7_train_cov':[],'strat7_acc':[],'strat7_cov':[],
                    'strat9_slow_length':[],'strat9_fast_length':[],'strat9_macd_length':[],'strat9_train_acc':[],'strat9_train_cov':[],'strat9_acc':[],'strat9_cov':[]
                }

            for strat in strat_results.keys():
                for param in strat_results[strat].keys():
                    if "window" in param or "Length" in param or "upper" in param or "lower" in param:
                        strat_results[strat][param] = [int(el) for el in strat_results[strat][param]]
                    else:
                        strat_results[strat][param] = [float(el) for el in strat_results[strat][param]]
            all_strat_results[ground_truth.split("_")[1]][str(horizon)+"d"] = deepcopy(strat_results)
            mx = max([len(list(strat_results[strat].values())[0]) for strat in strat_results.keys()])  
            
            for test_split_date in test_split_dates:
                print(test_split_date)
                ans['index'] = ans['index']+[test_split_date[1]]*mx
                # ts = time_series.loc[(time_series.index >= test_split_date[1])&(time_series.index < test_split_date[2])]
                activation_params = {'sar':True,'rsi':False,'strat1':False,'strat2':False,'strat3_high':False, 'strat3_close':False,'strat6':False, 'strat7':False, 'strat9': False}
                for i in range(mx):
                    if i < len(strat_results['sar']['initial']):
                        strat_dc = create_dc_from_comb("sar",strategy_params,[strat_results['sar']['initial'][i],strat_results['sar']['maximum'][i]])
                        train_results = output(ts, split_dates, args.ground_truth, strategy_params,activation_params,strat_dc)
                        results = output(ts, test_split_date,args.ground_truth,strategy_params,activation_params,strat_dc, check = False)
                        ans['sar_initial'].append(strat_results['sar']['initial'][i])
                        ans['sar_maximum'].append(strat_results['sar']['maximum'][i])
                        ans['sar_train_acc'].append(train_results[3])
                        ans['sar_train_cov'].append(train_results[4])
                        ans['sar_acc'].append(results[3])
                        ans['sar_cov'].append(results[4])   
                    else:
                        ans['sar_initial'].append(None)
                        ans['sar_maximum'].append(None)
                        ans['sar_train_acc'].append(None)
                        ans['sar_train_cov'].append(None)
                        ans['sar_acc'].append(None)
                        ans['sar_cov'].append(None)     

                
                activation_params = {'sar':False,'rsi':True,'strat1':False,'strat2':False,'strat3_high':False, 'strat3_close':False,'strat6':False, 'strat7':False, 'strat9': False}
                for i in range(mx):
                    if i < len(strat_results['rsi']['window']):    
                        strat_dc = create_dc_from_comb("rsi",strategy_params,[strat_results['rsi']['window'][i],strat_results['rsi']['upper'][i],strat_results['rsi']['lower'][i]])
                        train_results = output(ts, split_dates,args.ground_truth,strategy_params,activation_params,strat_dc)
                        results = output(ts, test_split_date,args.ground_truth,strategy_params,activation_params,strat_dc, check = False)
                        ans['rsi_window'].append(strat_results['rsi']['window'][i])
                        ans['rsi_upper'].append(strat_results['rsi']['upper'][i])
                        ans['rsi_lower'].append(strat_results['rsi']['lower'][i])
                        ans['rsi_train_acc'].append(train_results[4])
                        ans['rsi_train_cov'].append(train_results[5])
                        ans['rsi_acc'].append(results[4])
                        ans['rsi_cov'].append(results[5])
                    else:
                        ans['rsi_window'].append(None)
                        ans['rsi_upper'].append(None)
                        ans['rsi_lower'].append(None)
                        ans['rsi_train_acc'].append(None)
                        ans['rsi_train_cov'].append(None)
                        ans['rsi_acc'].append(None)
                        ans['rsi_cov'].append(None)

                activation_params = {'sar':False,'rsi':False,'strat1':True,'strat2':False,'strat3_high':False, 'strat3_close':False,'strat6':False, 'strat7':False, 'strat9': False}
                
                for i in range(mx):
                    if i < len(strat_results['strat1']['short window']):
                        strat_dc = create_dc_from_comb("strat1",strategy_params,[strat_results['strat1']['short window'][i],strat_results['strat1']['med window'][i]])
                        train_results = output(ts, split_dates,args.ground_truth,strategy_params,activation_params,strat_dc)
                        results = output(ts, test_split_date,args.ground_truth,strategy_params,activation_params,strat_dc, check = False)
                        ans['strat1_short_window'].append(strat_results['strat1']['short window'][i])
                        ans['strat1_med_window'].append(strat_results['strat1']['med window'][i])
                        ans['strat1_train_acc'].append(train_results[3])
                        ans['strat1_train_cov'].append(train_results[4])
                        ans['strat1_acc'].append(results[3])
                        ans['strat1_cov'].append(results[4])    
                    else:
                        ans['strat1_short_window'].append(None)
                        ans['strat1_med_window'].append(None)
                        ans['strat1_train_acc'].append(None)
                        ans['strat1_train_cov'].append(None)
                        ans['strat1_acc'].append(None)
                        ans['strat1_cov'].append(None)
                activation_params = {'sar':False,'rsi':False,'strat1':False,'strat2':True,'strat3_high':False, 'strat3_close':False,'strat6':False, 'strat7':False, 'strat9': False}
                
                for i in range(mx):
                    if i < len(strat_results['strat2']['window']):
                        strat_dc = create_dc_from_comb("strat2",strategy_params,[strat_results['strat2']['window'][i]])
                        train_results = output(ts, split_dates,args.ground_truth,strategy_params,activation_params,strat_dc)
                        results = output(ts, test_split_date,args.ground_truth,strategy_params,activation_params,strat_dc, check = False)
                        ans['strat2_window'].append(strat_results['strat2']['window'][i])
                        ans['strat2_train_acc'].append(train_results[2])
                        ans['strat2_train_cov'].append(train_results[3])
                        ans['strat2_acc'].append(results[2])
                        ans['strat2_cov'].append(results[3])
                    else:
                        ans['strat2_window'].append(None)
                        ans['strat2_train_acc'].append(None)
                        ans['strat2_train_cov'].append(None)
                        ans['strat2_acc'].append(None)
                        ans['strat2_cov'].append(None)
                    
                activation_params = {'sar':False,'rsi':False,'strat1':False,'strat2':False,'strat3_high':True, 'strat3_close':False,'strat6':False, 'strat7':False, 'strat9': False}
                
                for i in range(mx):
                    if i < len(strat_results['strat3_high']['window']):
                        strat_dc = create_dc_from_comb("strat3_high",strategy_params,[strat_results['strat3_high']['window'][i]])
                        train_results = output(ts, split_dates,args.ground_truth,strategy_params,activation_params,strat_dc)
                        results = output(ts, test_split_date,args.ground_truth,strategy_params,activation_params,strat_dc, check = False)
                        ans['strat3_high_window'].append(strat_results['strat3_high']['window'][i])
                        ans['strat3_train_high_acc'].append(train_results[2])
                        ans['strat3_train_high_cov'].append(train_results[3])
                        ans['strat3_high_acc'].append(results[2])
                        ans['strat3_high_cov'].append(results[3]) 
                    else:
                        ans['strat3_high_window'].append(None)
                        ans['strat3_train_high_acc'].append(None)
                        ans['strat3_train_high_cov'].append(None)
                        ans['strat3_high_acc'].append(None)
                        ans['strat3_high_cov'].append(None) 
            
                activation_params = {'sar':False,'rsi':False,'strat1':False,'strat2':False,'strat3_high':False, 'strat3_close':True,'strat6':False, 'strat7':False, 'strat9': False}
                
                for i in range(mx):
                    if i < len(strat_results['strat3_close']['window']):
                        strat_dc = create_dc_from_comb("strat3_close",strategy_params,[strat_results['strat3_close']['window'][i]])
                        train_results = output(ts, split_dates,args.ground_truth,strategy_params,activation_params,strat_dc)
                        results = output(ts, test_split_date,args.ground_truth,strategy_params,activation_params,strat_dc, check = False)
                        ans['strat3_close_window'].append(strat_results['strat3_close']['window'][i])
                        ans['strat3_train_close_acc'].append(train_results[2])
                        ans['strat3_train_close_cov'].append(train_results[3])
                        ans['strat3_close_acc'].append(results[2])
                        ans['strat3_close_cov'].append(results[3]) 
                    else:
                        ans['strat3_close_window'].append(None)
                        ans['strat3_train_close_acc'].append(None)
                        ans['strat3_train_close_cov'].append(None)
                        ans['strat3_close_acc'].append(None)
                        ans['strat3_close_cov'].append(None) 
                
                activation_params = {'sar':False,'rsi':False,'strat1':False,'strat2':False,'strat3_high':False, 'strat3_close':False,'strat6':True, 'strat7':False, 'strat9': False}
                
                for i in range(mx):
                    if i < len(strat_results['strat6']['window']):
                        strat_dc = create_dc_from_comb('strat6',strategy_params,[strat_results['strat6']['window'][i],strat_results['strat6']['limiting_factor'][i]])
                        train_results = output(ts, split_dates,args.ground_truth,strategy_params,activation_params,strat_dc)
                        results = output(ts, test_split_date,args.ground_truth,strategy_params,activation_params,strat_dc, check = False)
                        ans['strat6_window'].append(strat_results['strat6']['window'][i])
                        ans['strat6_limiting_factor'].append(strat_results['strat6']['limiting_factor'][i])
                        ans['strat6_train_acc'].append(train_results[3])
                        ans['strat6_train_cov'].append(train_results[4])
                        ans['strat6_acc'].append(results[3])
                        ans['strat6_cov'].append(results[4]) 
                    else:
                        ans['strat6_window'].append(None)
                        ans['strat6_limiting_factor'].append(None)
                        ans['strat6_train_acc'].append(None)
                        ans['strat6_train_cov'].append(None)
                        ans['strat6_acc'].append(None)
                        ans['strat6_cov'].append(None) 
                activation_params = {'sar':False,'rsi':False,'strat1':False,'strat2':False,'strat3_high':False, 'strat3_close':False,'strat6':False, 'strat7':True, 'strat9': False}
                
                for i in range(mx):
                    
                    if i < len(strat_results['strat7']['window']):
                        strat_dc = create_dc_from_comb("strat7",strategy_params,[strat_results['strat7']['window'][i],strat_results['strat7']['limiting_factor'][i]])
                        train_results = output(ts, split_dates,args.ground_truth,strategy_params,activation_params,strat_dc)
                        results = output(ts, test_split_date,args.ground_truth,strategy_params,activation_params,strat_dc, check = False)
                        ans['strat7_window'].append(strat_results['strat7']['window'][i])
                        ans['strat7_limiting_factor'].append(strat_results['strat7']['limiting_factor'][i])
                        ans['strat7_train_acc'].append(train_results[3])
                        ans['strat7_train_cov'].append(train_results[4])
                        ans['strat7_acc'].append(results[3])
                        ans['strat7_cov'].append(results[4]) 
                    else:
                        ans['strat7_window'].append(None)
                        ans['strat7_limiting_factor'].append(None)
                        ans['strat7_train_acc'].append(None)
                        ans['strat7_train_cov'].append(None)
                        ans['strat7_acc'].append(None)
                        ans['strat7_cov'].append(None) 
                activation_params = {'sar':False,'rsi':False,'strat1':False,'strat2':False,'strat3_high':False, 'strat3_close':False,'strat6':False, 'strat7':False, 'strat9': True}
                
                for i in range(mx):
                    if i < len(strat_results['strat9']['SlowLength']):
                        strat_dc = create_dc_from_comb("strat9",strategy_params,[strat_results['strat9']['SlowLength'][i],strat_results['strat9']['FastLength'][i],strat_results['strat9']['MACDLength'][i]])
                        train_results = output(ts, split_dates,args.ground_truth,strategy_params,activation_params,strat_dc)
                        results = output(ts, test_split_date,args.ground_truth,strategy_params,activation_params,strat_dc, check = False)
                        ans['strat9_slow_length'].append(strat_results['strat9']['SlowLength'][i])
                        ans['strat9_fast_length'].append(strat_results['strat9']['FastLength'][i])
                        ans['strat9_macd_length'].append(strat_results['strat9']['MACDLength'][i])
                        ans['strat9_train_acc'].append(train_results[4])
                        ans['strat9_train_cov'].append(train_results[5])
                        ans['strat9_acc'].append(results[4])
                        ans['strat9_cov'].append(results[5])
                    else:
                        ans['strat9_slow_length'].append(None)
                        ans['strat9_fast_length'].append(None)
                        ans['strat9_macd_length'].append(None)
                        ans['strat9_train_acc'].append(None)
                        ans['strat9_train_cov'].append(None)
                        ans['strat9_acc'].append(None)
                        ans['strat9_cov'].append(None)
            ans = pd.DataFrame(ans)
            # ans.to_csv(args.output)

    with open("strat_params_y20_h"+str(args.steps)+"_gt"+args.ground_truth+".conf","w") as f:
        json.dump(all_strat_results,f,indent = 4)

    

            



