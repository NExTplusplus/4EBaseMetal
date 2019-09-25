'''
    
'''
import os
import sys


import json
import argparse
import numpy as np
import pandas as pd
from copy import copy,deepcopy
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from utils.read_data import read_data_NExT, process_missing_value_v3,read_data_v5_4E
from utils.construct_data import labelling_v2, deal_with_abnormal_value_v1, rolling_half_year
from utils.Technical_indicator import strategy_7
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
                        type=str, default="LME_Co_Spot,LME_Co_Close")
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "NExT"
    )
    parser.add_argument(
        '-l','--lag', help='lag', type = int, default = 0
    )

    args = parser.parse_args()
    ground_truth = args.ground_truth.split(",")
    steps = [int(i) for i in args.steps.split(",")]
    lags = list(range(args.lag,args.lag+20))
    with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)[0]

    if args.source == "NExT":
        data_list, LME_dates = read_data_NExT(fname_columns, "2000-11-12")
        time_series = pd.concat(data_list, axis = 1, sort = True)
    elif args.source == "4E":
        time_series, LME_dates = read_data_v5_4E("2000-11-12")
    
    time_series = deal_with_abnormal_value_v1(time_series)
    time_series['Spread'] = time_series[ground_truth[0]]-time_series[ground_truth[1]]

#    #plot part
#    ticklabel = time_series.index
#    xticks = np.arange(0,len(ticklabel),100)
#    plt.figure(figsize = (15,12))
#    fig, ax1 = plt.subplots()
#    color = 'tab:red'
#    ax1.plot(time_series.index, time_series[ground_truth[0]],label = ground_truth[0],color=color)
#    ax1.set_ylabel(ground_truth[0], color=color)
#    ax1.tick_params(axis='y',color=color)
#    ax1.set_xticks(xticks)
#    ax1.set_xticklabels(time_series.index,rotation = 45)
#    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#    color = 'tab:blue'
#    ax2.plot(time_series.index, time_series.Spread,label = "Spread",color=color)
#    ax2.set_ylabel("Spread", color=color)
#    ax2.tick_params(axis='y',color=color)
#    ax2.set_xticks(xticks)
#    ax2.set_xticklabels(ticklabel[xticks],rotation = 45)
#    fig.tight_layout()  # otherwise the right y-label is slightly clipped
#    plt.savefig(ground_truth[0]+'_Spread'+'.png')
    time_series["BB"] = strategy_7(time_series.Spread,10,1.8)
    LME_dates = sorted(set(LME_dates).intersection(time_series.index.values.tolist()))
    time_series = time_series.loc[LME_dates]
    org_cols = set(time_series.columns.values.tolist())
    tttarget = labelling_v2(time_series,steps[0],[ground_truth[0]])[0]
    print(tttarget)
    
#    tttbc = labelling_v1(time_series,steps[1],["Spread"])[0]
    tttbc = time_series.BB
    print(tttbc)
    tttarget = tttarget.loc[(tttarget.index<"2008-01-01")|(tttarget.index>="2009-01-01")]
    tttbc = tttbc.loc[(tttbc.index<"2008-01-01")|(tttbc.index>="2009-01-01")]
#    print(tttarget.index)
    correlation = pd.DataFrame()
    for lag in lags:
        print(lag)
        ans = {"start":[],"end":[],"acc_"+str(lag):[],"cov_"+str(lag):[]}
        start_date = "2001-01-01"
        last_date =  "2017-01-01"
        ttbc = tttbc.shift(lag)
        ttbc.dropna(inplace = True)
        ttarget = tttarget.loc[ttbc.index]
        

        ans['start'].append(start_date)
        ans['end'].append(last_date)
        ans['acc_'+str(lag)].append(sum(ttarget.loc[(ttarget.index>=start_date)&(ttarget.index< last_date)] == ttbc.loc[(ttbc.index>=start_date)&(ttbc.index< last_date)])/len(ttarget.loc[(ttarget.index>=start_date)&(ttarget.index< last_date)&(ttbc!=0)]))
        ans['cov_'+str(lag)].append(len(ttarget.loc[ttbc!=0])/len(ttarget))
        while len(ttarget.loc[ttarget.index > start_date])>0 :
#            print(start_date)
            if start_date == "2008-01-01":
                start_date = "2009-01-01"
                continue
            end_date = str(int(start_date.split("-")[0])+1)+"-01-01"
            target = copy(ttarget)
            tbc = copy(ttbc)
            target = target[(target.index >= start_date)&(target.index <end_date)]
            tbc = tbc[(tbc.index >= start_date)&(tbc.index <end_date)]
            ans['start'].append(start_date)
            ans['end'].append(end_date)
            ans['acc_'+str(lag)].append(sum(target == tbc)/len(target.loc[tbc!=0]))
            ans['cov_'+str(lag)].append(len(target.loc[tbc!=0])/len(target))
            start_date = end_date
        ans = pd.DataFrame(ans)
        if correlation.empty:
            correlation = ans
        else:
            correlation = pd.concat([correlation,ans["acc_"+str(lag)],ans["cov_"+str(lag)]],axis = 1)
    correlation.to_csv(args.ground_truth+'_'+str(steps[0])+"_Spread_Strategy.csv")


    

    


    