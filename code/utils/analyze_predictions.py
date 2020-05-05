import pandas as pd
import numpy as np
from copy import copy
import argparse
import os
import json
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    desc = 'the script for Logistic Regression'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-c','--config',type = str,default = 'exp/classification_post_process.conf',
                        help = 'configuration file for post process'
)
    parser.add_argument('-s','--step_list',type=str,default="1,3,5",
                        help='list of horizons to be calculated, separated by ","')
    parser.add_argument('-gt', '--ground_truth_list', help='list of ground truths, separated by ","',
                        type=str, default="LME_Co_Spot,LME_Al_Spot,LME_Ni_Spot,LME_Ti_Spot,LME_Zi_Spot,LME_Le_Spot")
    parser.add_argument(
        '-m','--model', help='model used', type = str, default = "lr"
    )
    parser.add_argument(
        '-l','--lag_list', type=str, default = "1,5,10,20,30", help='list of lags, separated by ","'
    )
    parser.add_argument(
        '-v','--version_list', help='list of versions, separated by ","', type = str, default = 'v10'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
    parser.add_argument('-o', '--action', type=str, default='commands',
                        help='commands, testing')
    parser.add_argument('-d','--dates',type = str, help = "dates", default = "2014-12-31,2015-06-30,2015-12-31,2016-06-30,2016-12-31,2017-06-30,2017-12-31,2018-06-30,2018-12-31")
    parser.add_argument('-length','--length',type = str, help = "length of each period stated in dates",default = "129,124,129,125,128,125,127,125,128")
    parser.add_argument('-p','--path',type =str, help='path to 4EBaseMetal folder',default ='/NEXT/4EBaseMetal')

    args = parser.parse_args()
    args.step_list = args.step_list.split(",")
    args.ground_truth_list = args.ground_truth_list.split(",")
    args.dates = args.dates.split(",")
    args.length = [int(i) for i in args.length.split(",")]
    with open(args.config) as f:
        config = json.load(f)

    args.version_list = args.version_list.split(",")
    ans = {"version":[],"horizon":[],"ground_truth":[]}
    validation_dates = [d.split("-")[0]+"-01-01" if d[5:7] <= "06" else d.split("-")[0]+"-07-01" for d in args.dates]
    for d in validation_dates:
        ans[d+"_acc"] = []
        ans[d+"_length"] = []
    for version in args.version_list:
        v = version.split('_')[0]
        dc = None
        for key in config.keys():
            if version.split('_')[1] in config[key].keys():
                dc = config[key][version.split('_')[1]][version.split('_')[0]] 
        for h in args.step_list:
            for gt in args.ground_truth_list:
                ans["version"].append(version)
                ans["horizon"].append(h)
                ans["ground_truth"].append(gt)
                for i,date in enumerate(args.dates):
                    print(version,h,gt,date)
                    filepath = os.path.join("result","prediction",args.model)
                    f = "_".join([gt,date,h,version])+".csv"
                    if args.model in ['alstm']:
                        filepath = os.path.join("result","prediction",args.model,version)
                        f = "_".join([gt,date,h,v])+".csv"
                    if args.model not in ['post_process']:
                        if f not in os.listdir(filepath):
                            ans[validation_dates[i]+"_acc"].append(0)
                            ans[validation_dates[i]+"_length"].append(0)
                            continue
                    else:
                        if int(h) not in dc[gt]:
                            f = "_".join([gt,date,h,v])+".csv"
                        elif f not in os.listdir(filepath):
                            ans[validation_dates[i]+"_acc"].append(0)
                            ans[validation_dates[i]+"_length"].append(0)
                            continue
                    temp = pd.read_csv(os.path.join(filepath, f),index_col = 0)
                    label = pd.read_csv(os.path.join("data","Label","_".join([gt,"h"+str(h),validation_dates[i],"label.csv"])),index_col = 0)
                    if label.index[-1] > date:
                        label = label.iloc[:-1,:]
                    accuracy = accuracy_score(label[:len(temp)],temp)
                    ans[validation_dates[i]+"_acc"].append(accuracy)
                    ans[validation_dates[i]+"_length"].append(len(temp))
            ans["version"].append(version)
            ans["horizon"].append(h)
            ans["ground_truth"].append("average")
            for val_date in validation_dates:
                ans[val_date+"_acc"].append(np.average(ans[val_date+"_acc"][-6:]))
                ans[val_date+"_length"].append(np.average(ans[val_date+"_length"][-6:]))
    ans = pd.DataFrame(ans)
    total_acc = 0.0
    total_length = 0
    for date in validation_dates:
        total_acc = total_acc + ans[date+"_acc"]*ans[date+"_length"]
        total_length = total_length+ans[date+"_length"]
    ans["final average"] = total_acc/total_length
    ans.to_csv(args.output)
    