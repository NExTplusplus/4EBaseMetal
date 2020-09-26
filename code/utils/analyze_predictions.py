import pandas as pd
import numpy as np
from copy import copy
import argparse
import os
import json
import sys
sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],'..')))
from utils.general_functions import read_data_with_specified_columns
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error

if __name__ == '__main__':
    desc = 'the script for analyze predictions'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-c','--config',type = str,default = 'exp/classification_post_process.conf',
                        help = 'configuration file for post process'
    )
    parser.add_argument('-sou','--source',type = str, default = "NExT")
    parser.add_argument('-s','--step_list',type=str,default="1,3,5,10,20,60",
                        help='list of horizons to be calculated, separated by ","')
    parser.add_argument('-gt', '--ground_truth_list', help='list of ground truths, separated by ","',
                        type=str, default="LME_Al_Spot,LME_Co_Spot,LME_Le_Spot,LME_Ni_Spot,LME_Ti_Spot,LME_Zi_Spot")
    parser.add_argument(
        '-m','--model', help='model used', type = str, default = "lr"
    )
    parser.add_argument(
        '-v','--version_list', help='list of versions, separated by ","', type = str, default = 'v10'
    )
    parser.add_argument(
        '-mc', '--monte_carlo',type = str, default = "False"
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
    parser.add_argument('-d','--dates',type = str, help = "dates", default = "2014-12-31,2015-06-30,2015-12-31,2016-06-30,2016-12-31,2017-06-30,2017-12-31,2018-06-30,2018-12-31")
    parser.add_argument('-r','--regression',type = str, help = 'whether prediction is for regression', default = "off")

    args = parser.parse_args()
    args.step_list = args.step_list.split(",")
    args.ground_truth_list = args.ground_truth_list.split(",")
    args.dates = args.dates.split(",")
    args.monte_carlo = args.monte_carlo == "True"
    with open(os.path.join(os.getcwd(),args.config)) as f:
        config = json.load(f)

    args.version_list = args.version_list.split(",")
    ans = {"version":[],"horizon":[],"ground_truth":[]}
    validation_dates = [d.split("-")[0]+"-01-01" if d[5:7] <= "06" else d.split("-")[0]+"-07-01" for d in args.dates]
    if args.regression  == "off":
        for d in validation_dates:
            ans[d+"_acc"] = []
            ans[d+"_length"] = []
        for version in args.version_list:
            v = version.split('_')[0]
            dc = None
            if args.model == 'post_process':
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
    else:        
        for d in validation_dates:
            ans[d+"_mae"] = []
            ans[d+"_mse"] = []
            ans[d+"_length"] = []
            ans[d+"_acc"] = []
            if args.model in ['post_process']:
                ans[d+"_coverage"] = []
        for version in args.version_list:
            v = version.split('_')[0]
            dc = None
            for h in args.step_list:
                for gt in args.ground_truth_list:
                    ans["version"].append(version)
                    ans["horizon"].append(h)
                    ans["ground_truth"].append(gt)
                    for i,date in enumerate(args.dates):
                        print(version,h,gt,date)
                        label = pd.read_csv(os.path.join("data","Label","_".join([gt,"h"+str(h),validation_dates[i],"reg_label.csv"])),index_col = 0)
                        class_label = pd.read_csv(os.path.join("data","Label","_".join([gt,"h"+str(h),validation_dates[i],"label.csv"])),index_col = 0)
                        filepath = os.path.join("result","prediction",args.model,version)
                        f = "_".join([gt,date,h,version])+".csv"
                        if args.model in ['alstm']:
                            if args.monte_carlo:
                                f = "_".join([gt,date,h,v,"True"])+".csv"
                            else:
                                f = "_".join([gt,date,h,v])+".csv"
                        if args.model not in ['post_process']:
                            if f not in os.listdir(filepath):
                                ans[validation_dates[i]+"_mae"].append(0)
                                ans[validation_dates[i]+"_mse"].append(0)
                                ans[validation_dates[i]+"_length"].append(0)
                                ans[validation_dates[i]+"_acc"].append(0)
                                continue
                        else:
                            if f not in os.listdir(filepath):
                                ans[validation_dates[i]+"_mae"].append(0)
                                ans[validation_dates[i]+"_mse"].append(0)
                                ans[validation_dates[i]+"_acc"].append(0)
                                ans[validation_dates[i]+"_coverage"].append(0)
                                ans[validation_dates[i]+"_length"].append(len(label.index))
                                continue
                        temp = pd.read_csv(os.path.join(filepath, f),index_col = 0)
                        if label.index[-1] > date:
                            label = label.iloc[:-1,:]
                        data, LME_dates, length = read_data_with_specified_columns(args.source,'exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf','2003-11-12')
                        spot = data.loc[label.index[0]:label.index[-1],gt].to_frame()
                        if args.regression == "ret":
                            temp = (temp - np.array(spot.loc[temp.index,:])) / np.array(spot.loc[temp.index,:])
                            label = (label - np.array(spot)) / np.array(spot)
                        if len(temp.index) == 0:
                            mae = 0
                            mse = 0
                            acc = 0
                        else:
                            mae = mean_absolute_error(label.loc[temp.index,:],temp)
                            mse = mean_squared_error(label.loc[temp.index,:],temp)
                            if args.regression == "ret":
                                acc = accuracy_score(class_label.loc[temp.index,:], (np.sign(temp)+1)/2)
                            else:
                                acc = accuracy_score(class_label.loc[temp.index,:], np.array(temp.loc[temp.index,:]) > np.array(spot.loc[temp.index,:]))
                        ans[validation_dates[i]+"_mae"].append(mae)
                        ans[validation_dates[i]+"_mse"].append(mse)
                        ans[validation_dates[i]+"_acc"].append(acc)
                        if args.model in ['post_process']:
                            ans[validation_dates[i]+"_coverage"].append(len(temp.index)/len(label.index))
                        ans[validation_dates[i]+"_length"].append(len(label.index))
                ans["version"].append(version)
                ans["horizon"].append(h)
                ans["ground_truth"].append("average")
                for val_date in validation_dates:
                    if args.model in ['post_process']:
                        # print(ans[val_date+"_acc"][-6:],ans[val_date+"_mae"][-6:],ans[val_date+"_coverage"][-6:],ans[val_date+"_length"][-6:])
                        coverage = np.array(ans[val_date+"_coverage"][-6:]*np.array(ans[val_date+"_length"][-6:]))
                        ans[val_date+"_acc"].append(sum(ans[val_date+"_acc"][-6:]*coverage)/sum(coverage))
                        ans[val_date+"_mae"].append(sum(ans[val_date+"_mae"][-6:]*coverage)/sum(coverage))
                        ans[val_date+"_mse"].append(sum(ans[val_date+"_mse"][-6:]*coverage)/sum(coverage))
                        ans[val_date+"_coverage"].append(np.average(coverage)/np.average(ans[val_date+"_length"][-6:]))
                    else:
                        ans[val_date+"_acc"].append(np.average(ans[val_date+"_acc"][-6:]))
                        ans[val_date+"_mae"].append(np.average(ans[val_date+"_mae"][-6:]))
                        ans[val_date+"_mse"].append(np.average(ans[val_date+"_mse"][-6:]))
                    ans[val_date+"_length"].append(np.average(ans[val_date+"_length"][-6:]))
        [print(temp, len(ans[temp])) for temp in ans.keys()]
        ans = pd.DataFrame(ans)
        total_mae = np.zeros(len(ans.index))
        total_mse = np.zeros(len(ans.index))
        total_acc = np.zeros(len(ans.index))
        total_coverage = np.zeros(len(ans.index))
        total_length = np.zeros(len(ans.index))
        if args.model in ['post_process']:
            for date in validation_dates:
                total_mae = total_mae + ans[date+"_mae"]*ans[date+"_coverage"]*ans[date+"_length"]
                total_mse = total_mse + ans[date+"_mse"]*ans[date+"_coverage"]*ans[date+"_length"]
                total_acc = total_acc + ans[date+"_acc"]*ans[date+"_coverage"]*ans[date+"_length"]
                total_coverage = total_coverage + ans[date+"_coverage"]*ans[date+"_length"]
                total_length = total_length+ans[date+"_length"]
            ans["mae"] = total_mae/total_coverage
            ans["mse"] = np.sqrt(total_mse/total_coverage)
            ans["acc"] = total_acc/total_coverage
            ans["coverage"] = total_coverage/total_length
        else:
            for date in validation_dates:
                total_acc = total_acc + ans[date+"_acc"]*ans[date+"_length"]
                total_mae = total_mae + ans[date+"_mae"]*ans[date+"_length"]
                total_mse = total_mse + ans[date+"_mse"]*ans[date+"_length"]
                total_length = total_length+ans[date+"_length"]
            ans["mae"] = total_mae/total_length
            ans["mse"] = np.sqrt(total_mse/total_length)
            ans['acc'] = total_acc/total_length

        
        ans.to_csv(args.output)