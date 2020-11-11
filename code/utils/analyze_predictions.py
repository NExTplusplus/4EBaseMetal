import pandas as pd
import numpy as np
from copy import copy
import argparse
import os
import json
import sys
sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],'..')))
from utils.general_functions import read_data_with_specified_columns
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error,f1_score

if __name__ == '__main__':
    desc = 'the script for analyzing predictions'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-sou','--source', help = "source of data",type = str, default = "NExT")
    parser.add_argument('-s','--horizon_list',type=str,default="1,3,5,10,20,60",
                        help='list of horizons to be calculated, separated by ","')
    parser.add_argument('-gt', '--ground_truth_list', help='list of ground truths, separated by ","',
                        type=str, default="LME_Al_Spot,LME_Cu_Spot,LME_Pb_Spot,LME_Ni_Spot,LME_Xi_Spot,LME_Zn_Spot")
    parser.add_argument(
        '-m','--model', help='type of model (logistic,xgboost,alstm, etc)', type = str, default = "logistic"
    )
    parser.add_argument(
        '-v','--version_list', help='list of feature versions, separated by ","', type = str, default = 'v10'
    )
    parser.add_argument(
        '-mc', '--monte_carlo',help = 'string to identify if monte carlo is triggered',type = str, default = "False"
    )
    parser.add_argument ('-out','--output',type = str, help='filepath to store analysis results', default ="../../../Results/results")
    parser.add_argument('-d','--dates',type = str, help = "dates", default = "2014-12-31,2015-06-30,2015-12-31,2016-06-30,2016-12-31,2017-06-30,2017-12-31,2018-06-30,2018-12-31")
    parser.add_argument('-r','--regression',type = str, help = 'whether prediction is for regression', default = "off")

    args = parser.parse_args()
    args.horizon_list = args.horizon_list.split(",")
    args.ground_truth_list = args.ground_truth_list.split(",")
    args.dates = args.dates.split(",")
    args.monte_carlo = args.monte_carlo == "True"
    args.version_list = args.version_list.split(",")
    ans = {"version":[],"horizon":[],"ground_truth":[]}
    
    validation_dates = [d.split("-")[0]+"-01-01" if d[5:7] <= "06" else d.split("-")[0]+"-07-01" for d in args.dates]

    #for classification
    if args.regression  == "off":

        #initialize dictionary
        for d in validation_dates:
            ans[d+"_acc"] = []
            ans[d+"_pos_f1_score"] = []
            ans[d+"_neg_f1_score"] = []
            ans[d+"_f1_score"] = []
            ans[d+"_length"] = []

#             ans[d+"_tp"] = []
#             ans[d+"_fn"] = []
#             ans[d+"_tn"] = []
#             ans[d+"_fp"] = []

#             ans[d+"_ratio"] = []

        #iterate over feature version
        for version in args.version_list:
            v = version.split('_')[0]

            #iterate over horizon
            for h in args.horizon_list:
                for gt in args.ground_truth_list:
                    ans["version"].append(version)
                    ans["horizon"].append(h)
                    ans["ground_truth"].append(gt)

                    #iterate over dates
                    for i,date in enumerate(args.dates):
                        print(version,h,gt,date)

                        #generate filepath 
                        filepath = os.path.join("result","prediction",args.model)
                        f = "_".join([gt,date,h,version])+".csv"

                        #tweak filepath for alstm and post process
                        if args.model in ['alstm',"post_process"]:
                            filepath = os.path.join("result","prediction",args.model,version)
                            f = "_".join([gt,date,h,v])+".csv"
                        print(filepath,f)

                        #if there is no prediction file, then we will enter 0
                        if args.model not in ['post_process']:
                            if not os.path.exists(os.path.join(filepath,f)):
                                ans[validation_dates[i]+"_acc"].append(0)
                                ans[validation_dates[i]+"_pos_f1_score"].append(0)
                                ans[validation_dates[i]+"_neg_f1_score"].append(0)
                                ans[validation_dates[i]+"_f1_score"].append(0)
                                ans[validation_dates[i]+"_length"].append(0)

#                                 ans[validation_dates[i]+"_tp"].append(0)
#                                 ans[validation_dates[i]+"_fn"].append(0)
#                                 ans[validation_dates[i]+"_tn"].append(0)
#                                 ans[validation_dates[i]+"_fp"].append(0)
#                                 ans[validation_dates[i]+"_ratio"].append(0)
                                continue
                        else:
                            if not os.path.exists(os.path.join(filepath,f)):
                                ans[validation_dates[i]+"_acc"].append(0)
                                ans[validation_dates[i]+"_pos_f1_score"].append(0)
                                ans[validation_dates[i]+"_neg_f1_score"].append(0)
                                ans[validation_dates[i]+"_f1_score"].append(0)
                                ans[validation_dates[i]+"_length"].append(0)

#                                 ans[validation_dates[i]+"_tp"].append(0)
#                                 ans[validation_dates[i]+"_fn"].append(0)
#                                 ans[validation_dates[i]+"_tn"].append(0)
#                                 ans[validation_dates[i]+"_fp"].append(0)
#                                 ans[validation_dates[i]+"_ratio"].append(0)
                                continue

                        
                        temp = pd.read_csv(os.path.join(filepath, f),index_col = 0)
                        
                        #read label
                        label = pd.read_csv(os.path.join("data","Label","_".join([gt,"h"+str(h),validation_dates[i],"label.csv"])),index_col = 0)
                        if label.index[-1] > date:
                            label = label.iloc[:-1,:]
                        
                        #generate the metrics
                        accuracy = accuracy_score(label[:len(temp)],temp)
                        f1 = f1_score(label[:len(temp)],temp)
                        inverse_label = 1*(label[:len(temp)] == 0)
                        inverse_temp = 1*(temp == 0)
                        inverse_f1 = f1_score(inverse_label,inverse_temp)

                        ans[validation_dates[i]+"_acc"].append(accuracy)
                        ans[validation_dates[i]+"_pos_f1_score"].append(f1)
                        ans[validation_dates[i]+"_neg_f1_score"].append(inverse_f1)
                        ans[validation_dates[i]+"_f1_score"].append((f1+inverse_f1)/2)
                        ans[validation_dates[i]+"_length"].append(len(temp))

#                         pos_label = label.loc[label["Label"] == 1].index
#                         neg_label = label.loc[label["Label"] == 0].index
#                         tp = sum(1*(temp.loc[pos_label] == 1).values)
#                         fn = len(pos_label) - tp
#                         tn = sum(1*(temp.loc[neg_label] == 0).values)
#                         fp = len(neg_label) - tn
#                         ans[validation_dates[i]+"_tp"].append(tp)
#                         ans[validation_dates[i]+"_fn"].append(fn)
#                         ans[validation_dates[i]+"_tn"].append(tn)
#                         ans[validation_dates[i]+"_fp"].append(fp)

#                         pos = sum(1*(temp == 1).values)[0]
#                         neg = len(temp) - pos
#                         ans[validation_dates[i]+"_ratio"].append(pos/neg)

                #generate average over metals
                ans["version"].append(version)
                ans["horizon"].append(h)
                ans["ground_truth"].append("average")
                for val_date in validation_dates:
                    ans[val_date+"_acc"].append(np.average(ans[val_date+"_acc"][-6:]))
                    ans[val_date+"_pos_f1_score"].append(np.average(ans[val_date+"_pos_f1_score"][-6:]))
                    ans[val_date+"_neg_f1_score"].append(np.average(ans[val_date+"_neg_f1_score"][-6:]))
                    ans[val_date+"_f1_score"].append(np.average(ans[val_date+"_f1_score"][-6:]))
                    ans[val_date+"_length"].append(np.average(ans[val_date+"_length"][-6:]))

        #generate final averages
        ans = pd.DataFrame(ans)
        total_acc = 0.0
        total_pos_f1 = 0.0
        total_neg_f1 = 0.0
        total_f1 = 0.0
        total_length = 0
        for date in validation_dates:
            total_acc = total_acc + ans[date+"_acc"]*ans[date+"_length"]
            total_pos_f1 = total_pos_f1 + ans[date+"_pos_f1_score"]*ans[date+"_length"]
            total_neg_f1 = total_neg_f1 + ans[date+"_neg_f1_score"]*ans[date+"_length"]
            total_f1 = total_f1 + ans[date+"_f1_score"]*ans[date+"_length"]
            total_length = total_length+ans[date+"_length"]
        ans["final average"] = total_acc/total_length
        ans["final pos f1"] = total_pos_f1/total_length
        ans["final neg f1"] = total_neg_f1/total_length
        ans["final f1"] = total_f1/total_length
        
        ans.to_csv(args.output)

    #regression
    else:        
        #initialize dictionary
        for d in validation_dates:
            ans[d+"_mae"] = []
            ans[d+"_mse"] = []
            ans[d+"_length"] = []
            ans[d+"_acc"] = []
            #if it is post process, then we need to consider coverage
            if args.model in ['post_process']:
                ans[d+"_coverage"] = []
        
        #iterate over feature version
        for version in args.version_list:
            v = version.split('_')[0]

            #iterate over horizon
            for h in args.horizon_list:

                #iterate over ground truth
                for gt in args.ground_truth_list:
                    ans["version"].append(version)
                    ans["horizon"].append(h)
                    ans["ground_truth"].append(gt)

                    #iterate over date
                    for i,date in enumerate(args.dates):
                        print(version,h,gt,date)
                        #load labels
                        label = pd.read_csv(os.path.join("data","Label","_".join([gt,"h"+str(h),validation_dates[i],"reg_label.csv"])),index_col = 0)
                        class_label = pd.read_csv(os.path.join("data","Label","_".join([gt,"h"+str(h),validation_dates[i],"label.csv"])),index_col = 0)
                        filepath = os.path.join("result","prediction",args.model,version)
                        f = "_".join([gt,date,h,version])+".csv"

                        #generate filename
                        if args.model in ['alstm']:
                            if args.monte_carlo:
                                f = "_".join([gt,date,h,v,"True"])+".csv"
                            else:
                                f = "_".join([gt,date,h,v])+".csv"

                        #generate 0 if there is file
                        if args.model not in ['post_process']:
                            if not os.path.exists(os.path.join(filepath,f)):
                                ans[validation_dates[i]+"_mae"].append(0)
                                ans[validation_dates[i]+"_mse"].append(0)
                                ans[validation_dates[i]+"_length"].append(0)
                                ans[validation_dates[i]+"_acc"].append(0)
                                continue
                        else:
                            if not os.path.exists(os.path.join(filepath,f)):
                                ans[validation_dates[i]+"_mae"].append(0)
                                ans[validation_dates[i]+"_mse"].append(0)
                                ans[validation_dates[i]+"_acc"].append(0)
                                ans[validation_dates[i]+"_coverage"].append(0)
                                ans[validation_dates[i]+"_length"].append(len(label.index))
                                continue

                        #generate labels
                        temp = pd.read_csv(os.path.join(filepath, f),index_col = 0)
                        if label.index[-1] > date:
                            label = label.iloc[:-1,:]
                        data, LME_dates, length = read_data_with_specified_columns(args.source,'exp/LMCADY_v3.conf','2003-11-12')
                        spot = data.loc[label.index[0]:label.index[-1],gt].to_frame()
                        if args.regression == "ret":
                            temp = (temp - np.array(spot.loc[temp.index,:])) / np.array(spot.loc[temp.index,:])
                            label = (label - np.array(spot)) / np.array(spot)

                        #generate metrics
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
                                print(class_label.loc[temp.index,:],temp.loc[temp.index,:],spot.loc[temp.index,:])
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

                #generate average for 6 metals
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

        #generate final average across dates
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