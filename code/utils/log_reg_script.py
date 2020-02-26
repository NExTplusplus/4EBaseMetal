import pandas as pd
import numpy as np
from copy import copy
import argparse
import os

if __name__ == '__main__':
    desc = 'the script for Logistic Regression'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s','--step_list',type=str,default="1,3,5,10,20,60",
                        help='list of horizons to be calculated, separated by ","')
    parser.add_argument('-gt', '--ground_truth_list', help='list of ground truths, separated by ","',
                        type=str, default="LME_Co_Spot,LME_Al_Spot,LME_Ni_Spot,LME_Ti_Spot,LME_Zi_Spot,LME_Le_Spot")
    parser.add_argument(
        '-sou','--source', help='source of data to be inserted into commands', type = str, default = "4E"
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
    args.lag_list = args.lag_list.split(",")
    args.version_list = args.version_list.split(",")
    args.dates = args.dates.split(",")
    args.length = [int(i) for i in args.length.split(",")]

    if args.action == "train commands":
        i = 0
        validation_dates = [d.split("-")[0]+"-01-01" if d[4:] == "-06-30" else d.split("-")[0]+"-07-01" for d in args.dates]
        with open(args.output,"w") as out:
            for version in args.version_list:
                ground_truth_list = copy(args.ground_truth_list)
                xgb = 0
                if version in ["v10","v12","v16","v26"]:
                    ground_truth_list = ["all"]
                    exp = "exp/online_v10.conf"
                elif version in ["v5","v7"]:
                    exp = "exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf"
                elif version in ["v3","v23","v37"]:
                    exp = "exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf"
                elif version in ["v9"]:
                    exp = "exp/online_v10.conf"
                elif version in ["v24","v28","v30"]:
                    ground_truth_list = ["all"]
                    exp = "exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf"
                elif version in ["v31"]:
                    exp = "exp/supply and demand.conf"
                elif version in ['v33','v35']:
                    exp = "exp/TP_v1.conf"
                train = "code/train_data_lr.py"
                for gt in ground_truth_list:
                    for h in args.step_list:
                        total = pd.DataFrame()
                        for lag in args.lag_list:
                            if '_'.join(['log_reg',gt,version,lag,h])+".csv" not in os.listdir(args.path):
                                continue
                            temp = pd.read_csv(os.path.join(args.path,'_'.join(["log_reg",gt,version,lag,h])+".csv"))
                            f = pd.concat([pd.DataFrame({"lag":[lag]*len(temp)}),temp], axis = 1)
                            total = pd.concat([total,f],axis = 0)
                        if total.empty:
                            continue
                        total.reset_index(inplace =True,drop = True)
                        temp_arr = {"average":[0]*len(f)*len(args.lag_list), "length":[0]*len(f)*len(args.lag_list)}
                        for col in total.columns.values.tolist():
                            if "_length" in col:
                                split_date = col[:-7]
                                if split_date in validation_dates:
                                    length = args.length[validation_dates.index(split_date)]
                                curr_ave = [i*length for i in list(total[split_date+"_acc"])]
                                temp_arr['average'] = [sum(x) for x in zip(temp_arr['average'],list(curr_ave))]
                                temp_arr['length'] = [sum(x) for x in zip(temp_arr['length'],list([length]*len(total[col])))]

                        temp = pd.DataFrame({"true_average":np.true_divide(temp_arr['average'],temp_arr['length'])})
                    
                        ans = pd.concat([total,temp],axis = 1).sort_values(by = ["true_average","lag","C"], ascending = [False,True,True])
                        
                        for d in args.dates:
                            out.write("python "+train+" "+" ".join(["-sou",args.source,"-v",version,"-c",exp,"-s",h,"-l",str(ans.iloc[0,0]),"-C",str(ans.iloc[0,2]),"-gt",gt,"-o","train",'-d',d,">","/dev/null", "2>&1", "&"]))
                            out.write("\n")
                            i+=1
                            if i%9 == 0 and args.source == "4E":
                                out.write("sleep 7m\n")
                            elif args.source == "NExT" and i %20 == 0:
                                out.write("sleep 3m\n")

    elif args.action == "test commands":
        i = 0
        validation_dates = [d.split("-")[0]+"-01-01" if d[4:] == "-06-30" else d.split("-")[0]+"-07-01" for d in args.dates]
        with open(args.output,"w") as out:
            for version in args.version_list:
                ground_truth_list = copy(args.ground_truth_list)
                xgb = 0
                if version in ["v10","v12","v16","v26"]:
                    exp = "exp/online_v10.conf"
                elif version in ["v5","v7"]:
                    exp = "exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf"
                elif version in ["v3","v23"]:
                    exp = "exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf"
                elif version in ["v9"]:
                    exp = "exp/online_v10.conf"
                elif version in ["v24","v28","v30"]:
                    exp = "exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf"
                train = "code/train_data_lr.py"
                for gt in ground_truth_list:
                    for h in args.step_list:
                        for j,d in enumerate(args.dates):
                            for lag in args.lag_list:
                                if "_".join([version,gt,h,lag,"lr",validation_dates[j]+".pkl"]) in os.listdir(os.path.join(os.getcwd(),"result","model","lr")) or (int(version[1:]) % 2 == 0 and\
                                    "_".join([version,"all",h,lag,"lr",validation_dates[j]+".pkl"]) in os.listdir(os.path.join(os.getcwd(),"result","model","lr"))):
                                    out.write("python "+train+" "+" ".join(["-sou",args.source,"-v",version,"-c",exp,"-s",h,"-l",lag,"-gt",gt,"-o","test",'-d',d,">","/dev/null", "2>&1", "&"]))
                                    out.write("\n")
                                    i+=1
                                    if i%9 == 0 and args.source == "4E":
                                        out.write("sleep 7m\n")
                                    elif args.source == "NExT" and i %20 == 0:
                                        out.write("sleep 3m\n")

    elif args.action == "testing":
        total = pd.DataFrame()
        for version in args.version_list:
            ground_truth_columns = copy(args.ground_truth_list)
            for h in args.step_list:
                for gt in ground_truth_columns:
                    f = pd.read_csv(os.path.join(args.path,'_'.join(["log_reg_online",version,gt,h])+".csv"))
                    total = pd.concat([total,f],axis = 0)
                total.reset_index(inplace =True,drop = True)
        total.to_csv(args.output)

# with open("log reg res.csv","w") as out:
#   for version in ["v5","v7","v9","v10","v12"]:
#     out.write(version+",,1,,,,,,,3,,,,,,,5\n")
#     for ground_truth in [("Co","LME_Co_Spot"),("Al","LME_Al_Spot"),("Ni","LME_Ni_Spot"),("Ti","LME_Ti_Spot"),("Le","LME_Le_Spot"),("Zi","LME_Zi_Spot")]:
#       if version == "v10" or version == "v12":
#         ground_truth = ("all","all")
#       out.write(ground_truth[0]+",")
    
#       for lag in [1,5,10,20,30]:
#         if lag == 1:
#           out.write(str(1)+",")
#         else:
#           out.write(","+str(lag)+",")
#         for h in [1,3,5]:
#           if '_'.join(["log_reg",ground_truth[1],version,str(lag),str(h)])+".csv" not in os.listdir():
#             print('_'.join(["log_reg",ground_truth[1],version,str(lag),str(h)])+".csv")
#             continue
#           f = pd.read_csv('_'.join(["log_reg",ground_truth[1],version,str(lag),str(h)])+".csv")
#           temp = {"average":[0]*7, "length":[0]*7}
#           for col in f.columns.values.tolist():
#             if "_length" in col and (sum(f[col]) < 7000):
#               split_date = col[:-7]
#               temp['average'] = [sum(x) for x in zip(temp['average'],list(f[split_date+"_acc"]*f[col]))]
#               temp['length'] = [sum(x) for x in zip(temp['length'],list(f[col]))]
#           acc = np.true_divide(temp['average'],temp['length'])
#           # print(acc)
#           # acc = list(f.iloc[:,-1])
#           for ac in acc:
#             out.write(str(ac)+",")
#         out.write("\n")

