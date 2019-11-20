import pandas as pd
import numpy as np
from copy import copy
import argparse
import os

if __name__ == '__main__':
    desc = 'the script for Logistic Regression'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s','--step_list',type=str,default="1,3,5",
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
                        
    parser.add_argument('-p','--path',type =str, help='path to 4EBaseMetal folder',default ='/NEXT/4EBaseMetal')

    args = parser.parse_args()
    args.step_list = args.step_list.split(",")
    args.ground_truth_list = args.ground_truth_list.split(",")
    args.lag_list = args.lag_list.split(",")
    args.version_list = args.version_list.split(",")


    if args.action == "commands":
        i = 0
        with open(args.output,"w") as out:
            for version in args.version_list:
                ground_truth_list = copy(args.ground_truth_list)
                xgb = 0
                if version in ["v10","v12"]:
                    ground_truth_list = ["all"]
                    exp = "exp/online_v10.conf"
                    train = "code/train/train_log_reg_v10_online.py"
                elif version in ["v5","v7"]:
                    exp = "exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf"
                    train = "code/train/train_log_reg_online.py"
                elif version in ["v3"]:
                    exp = "exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf"
                    train = "code/train/train_log_reg_online.py"
                elif version in ["v9"]:
                    exp = "exp/online_v10.conf"
                    train = "code/train/train_log_reg_online.py"
                for gt in ground_truth_list:
                    for h in args.step_list:
                        total = pd.DataFrame()
                        for lag in args.lag_list:
                            temp = pd.read_csv(os.path.join(args.path,'_'.join(["log_reg",gt,version,lag,h])+".csv"))
                            f = pd.concat([pd.DataFrame({"lag":[lag]*len(temp)}),temp], axis = 1)
                            total = pd.concat([total,f],axis = 0)
                        total.reset_index(inplace =True,drop = True)
                        temp_arr = {"average":[0]*len(f)*len(args.lag_list), "length":[0]*len(f)*len(args.lag_list)}
                        for col in total.columns.values.tolist():
                            if "_length" in col and (sum(total[col]) < 1000*len(f)*len(args.lag_list)):
                                split_date = col[:-7]
                                if split_date == "2014-07-01":
                                    length = 129
                                elif split_date == "2015-01-01":
                                    length = 124
                                elif split_date == "2015-07-01":
                                    length = 129
                                elif split_date == "2016-01-01":
                                    length = 125
                                elif split_date == "2016-07-01":
                                    length = 128
                                temp_arr['average'] = [sum(x) for x in zip(temp_arr['average'],list(total[split_date+"_acc"]*length))]
                                temp_arr['length'] = [sum(x) for x in zip(temp_arr['length'],list([length]*len(total[col])))]
                        temp = pd.DataFrame({"true_average":np.true_divide(temp_arr['average'],temp_arr['length'])})
                        ans = pd.concat([total,temp],axis = 1).sort_values(by = ["true_average","lag","C"], ascending = [False,True,True])

                        out.write("python "+train+" ".join(["-sou",args.source,"-v",version,"-c",exp,"-s",h,"-l",str(ans.iloc[0,0]),"-C",str(ans.iloc[0,2]),"-gt",gt,"-xgb",str(xgb),">",gt+"_"+h+"_"+version+"_1718.txt", "2>&1", "&"]))
                        out.write("\n")
                        i+=1
                        if i%9 == 0:
                            out.write("sleep 10m\n")

    if args.action == "testing":
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

