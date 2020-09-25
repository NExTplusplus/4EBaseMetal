import pandas as pd
import numpy as np
from copy import copy
import argparse
import os

if __name__ == '__main__':
    desc = 'the script for Logistic Regression to generate train and test commands'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s','--step_list',type=str,default="1,3,5,10,20,60",
                        help='list of horizons to be calculated, separated by ","')
    parser.add_argument('-gt', '--ground_truth_list', help='list of ground truths, separated by ","',
                        type=str, default="LME_Co_Spot,LME_Al_Spot,LME_Ni_Spot,LME_Ti_Spot,LME_Zi_Spot,LME_Le_Spot")
    parser.add_argument(
        '-sou','--source', help='source of data to be inserted into commands', type = str, default = "4E"
    )
    parser.add_argument(
        '-l','--lag_list', type=str, default = "1,5,10,20", help='list of lags, separated by ","'
    )
    parser.add_argument(
        '-v','--version_list', help='list of versions, separated by ","', type = str, default = 'v10'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="test.sh")
    parser.add_argument('-o', '--action', type=str, default='commands',
                        help='commands, testing')
    parser.add_argument('-d','--dates',type = str, help = "dates", default = "2014-12-31,2015-06-30,2015-12-31,2016-06-30,2016-12-31,2017-06-30,2017-12-31,2018-06-30,2018-12-31")
    parser.add_argument('-p','--path',type =str, help='path to 4EBaseMetal folder',default ='/NEXT/4EBaseMetal')

    # initialize parameters
    args = parser.parse_args()
    args.step_list = args.step_list.split(",")
    args.ground_truth_list = args.ground_truth_list.split(",")
    args.lag_list = args.lag_list.split(",")
    args.version_list = args.version_list.split(",")
    dates_list = args.dates.split(",")

    i = 0
    validation_dates = [d.split("-")[0]+"-01-01" if d[5:7] <= "06" else d.split("-")[0]+"-07-01" for d in dates_list]

    #open output file to write
    with open(args.output,"w") as out:

        #iterate across version
        for version in args.version_list:
            ground_truth_list = copy(args.ground_truth_list)
            xgb = 0
            train = "code/train_data_logistic.py"

            #even version has different name formats 
            if version in ["v10","v12","v24","v26","v28","v30"]:
                ground_truth_list = ['all']

            for gt in args.ground_truth_list:
                if ground_truth_list[0] == "all":
                    fname = "LME_All_Spot"
                else:
                    fname = gt
                
                for h in args.step_list:
                    total = pd.DataFrame()

                    for lag in args.lag_list:
                        # if file not in folder, skip
                        if '_'.join(['log_reg',fname,version,lag,h])+".csv" not in os.listdir(args.path):
                            continue

                        #read file and add lag column
                        temp = pd.read_csv(os.path.join(args.path,'_'.join(["log_reg",fname,version,lag,h])+".csv"))
                        f = pd.concat([pd.DataFrame({"lag":[lag]*len(temp)}),temp], axis = 1)
                        total = pd.concat([total,f],axis = 0)
                    if total.empty:
                        continue

                    #identify best hyperparameter combination with weighted average
                    total.reset_index(inplace =True,drop = True)
                    temp_arr = {"average":[0]*len(f)*len(args.lag_list), "length":[0]*len(f)*len(args.lag_list)}
                    for col in total.columns.values.tolist():
                        if "_length" in col:
                            split_date = col[:-7]
                            curr_ave = list(total[split_date+"_acc"]*total[col])
                            temp_arr['average'] = [sum(x) for x in zip(temp_arr['average'],list(curr_ave))]
                            temp_arr['length'] = [sum(x) for x in zip(temp_arr['length'],list(total[col]))]

                    temp = pd.DataFrame({"true_average":np.true_divide(temp_arr['average'],temp_arr['length'])})
                
                    ans = pd.concat([total,temp],axis = 1).sort_values(by = ["true_average","lag","C"], ascending = [False,True,True])
                    
                    #write to output file
                    out.write("python "+train+" "+" ".join(["-sou",args.source,"-v",version,"-s",h,"-l",str(ans.iloc[0,0]),"-C",str(ans.iloc[0,2]),"-gt",gt,"-o",args.action,'-d',args.dates,">","/dev/null", "2>&1", "&"]))
                    out.write("\n")
                    i+=1
                    if i%6 == 0 and args.source == "4E":
                        out.write("sleep 10m\n")
                    elif args.source == "NExT" and i %6 == 0:
                        out.write("sleep 5m\n")
