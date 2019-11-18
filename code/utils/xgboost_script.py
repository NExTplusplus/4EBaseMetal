import pandas as pd
import numpy as np
from copy import copy
from multiprocessing import Pool as pl
from itertools import product
import argparse
import os

if __name__ == '__main__':
    desc = 'the script for XGBoost'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s','--step_list',type=str,default="1",
                        help='list of horizons to be calculated, separated by ","')
    parser.add_argument('-gt', '--ground_truth_list', help='list of ground truths, separated by ","',
                        type=str, default="LME_Co_Spot")
    parser.add_argument(
        '-sou','--source', help='source of data to be inserted into commands', type = str, default = "4E"
    )
    parser.add_argument(
        '-l','--lag_list', type=str, default = 5, help='list of lags, separated by ","'
    )
    parser.add_argument(
        '-v','--version_list', help='list of versions, separated by ","', type = str, default = 'v10'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
    parser.add_argument('-o', '--action', type=str, default='commands',
                        help='tuning,commands, testing')

    args = parser.parse_args()
    args.step_list = args.step_list.split(",")
    args.ground_truth_list = args.ground_truth_list.split(",")
    args.lag_list = args.lag_list.split(",")
    args.version_list = args.version_list.split(",")

    #generates the top 5 results for each voting method for every combination produced from version list, step list, ground truth list and lag list
    if args.action == "tuning":
        path_list = []
        combinations = product(args.ground_truth_list,args.lag_list,args.step_list,args.version_list)
        for c in combinations:
            path_list.append("_".join(c[0].split("_")[1],"xgboost","l"+c[1],"h"+c[2],c[3])+".txt")
        pool = pl()
        results = pool.starmap_async(retrieve_top,path_list)
        pool.close()
        pool.join()

    #generates the command line to be used for online testing
    if args.action == "commands":
        i = 0
        with open(args.output,"w") as out:
            for version in args.version_list:
                ground_truth_list = copy(args.ground_truth_list)
                xgb = 0
                if version in ["v10","v12"]:
                    ground_truth_list = ["all"]
                    exp = "exp/online_v10.conf"
                elif version in ["v5","v7"]:
                    exp = "exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf"
                elif version in ["v3"]:
                    exp = "exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf"
                elif version in ["v9"]:
                    exp = "exp/online_v10.conf"

                for gt in ground_truth_list:
                    for h in args.step_list:
                        total = pd.DataFrame()
                        for lag in args.lag_list:
                            f = pd.read_csv('_'.join([gt.split("_")[1],version,lag,"h"+h])+".csv").iloc[:5,:]
                            total = pd.concat([total,f],axis = 0)
                        total = total.sort_values(by=['result','lag','max_depth','learning_rate','gamma','min_child_weigh','subsample'],ascending=[False,True,True,True,True,True,True]).reset_index(drop = True)
                        out.write(" ".join(["python xgboost.py",
                          "-gt",gt,
                          "-l",str(total.iloc[0,0]),
                          "-s",h,
                          "-xgb",str(xgb),
                          "-v",version.split("_")[0],
                          "-c",exp,
                          "-sou",args.source,
                          "-max_depth",str(int(total.iloc[0,1])),
                          "-learning_rate",str(total.iloc[0,2]),
                          "-gamma",str(total.iloc[0,3]),
                          "-min_child",str(int(total.iloc[0,4])),
                          "-subsample",str(total.iloc[0,5]),"-voting all",
                          ">","_".join([gt,"xgboost","h"+h,version,"1718.txt"]),"2>&1 &"])+"\n")
        i+=1
        if i == 9:
          i = 0
          out.write("sleep 10m\n")

    if args.action == "testing":
        total = pd.DataFrame()
        for version in args.version_list:
            ans = None
            temp_ans = {}
            all_file = []
            ground_truth_columns = copy(args.ground_truth_list)
            for h in args.step_list:
                for gt in ground_truth_columns:
                    path = gt+"_xgboost_h"+h+"_"+version+"_1718.txt"
                    sub_file = []
                    all_voting_Str = 'the all folder voting precision is'
                    lag_Str = 'the lag is'
                    if path in os.listdir(os.path.join(os.curdir,"xgboost_res")):
                        with open(os.path.join(os.curdir,"xgboost_res",path),"r") as f:
                            lines = f.readlines()
                            for i,line in enumerate(lines):
                                if all_voting_Str.lower() in line.lower():
                                    file = []
                                    file.append(float(line.strip("\n").split(" ")[-1]))
                                    sub_file.append(file)
                                    if lag_Str.lower() in lines[i+1].lower():
                                        for result in sub_file:
                                            result.append(lines[i+1].strip("\n").split(" ")[-1])
                                            result.append(lines[i+3].strip("\n").split(" ")[-1])
                                        all_file+=sub_file
                                        sub_file = []
                    else:
                        continue
            for arr in all_file:
                if arr[2] not in temp_ans.keys():
                    temp_ans[arr[2]] = [arr[0]]
                else:
                    temp_ans[arr[2]].append(arr[0])
            temp_ans = pd.DataFrame(temp_ans)
            if ans is None:
                ans = temp_ans
            else:
                ans = pd.concat([ans,temp_ans], axis = 0, sort = False)
            ans.to_csv("xgboost_"+version+"_res.csv")

'''
    This function is used to extract the top 5 combinations of hyperparameters in terms of weighted accuracy for each voting method
'''
def retrieve_top(path):
    all_file = []
    sub_file = []
    all_voting_Str = 'the all folder voting precision is'
    lag_Str = 'the lag is'
    #read the file
    with open(path,"r") as f:
        lines = f.readlines()
        for i,line in enumerate(lines):

            # identify the line in which the relevant information is conveyed in the txt
            if all_voting_Str.lower() in line.lower():
                file = []
                file.append(float(line.strip("\n").split(" ")[-1]))

                #extract information regarding model hyperparameters and accuracy
                for new_line in lines[i+1:i+10]:
                    file.append(float(new_line.strip("\n").split(" ")[-1]))
                sub_file.append(file)

                #extract information regarding time period
                if lag_Str.lower() in lines[i+10].lower():
                    for result in sub_file:
                        result.append(lines[i+10].strip("\n").split(" ")[-1])
                        result.append(lines[i+11].strip("\n").split(" ")[-1])
                        result.append(lines[i+12].strip("\n").split(" ")[-1])
                    all_file+=sub_file
                    sub_file = []

    #generate data frame to congregrate data
    file_dataframe = pd.DataFrame(all_file,columns=['all_voting',
    'near_voting','far_voting','same_voting','reverse_voting',
    'max_depth','learning_rate','gamma','min_child_weight','subsample','lag','train_date','test_date'
    ])
    lag_list = list(file_dataframe['lag'].unique())
    max_depth_list = list(file_dataframe['max_depth'].unique())
    learning_rate_list = list(file_dataframe['learning_rate'].unique())
    gamma_list = list(file_dataframe['gamma'].unique())
    min_child_weight_list = list(file_dataframe['min_child_weight'].unique())
    subsample_list = list(file_dataframe['subsample'].unique())
    all_mean=[]

    #Calculate weighted mean for each combination of model hyperparameters across different time periods
    for lag in lag_list:
        for max_depth in max_depth_list:
            for learning_rate in learning_rate_list:
                for gamma in gamma_list:
                    for min_child_weight in min_child_weight_list:
                        for subsample in subsample_list:
                            mean_list = []
                            mean_list.append(lag)
                            mean_list.append(max_depth)
                            mean_list.append(learning_rate)
                            mean_list.append(gamma)
                            mean_list.append(min_child_weight)
                            mean_list.append(subsample)
                            df = file_dataframe[(file_dataframe['max_depth']==max_depth)&(file_dataframe['learning_rate']==learning_rate)
                                        &(file_dataframe['gamma']==gamma)&(file_dataframe['min_child_weight']==min_child_weight)&(file_dataframe['subsample']==subsample)&(file_dataframe['lag']==lag)].sort_values(by = "test_date")
                            all_mean_result = 0
                            near_mean_result = 0
                            far_mean_result = 0
                            reverse_mean_result = 0
                            same_mean_result = 0
                            for i in range(len(df)):

                                # generate weights
                                if df.iloc[i,-1] == "2014-07-01":
                                    length = 129
                                elif df.iloc[i,-1] == "2015-01-01":
                                    length = 124
                                elif df.iloc[i,-1] == "2015-07-01":
                                    length = 129
                                elif df.iloc[i,-1] == "2016-01-01":
                                    length = 125
                                elif df.iloc[i,-1] == "2016-07-01":
                                    length = 128
                                
                                #calculate amount of correct predictions
                                all_mean_result += df.iloc[i,0]*length
                                near_mean_result += df.iloc[i,1]*length
                                far_mean_result += df.iloc[i,2]*length
                                same_mean_result += df.iloc[i,3]*length
                                reverse_mean_result += df.iloc[i,4]*length

                            #Calculate accuracy
                            all_mean_result = all_mean_result/(129+124+129+125+128)
                            near_mean_result = near_mean_result/(129+124+129+125+128)
                            far_mean_result = far_mean_result/(129+124+129+125+128)
                            same_mean_result = same_mean_result/(129+124+129+125+128)
                            reverse_mean_result = reverse_mean_result/(129+124+129+125+128)

                            #append them to congregrated list
                            mean_list.append(all_mean_result)
                            mean_list.append(near_mean_result)
                            mean_list.append(far_mean_result)
                            mean_list.append(reverse_mean_result)
                            mean_list.append(same_mean_result)
                            all_mean.append(mean_list)
        new_frame = pd.DataFrame(all_mean,columns = ['lag','max_depth','learning_rate','gamma','min_child_weigh','subsample','all_mean_result'
        ,'near_mean_result','far_mean_result','reverse_mean_result','same_mean_result'
        ])

        all_frame = new_frame.sort_values(by=['all_mean_result','lag','max_depth','learning_rate','gamma','min_child_weigh','subsample'],ascending=[False,True,True,True,True,True,True])[:5].rename(columns = {'all_mean_result':'result'}).loc[:,['lag','max_depth','learning_rate','gamma','min_child_weigh','subsample','result']].sort_values(by="result",ascending = False)
        near_frame = new_frame.sort_values(by=['near_mean_result','lag','max_depth','learning_rate','gamma','min_child_weigh','subsample'],ascending=[False,True,True,True,True,True,True])[:5].rename(columns = {'near_mean_result':'result'}).loc[:,['lag','max_depth','learning_rate','gamma','min_child_weigh','subsample','result']].sort_values(by="result",ascending = False)
        far_frame = new_frame.sort_values(by=['far_mean_result','lag','max_depth','learning_rate','gamma','min_child_weigh','subsample'],ascending=[False,True,True,True,True,True,True])[:5].rename(columns = {'far_mean_result':'result'}).loc[:,['lag','max_depth','learning_rate','gamma','min_child_weigh','subsample','result']].sort_values(by="result",ascending = False)
        reverse_frame = new_frame.sort_values(by=['reverse_mean_result','lag','max_depth','learning_rate','gamma','min_child_weigh','subsample'],ascending=[False,True,True,True,True,True,True])[:5].rename(columns = {'reverse_mean_result':'result'}).loc[:,['lag','max_depth','learning_rate','gamma','min_child_weigh','subsample','result']].sort_values(by="result",ascending = False)
        same_frame = new_frame.sort_values(by=['same_mean_result','lag','max_depth','learning_rate','gamma','min_child_weigh','subsample'],ascending=[False,True,True,True,True,True,True])[:5].rename(columns = {'same_mean_result':'result'}).loc[:,['lag','max_depth','learning_rate','gamma','min_child_weigh','subsample','result']].sort_values(by="result",ascending = False)

        frame = pd.concat([all_frame,near_frame,far_frame,reverse_frame,same_frame], axis = 0)
        frame.to_csv(path.split("_")[0]+"_"+path.split("_")[-1].strip(".txt")+"_"+str(lag)+"_"+path.split("_")[3]+".csv",index = False)

