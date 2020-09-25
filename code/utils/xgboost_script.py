import pandas as pd
import numpy as np
from copy import copy
from multiprocessing import Pool as pl
from itertools import product
from pathlib import Path
import argparse
import os

'''
    This function is used to extract the top 5 combinations of hyperparameters in terms of weighted accuracy for each voting method
'''
def retrieve_top(path):
    validation_dates = path[1]
    path = path[0]
    all_file = []
    sub_file = []
    all_voting_Str = 'the all folder voting precision is'
    lag_Str = 'the lag is'
    print(path)
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
                        result.append(lines[i+13].strip("\n").split(" ")[-1])
                    all_file+=sub_file
                    sub_file = []

    path = Path(path)
    parts = list(path.parts)
    path = parts[-1]
    directory = os.path.join(*(parts[:-1]))
    
    #generate data frame to congregrate data
    file_dataframe = pd.DataFrame(all_file,columns=['all_voting',
    'near_voting','far_voting','same_voting','reverse_voting',
    'max_depth','learning_rate','gamma','min_child_weight','subsample','lag','train_date','test_date','length'
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

                            total_length = 0
                            for i in range(len(df)):
                                #calculate amount of correct predictions
                                df.iloc[i,-1] = int(df.iloc[i,-1])
                                all_mean_result += df.iloc[i,0]*df.iloc[i,-1]
                                near_mean_result += df.iloc[i,1]*df.iloc[i,-1]
                                far_mean_result += df.iloc[i,2]*df.iloc[i,-1]
                                same_mean_result += df.iloc[i,3]*df.iloc[i,-1]
                                reverse_mean_result += df.iloc[i,4]*df.iloc[i,-1]
                                total_length += df.iloc[i,-1]

                            #Calculate accuracy
                            all_mean_result = all_mean_result/total_length
                            near_mean_result = near_mean_result/total_length
                            far_mean_result = far_mean_result/total_length
                            same_mean_result = same_mean_result/total_length
                            reverse_mean_result = reverse_mean_result/total_length

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
        frame.to_csv(os.path.join(directory,path.split("_")[0]+"_"+path.split("_")[-1].strip(".txt")+"_"+str(lag)+"_"+path.split("_")[3]+".csv"),index = False)




if __name__ == '__main__':
    desc = 'the script for XGBoost'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-s','--step_list',type=str,default="1,3,5,10,20,60",
                        help='list of horizons to be calculated, separated by ","')
    parser.add_argument('-gt', '--ground_truth_list', help='list of ground truths, separated by ","',
                        type=str, default="LME_All_Spot,LME_Co_Spot,LME_Al_Spot,LME_Ni_Spot,LME_Ti_Spot,LME_Zi_Spot,LME_Le_Spot")

    parser.add_argument(
        '-sou','--source', help='source of data to be inserted into commands', type = str, default = "NExT"
    )

    parser.add_argument(
        '-l','--lag_list', type=str, default = '1,5,10', help='list of lags, separated by ","'
    )
    parser.add_argument(
        '-v','--version_list', help='list of versions, separated by ","', type = str, default = 'v10'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
    parser.add_argument('-o', '--action', type=str, default='commands',
                        help='tune,train,test')
    parser.add_argument('-d','--dates',type = str, help = "dates", default = "2014-12-31,2015-06-30,2015-12-31,2016-06-30,2016-12-31,2017-06-30,2017-12-31,2018-06-30,2018-12-31")
    parser.add_argument('-p','--path',type =str, help='path to 4EBaseMetal folder',default ='NEXT/4EBaseMetal')

    args = parser.parse_args()
    args.step_list = args.step_list.split(",")
    args.ground_truth_list = args.ground_truth_list.split(",")
    args.lag_list = args.lag_list.split(",")
    args.version_list = args.version_list.split(",")
    dates_list = args.dates.split(",")

    #generates the top 5 results for each voting method for every combination produced from version list, step list, ground truth list and lag list
    if args.action == "tune":
        path_list = []
        validation_dates = [d.split("-")[0]+"-01-01" if d[5:7] <= "06" else d.split("-")[0]+"-07-01" for d in dates_list]
        combinations = product(args.ground_truth_list,args.lag_list,args.step_list,args.version_list)
        for c in combinations:
            
            if "_".join([c[0].split("_")[1],"xgboost","l"+c[1],"h"+c[2],c[3]])+".txt" in os.listdir(os.path.join(os.getcwd(),"result","validation","xgboost")):
                path_list.append([os.path.join(args.path,"_".join([c[0].split("_")[1],"xgboost","l"+c[1],"h"+c[2],c[3]])+".txt"),validation_dates])
        print(path_list)

        pool = pl()
        results = pool.map_async(retrieve_top,path_list)
        pool.close()
        pool.join()

    #generates the command line to be used for online testing
    if args.action == "train":
        i = 0
        with open(args.output,"w") as out:
            for version in args.version_list:
                ground_truth_list = copy(args.ground_truth_list)
                xgb = 0
                train = "code/train_data_xgboost.py"
                

                for gt in ground_truth_list:
                    for h in args.step_list:
                        total = pd.DataFrame()
                        for lag in args.lag_list:
                            print(gt)
                            if '_'.join([gt.split("_")[1],version,lag,"h"+h])+".csv" in os.listdir(args.path):
                                f = pd.read_csv(os.path.join(args.path,'_'.join([gt.split("_")[1],version,lag,"h"+h])+".csv")).iloc[:5,:]
                                total = pd.concat([total,f],axis = 0)
                        if total.empty:
                            continue
                        total = total.sort_values(by=['result','lag','max_depth','learning_rate','gamma','min_child_weigh','subsample'],ascending=[False,True,True,True,True,True,True]).reset_index(drop = True)
                        out.write(" ".join(["python",train,
                        "-d",args.dates,
                        "-gt",gt,
                        "-l",str(total.iloc[0,0]),
                        "-s",h,
                        "-v",version.split("_")[0],
                        "-sou",args.source,
                        "-max_depth",str(int(total.iloc[0,1])),
                        "-learning_rate",str(total.iloc[0,2]),
                        "-gamma",str(total.iloc[0,3]),
                        "-min_child",str(int(total.iloc[0,4])),
                        "-subsample",str(total.iloc[0,5]),"-o train",
                        ">","/dev/null","2>&1 &"])+"\n")
                        i+=1
                        if i%6 == 0 and args.source == "4E":
                            out.write("sleep 10m\n")
                        elif args.source == "NExT" and i %20 == 0:
                            out.write("sleep 5m\n")

    #generates the command line to be used for online testing
    if args.action == "test":
        i = 0
        validation_dates = [d.split("-")[0]+"-01-01" if d[5:7] <= "06" else d.split("-")[0]+"-07-01" for d in dates_list]
        with open(args.output,"w") as out:
            for version in args.version_list:
                ground_truth_list = copy(args.ground_truth_list)
                xgb = 0
                train = "code/train_data_xgboost.py"
                for gt in ground_truth_list:
                    for h in args.step_list:
                        total = pd.DataFrame()

                        #analyze validation results to get best hyperparameter combination
                        for lag in args.lag_list:
                            if int(version[1:]) %2 == 0:
                                ground_truth = "LME_All"
                            else:
                                ground_truth = gt
                            if '_'.join([ground_truth.split("_")[1],version,lag,"h"+h])+".csv" in os.listdir(args.path):
                                print(gt,h,lag)
                                f = pd.read_csv(os.path.join(args.path,'_'.join([ground_truth.split("_")[1],version,lag,"h"+h])+".csv")).iloc[:5,:]
                                total = pd.concat([total,f],axis = 0)
                        if total.empty:
                            continue
                        total = total.sort_values(by=['result','lag','max_depth','learning_rate','gamma','min_child_weigh','subsample'],ascending=[False,True,True,True,True,True,True]).reset_index(drop = True)
                        
                        #write command
                        out.write(" ".join(["python",train,
                            "-d",args.dates,
                            "-gt",gt,
                            "-l",str(total.iloc[0,0]),
                            "-s",h,
                            "-v",version.split("_")[0],
                            "-sou",args.source,
                            "-max_depth",str(int(total.iloc[0,1])),
                            "-learning_rate",str(total.iloc[0,2]),
                            "-gamma",str(total.iloc[0,3]),
                            "-min_child",str(int(total.iloc[0,4])),
                            "-subsample",str(total.iloc[0,5]),"-o test",
                            "> /dev/null 2>&1 &"])+"\n")
                        i+=1
                        if i%6 == 0 and args.source == "4E":
                            out.write("sleep 10m\n")
                        elif args.source == "NExT" and i %6 == 0:
                            out.write("sleep 5m\n")

