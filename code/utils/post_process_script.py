import pandas as pd
import numpy as np
from copy import copy
import argparse
import os

if __name__ == '__main__':
    desc = 'the script for Post Process'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s','--horizon_list',type=str,default="1,3,5,10,20,60",
                        help='list of horizons to be calculated, separated by ","')
    parser.add_argument('-gt', '--ground_truth_list', help='list of ground truths, separated by ","',
                        type=str, default="LME_Co_Spot,LME_Al_Spot,LME_Ni_Spot,LME_Ti_Spot,LME_Zi_Spot,LME_Le_Spot")
    parser.add_argument(
        '-sou','--source', help='source of data to be inserted into commands', type = str, default = "4E"
    )
    parser.add_argument(
        '-m','--model', type=str, default = "Filter", help='list of models, separated by ","'
    )
    parser.add_argument(
        '-v','--version_list', help='list of versions, separated by ","', type = str, default = 'v10'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
    parser.add_argument('-o', '--action', type=str, default='commands',
                        help='commands, testing')
    parser.add_argument('-d','--dates',type = str, help = "dates", default = "2014-12-31,2015-06-30,2015-12-31,2016-06-30,2016-12-31,2017-06-30,2017-12-31,2018-06-30,2018-12-31")
    parser.add_argument('-p','--path',type =str, help='path to 4EBaseMetal folder',default ='/NEXT/4EBaseMetal')

    args = parser.parse_args()
    args.path = args.path.split(',')
    args.horizon_list = args.horizon_list.split(",")
    args.ground_truth_list = args.ground_truth_list.split(",")
    dates_list = args.dates.split(",")
    total_classification = pd.DataFrame()
    total_regression = pd.DataFrame()

    #simple analysis (single metal, single horizon)
    if args.action == "simple":
        args.version_list = args.version_list.split(',')

        #iterate over ground truth
        for ground_truth in args.ground_truth_list:

            #
            for horizon in args.horizon_list:
                print(ground_truth,horizon)
                temp_class = pd.DataFrame()
                temp_reg = pd.DataFrame()

                #iterate over path
                for p,path in enumerate(args.path):
                    classification = pd.read_csv(os.path.join(path,'_'.join([ground_truth,args.version_list[0],horizon,"classification.csv"])), index_col = 0)
                    regression = pd.read_csv(os.path.join(path,'_'.join([ground_truth,args.version_list[1],horizon,"regression.csv"])), index_col = 0)
                    regression['rank'] = (regression['mae_rank']+regression['coverage_rank'])/2
                    if p == 1:
                        classification = classification.rename(lambda x : x+str(1), axis = 'columns')
                        regression = regression.rename(lambda x : x+str(1), axis = 'columns')
                    temp_class = pd.concat([temp_class,classification],axis = 1)
                    temp_reg = pd.concat([temp_reg,regression],axis = 1)
                

                classification = temp_class
                regression = temp_reg

                #concatenate ground truth, horizons to dataframe
                classification = pd.concat([pd.Series([ground_truth]*len(classification.index)),pd.Series([horizon]*len(classification.index)),classification],axis = 1).sort_values(by = "rank").reset_index(drop = True)
                regression = pd.concat([pd.Series([ground_truth]*len(regression.index)),pd.Series([horizon]*len(regression.index)),regression],axis = 1).sort_values(by = "rank").reset_index(drop = True)
                
                #append final results to main dataframe
                total_classification = pd.concat([total_classification,classification], axis = 0, sort = False)
                total_regression = pd.concat([total_regression,regression], axis = 0, sort = False)
        total_classification.to_csv('classification.csv')
        total_regression.to_csv('regression.csv')

    #extraction of best hyperparameters based on simple analysis
    elif args.action == "commands":
        validation_dates = [d.split("-")[0]+"-01-01" if d[5:7] <= "06" else d.split("-")[0]+"-07-01" for d in args.dates]
        classification = pd.read_csv('classification.csv')
        regression = pd.read_csv("regression.csv")
        with open(args.output,"w") as out:
            train = "code/train_data_pp.py"
            for gt in args.ground_truth_list:
                for h in args.horizon_list:
                    temp_class = classification.loc[classification['0'] == gt].loc[classification['1'] == int(h)]
                    # print(temp_class,temp_class.loc[classification['1'] == int(h)])
                    temp_reg = regression.loc[regression['0'] == gt].loc[regression['1'] == int(h)]
                    temp_reg = temp_reg[temp_reg.loc[:,"mae"] != 0]
                    temp_reg = temp_reg.sort_values(by= ["rank","coverage","threshold"],ascending = [True,False,True]).reset_index()
                    temp_class = temp_class.sort_values(by= "rank").reset_index()
                    out.write("python "+train+" "+ \
                                " ".join(["-sou",args.source,
                                        "-v",args.version_list,
                                        "-m", args.model,
                                        "-w", "60",
#                                         "-ct", temp_class.loc[:,"threshold"].values[0].strip("(").split(",")[0],
                                        "-ct", "0.5",
                                        "-rt",temp_reg.loc[:,"threshold"].values[0].strip("(").split(",")[0],
#                                         "-rt","10",
                                        "-s",h,
                                        "-gt",gt,
                                        "-o","test",
                                        '-d',args.dates,
                                        ">","/dev/null", "2>&1", "&"]))
                    out.write("\n")


#------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                 No longer in use
#------------------------------------------------------------------------------------------------------------------------------------------------------
    #complex analysis (average over all metals and horizons)
    elif args.action =="complex":
        args.path = args.path.split(',')
        ave_class = pd.DataFrame()
        ave_reg = pd.DataFrame()
        classification = pd.read_csv('classification.csv')
        regression = pd.read_csv("regression.csv")
        classification = classification[(classification['0'].isin(args.ground_truth_list))& (classification['1'].isin([int(i) for i in args.horizon_list]))]

        regression = regression[(regression['0'].isin(args.ground_truth_list))& (regression['1'].isin([int(i) for i in args.horizon_list]))]
        for threshold in sorted(classification['threshold'].unique()):
            temp_class = classification[classification['threshold'] == threshold]
            val_coverage = temp_class['coverage']*temp_class['total_len']
            val_accuracy = temp_class['acc']*val_coverage
            val_total_len = temp_class['total_len']
            temp_class = pd.DataFrame({'threshold':[threshold],'val_accuracy':[sum(val_accuracy)/sum(val_coverage)],'val_coverage':[sum(val_coverage)/sum(val_total_len)]

                })
            ave_class = pd.concat([ave_class,temp_class],axis = 0)
            
        for threshold in sorted(regression['threshold'].unique()):
            temp_reg = regression[regression['threshold'] == threshold]

            val_coverage = temp_reg['coverage']*temp_reg['total_len']
            # val_acc = temp_reg['acc']*val_coverage
            val_mae = temp_reg['mae']*val_coverage
            val_total_len = temp_reg['total_len']

            temp_reg = pd.DataFrame({'threshold':[threshold],
                # 'val_acc':[sum(val_acc)/sum(val_coverage) if sum(val_coverage) > 0 else 0],
                'val_mae':[sum(val_mae)/sum(val_coverage) if sum(val_coverage) > 0 else 0],
                'val_coverage':[sum(val_coverage)/sum(val_total_len)]
                })
            ave_reg = pd.concat([ave_reg,temp_reg],axis = 0)
        ave_class.to_csv("ave_class.csv")
        ave_reg.to_csv("ave_reg.csv")

    #generate commands for every metal and horizon with an overaching threshold
    elif args.action == "average commands":
        validation_dates = [d.split("-")[0]+"-01-01" if d[5:7] <= "06" else d.split("-")[0]+"-07-01" for d in args.dates]
        classification = pd.read_csv('ave_class.csv')
        regression = pd.read_csv("ave_reg.csv")        
        temp_class = classification[classification.loc[:,"val_accuracy"] != 0]            
        temp_class['acc_rank'] = temp_class['val_accuracy'].rank(method = 'min', ascending = False)
        temp_class['cov_rank'] = temp_class['val_coverage'].rank(method = 'min', ascending = False)
        temp_class['rank'] = (temp_class['acc_rank']+temp_class['cov_rank'])/2
        temp_class = temp_class.sort_values(by = 'rank', ascending = True).reset_index()
        temp_class = temp_class.sort_values(by= "rank").reset_index()


        
        temp_reg = regression[regression.loc[:,"val_mae"] != 0]            
        temp_reg['mae_rank'] = temp_reg['val_mae'].rank(method = 'min', ascending = True)
        temp_reg['cov_rank'] = temp_reg['val_coverage'].rank(method = 'min', ascending = False)
        temp_reg['rank'] = (temp_reg['mae_rank']+temp_class['cov_rank'])/2
        temp_reg = temp_reg.sort_values(by = 'rank', ascending = True).reset_index()
        temp_reg = temp_reg.sort_values(by= "rank").reset_index()
        with open(args.output,"w") as out:
            train = "code/train_data_pp.py"

            for gt in args.ground_truth_list:
                for h in args.horizon_list:
                    out.write("python "+train+" "+ \
                                " ".join(["-sou",args.source,
                                        "-v",args.version_list,
                                        "-m", args.model,
                                        "-w", "60",
                                        "-ct", temp_class.loc[0,"threshold"].strip("(").split(",")[0],
                                        "-rt",temp_reg.loc[0,"threshold"].strip("(").split(",")[0],
                                        "-s",h,
                                        "-gt",gt,
                                        "-o","test",
                                        '-d',args.dates,
                                        ">","/dev/null", "2>&1", "&"]))
                    out.write("\n")
