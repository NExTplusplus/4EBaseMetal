import pandas as pd
import numpy as np
from copy import copy
import argparse
import os

if __name__ == '__main__':
    desc = 'the script for Ensemble'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s','--step_list',type=str,default="1,3,5,10,20,60",
                        help='list of horizons to be calculated, separated by ","')
    parser.add_argument('-gt', '--ground_truth_list', help='list of ground truths, separated by ","',
                        type=str, default="LME_Co_Spot,LME_Al_Spot,LME_Ni_Spot,LME_Ti_Spot,LME_Zi_Spot,LME_Le_Spot")
    parser.add_argument(
        '-v','--version_list', help='list of versions, separated by ","', type = str, default = 'v10'
    )
    parser.add_argument(
        '-sm','--sm_methods', type = str,
        help='method',
        default='vote:vote:vote'
    )
    parser.add_argument(
        '-ens','--ens_method', type = str,
        help='ensemble method',
        default='vote'
    )
    parser.add_argument(
        '-hier','--hierarchical', type = str,
        help='hierarchical',
        default='True'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
    parser.add_argument('-o', '--action', type=str, default='commands',
                        help='commands, testing')
    parser.add_argument('-d','--dates',type = str, help = "dates", default = "2014-12-31,2015-06-30,2015-12-31,2016-06-30,2016-12-31,2017-06-30,2017-12-31,2018-06-30,2018-12-31")
    parser.add_argument('-p','--path',type =str, help='path to 4EBaseMetal folder',default ='~/NEXT/4EBaseMetal')

    args = parser.parse_args()
    args.step_list = args.step_list.split(",")
    args.ground_truth_list = args.ground_truth_list.split(",")
    args.hierarchical = args.hierarchical == "True"
    print(args.hierarchical)
    

    if args.action == "commands":
        j = 0
        with open(args.output,"w") as out:
            for i,version in enumerate(args.version_list.split(",")):
                print(version)
                train = "code/train_data_ensemble.py"
                for gt in args.ground_truth_list:
                    for h in args.step_list:
                        if '_'.join([gt,h,version])+".csv" not in os.listdir(args.path):
                            continue
                        temp = pd.read_csv(os.path.join(args.path,'_'.join([gt,h,version])+".csv"))
                        ans = temp.sort_values(by = ['model','average','window'],ascending = [True,False, True])
                        weight = 0
                        if args.sm_methods == 'vote:vote:vote':
                            if args.hierarchical:
                                weight =str(ans.iloc[8,1])
                            else:
                                weight = str(ans.iloc[12,1])
                        elif args.sm_methods == 'weight:weight:weight':
                            if args.hierarchical:
                                weight = str(ans.iloc[16,1])
                            else:
                                weight = str(ans.iloc[20,1])
                        out.write("python "+train+" "+" ".join(["-v",version,"-s",h,'-o','test',"-w",':'.join([str(ans.iloc[4,1]),str(ans.iloc[-4,1]),str(ans.iloc[0,1]),weight]),"-gt",gt,'-d',args.dates,'-sm',args.sm_methods,'-ens',args.ens_method,'-hier',str(args.hierarchical),">","/dev/null", "2>&1", "&"]))
                        out.write("\n")
                        j+=1
                        if j % 10 == 0:
                            j = 0
                            out.write("sleep 1m\n")



