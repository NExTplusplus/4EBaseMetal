import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
import warnings
from live.ensemble_live_new import Ensemble_online
from sklearn import metrics


if __name__ == '__main__':
    desc = 'the ensemble model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s','--steps',type=str,default="5",
                        help='steps in the future to be predicted')
    parser.add_argument('-c','--config',type=str,default="exp/ensemble_tune.conf",
                        help='configuration file path')
    parser.add_argument('-l','--length',type=int,default=1,
                        help='steps in the future to be predicted')
    parser.add_argument('-model','--model', help='which single model we want to ensemble', type = str, default = 'alstm')
    parser.add_argument('-d', '--dates', help = "the date is the final data's date which you want to use for testing",type=str)
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                                            type=str, default="LME_Co_Spot")
    parser.add_argument('-hier', '--hierarchical', help='ground truth column',
                                            type=str, default='True')
    parser.add_argument(
                '-v','--version', type = str,
                help='which version model is to be deleted',
                default=""
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
                '-w','--window', type = str,
                help='window',
                default='5:5:5'
        )
    parser.add_argument(
                '-o','--action', type = str,
                help='action',
                default="tune,test"
        )
    args = parser.parse_args()
    args.steps = args.steps.split(",")
    args.ground_truth = args.ground_truth.split(",")
    args.config = args.config
    hierarchical = args.hierarchical == "True"
    args.config = os.path.join(*args.config.split('/'))
    
    if args.action == "tune":
        for horizon in [int(step) for step in args.steps]:
            for ground_truth in args.ground_truth:
                ensemble = Ensemble_online(horizon=horizon,gt = ground_truth,dates = args.dates, window = args.window, version = args.version, config = args.config, hierarchical = args.hierarchical)
                results = ensemble.choose_parameter()
                results.to_csv(os.path.join('result','validation','ensemble','_'.join([ground_truth,str(horizon),args.version+".csv"])))
    if args.action == "test":
        for horizon in [int(step) for step in args.steps]:
            for ground_truth in args.ground_truth:
                for date in args.dates.split(","):
                    print(ground_truth,horizon,date)
                    ensemble = Ensemble_online(horizon=horizon,gt = ground_truth,dates = args.dates, window = args.window, version = args.version, config = args.config, hierarchical = args.hierarchical) 
                    prediction = ensemble.predict(date, args.version, args.sm_methods, args.ens_method).to_frame()
                    prediction.columns = ['result']
                    prediction.to_csv(os.path.join(os.getcwd(),"result","prediction","ensemble",'_'.join([ground_truth,date,str(horizon),args.version,args.sm_methods,args.ens_method,'hier',str(args.hierarchical)+".csv"])))
    if args.action == "delete":
        ans = {'horizon':[],"ground_truth":[],"version":[]}
        for horizon in [int(step) for step in args.steps]:
            for ground_truth in args.ground_truth:
                check = 0
                for date in args.dates.split(","):
                    validation_date = date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" 
                    if validation_date+"_acc" not in ans.keys():
                        ans[validation_date+"_acc"]=[]
                        ans[validation_date+"_length"]= []
                    print(ground_truth,horizon,date)
                    ensemble = Ensemble_online(horizon=horizon,gt = ground_truth,dates = args.dates, window = args.window, version = args.version, config = args.config, hierarchical = args.hierarchical) 
                    prediction = ensemble.delete_model(date, args.version, args.sm_methods,args.ens_method, args.length)
                    for col in prediction.columns:
                        label = pd.read_csv("data/Label/"+ground_truth+"_h"+str(horizon)+"_"+validation_date+"_label.csv",index_col = 0)[:len(prediction.index)]
                        acc = metrics.accuracy_score(label,prediction[col])
                        if check == 0:
                            ans['version'].append(col)
                            ans['horizon'].append(horizon)
                            ans['ground_truth'].append(ground_truth)
                        ans[validation_date+"_acc"].append(acc)
                        ans[validation_date+"_length"].append(len(prediction))
                    check+=1
        ans = pd.DataFrame(ans)
        average = np.zeros(len(ans.index))
        length = np.zeros(len(ans.index))
        for date in args.dates.split(","):
            validation_date = date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" 
            average += ans[validation_date+"_acc"]*ans[validation_date+"_length"]
            length += ans[validation_date+"_length"]
        ans['average'] = average/length
        ans.sort_values(by = ['version','horizon','ground_truth'],ascending = [True,True,True],inplace = True)
        ans.to_csv("delete_model"+'_'+','.join(args.steps)+'_'+str(args.length)+".csv")
    



        