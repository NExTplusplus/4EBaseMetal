import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
import warnings
from live.Ensemble_live import Ensemble_online
from sklearn import metrics


if __name__ == '__main__':
    desc = 'the ensemble model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s','--steps',type=str,default="1,3,5,10,20,60",
                        help='steps in the future to be predicted')
    parser.add_argument('-c','--config',type=str,default="exp/ensemble_tune.conf",
                        help='configuration file path')
    parser.add_argument('-d', '--dates', help = "the date is the final data's date which you want to use for testing",type=str
                        ,default = "2014-12-31,2015-06-30,2015-12-31,2016-06-30,2016-12-31.2017-06-30,2017-12-31,2018-06-30,2018-12-31")
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                                            type=str, default="LME_Al_Spot,LME_Co_Spot,LME_Le_Spot,LME_Ni_Spot,LME_Ti_Spot,LME_Zi_Spot")
    parser.add_argument(
                '-o','--action', type = str,
                help='action',
                default="tune,test"
        )
    parser.add_argument(
                '-t','--target', type = str,
                help='target for tune',
                default="window"
        )
    args = parser.parse_args()
    args.steps = args.steps.split(",")
    args.ground_truth = args.ground_truth.split(",")

    with open(args.config) as f:
        args.config = json.load(f)
    
    for key in args.config.keys():
        args.config[key] = args.config[key].split(',')
    
    if args.action == "tune":
        for horizon in [int(step) for step in args.steps]:
            for ground_truth in args.ground_truth:
                ensemble = Ensemble_online(horizon=horizon,gt = ground_truth,dates = args.dates, feature_version = args.config)
                ensemble.tune(args.target)
                if args.target == "fv":
                    break
            
            if args.target == "fv":
                break

    elif args.action == "test":
        for horizon in [int(step) for step in args.steps]:
            for ground_truth in args.ground_truth:
                for date in args.dates.split(","):
                    print(ground_truth,horizon,date)
                    ensemble = Ensemble_online(horizon=horizon,gt = ground_truth,dates = args.dates, feature_version = args.config)
                    prediction = ensemble.test()