import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
import warnings
from live.ensemble_live import Ensemble_online
from sklearn import metrics


if __name__ == '__main__':
    desc = 'the ensemble model'
    parser = argparse.ArgumentParser(description=desc)

    #script parameters
    parser.add_argument('-o','--action', type = str,
                        help='action that we wish to take, has potential values of : test, tune',
                        default="tune"
                        )

    #result parameters
    parser.add_argument('-s','--horizon_str',type=str,default="1,3,5,10,20,60",
                        help='string which specifies the prediction horizon. It is comma-separated.'
                        )
    parser.add_argument('-c','--config',type=str,
                        help='configuration file path which holds the feature version combination',
                        default="exp/ensemble_tune_all.conf"
                        )
    parser.add_argument('-d', '--dates', type=str, 
                        help = "string of comma-separated dates which identify the total period of deployment by half-years",
                        default = "2014-12-31,2015-06-30,2015-12-31,2016-06-30,2016-12-31.2017-06-30,2017-12-31,2018-06-30,2018-12-31"
                        )
    parser.add_argument('-gt', '--ground_truth_str', type=str, 
                        help = 'string which specifies the metals. It is comma-separated.',
                        default = "LME_Al_Spot,LME_Cu_Spot,LME_Pb_Spot,LME_Ni_Spot,LME_Xi_Spot,LME_Zn_Spot"
                        )
    parser.add_argument('-t','--target', type = str,
                        help='target for tune. can take values: window, fv',
                        default="window"
                        )
    args = parser.parse_args()
    args.horizon_str = args.horizon_str.split(",")
    args.ground_truth_str = args.ground_truth_str.split(",")

    #read ensemble configuration
    with open(args.config) as f:
        args.config = json.load(f)
    
    #process the configuration to generate from comma-sepearated string
    for horizon in args.config.keys():
        for key in args.config[horizon].keys():
            args.config[horizon][key] = args.config[horizon][key].split(',')
    
    #case if tune
    if args.action == "tune":
        for horizon in [int(step) for step in args.horizon_str]:
            for ground_truth in args.ground_truth_str:
                ensemble = Ensemble_online(horizon=horizon,gt = ground_truth,dates = args.dates, feature_version = args.config)
                ensemble.tune(args.target)
                
                #if tuning for delete feature versions, don't need to loop
                if args.target == "fv":
                    break
            
            if args.target == "fv":
                break

    #case if test (ie making predictions)
    elif args.action == "test":
        for horizon in [int(step) for step in args.horizon_str]:
            for ground_truth in args.ground_truth_str:
                for date in args.dates.split(","):
                    ensemble = Ensemble_online(horizon=horizon,gt = ground_truth,dates = args.dates, feature_version = args.config)
                    prediction = ensemble.test()
