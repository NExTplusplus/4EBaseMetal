import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
import warnings
from live.xgboost_live import XGBoost_online

if __name__ == '__main__':
    desc = 'the XGBoost model'
    parser = argparse.ArgumentParser(description=desc)

    #script parameter
    parser.add_argument('-o', '--action', type=str, 
                        help='action that we wish to take, has potential values of : train, test, tune',
                        default='train'
                        )

    #result parameter
    parser.add_argument('-s','--horizon',type=int,
                        help='the prediction horizon',
                        default=5
                        )
    parser.add_argument('-gt', '--ground_truth', type=str, 
                        help='the name of the column that we are predicting either value or direction',
                        default="LME_Cu_Spot"
                        )
    parser.add_argument('-sou','--source', type = str, 
                        help='source of data', 
                        default = "NExT"
                        )
    parser.add_argument('-v','--version',  type = str, 
                        help='feature version for data',
                        default = 'v10'
                        )
    parser.add_argument('-d', '--date', type=str, 
                        help = "string of comma-separated dates which identify the total period of deployment by half-years"
                        )

    #hyperparameters
    parser.add_argument('-l','--lag', type=int, 
                        help='lag', 
                        default = 5
                        )
    parser.add_argument('-max_depth', '--max_depth', type=int)
    parser.add_argument('-learning_rate', '--learning_rate', type=float)
    parser.add_argument('-gamma', '--gamma', type=float)
    parser.add_argument('-min_child', '--min_child', type=int)
    parser.add_argument('-subsample', '--subsample', type=float)
    args = parser.parse_args()
    #initialize model
    model = XGBoost_online(lag = args.lag, horizon = args.horizon, version = args.version, gt = args.ground_truth, date = args.date, source = args.source)
    
    #case if action is tune
    if args.action=="tune":
        model.tune()
    
    #case if action is train
    elif args.action=='train':
        model.train(max_depth = args.max_depth,learning_rate = args.learning_rate, gamma = args.gamma, min_child_weight = args.min_child, subsample = args.subsample)
    
    #csae if action is test
    else:
        final = model.test()
