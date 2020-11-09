import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],"..")))
from live.linear_live import Linear_online
import warnings
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn import metrics
from sklearn.model_selection import KFold

if __name__ == '__main__':
    desc = 'the linear model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-c','--config',type=str,
                        help='configuration file path',
                        default=""
                        )
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
    parser.add_argument('-l','--lag', type=int, 
                        help='lag',
                        default = 5
                        )
    parser.add_argument('-v','--version', type = str, 
                        help='feature version for data', 
                        default = 'v10'
                        )
    parser.add_argument('-o', '--action', type=str, 
                        help='action that we wish to take, has potential values of : train, test',
                        default='train'
                        )
    parser.add_argument('-d', '--date',type=str,
                        help = "string of comma-separated dates which identify the total period of deployment by half-years"
                        )
    args = parser.parse_args()

    #initialize model
    model = Linear_online(lag = args.lag, horizon = args.horizon, version = args.version, gt = args.ground_truth, date = args.date, source = args.source, path =args.config)

    if args.action=='train':
        model.train()
    else:
        final = model.test()
