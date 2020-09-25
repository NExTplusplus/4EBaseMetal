import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
import warnings
from live.post_process_live import Post_process_live
from itertools import product


if __name__ == '__main__':
    desc = 'the post process'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-gt','--ground_truth', type = str, default = "LME_Co_Spot")
    parser.add_argument('-s','--steps', type = int, default = 1)
    
    parser.add_argument('-d', '--dates', help = "the date is the final data's date which you want to use for testing",type=str,default = '2017-06-30,2017-12-31,2018-06-30,2018-12-31')
    parser.add_argument(
                '-v','--versions', type = str,
                help='versions used in ensemble',
                default="v1:v1:v1"
        )
    parser.add_argument(
                '-o','--action', type = str,
                help='action',
                default="test"
        )
    parser.add_argument(
                '-m','--model', type = str,
                help ='method of post process',
                default = 'substitution,analyst'
        )
    parser.add_argument(
                '-w','--window',type = int,
                help = "window size for regression",
                default = 60
    )
    parser.add_argument(
                '-ct','--class_threshold', type = float,
                help = "classification threshold"
    )
    parser.add_argument(
                '-rt','--reg_threshold', type = float,
                help = "regression threshold"
    )
    parser.add_argument(
                '-sou','--source',type = str,
                help = "source of data",
                default = "NExT"
    )

    args = parser.parse_args()
    args.model = [x.strip() for x in args.model.split(',')]
    substitution = args.model[1] if len(args.model) > 1 else args.model[0]
    args.versions = args.versions.split(',')
    pp = Post_process_live(args.ground_truth,args.steps,args.dates,args.model[0],args.versions[::-1])
    if args.action == "tune":
        inputs = {"source":args.source, "class_version":args.versions[1], "reg_version":args.versions[0]}
        classification, regression = pp.tune(inputs)
        regression.to_csv(os.path.join('result','validation','post_process',args.model[0],'_'.join([args.ground_truth,args.versions[0],str(args.steps),"regression.csv"])))
        classification.to_csv(os.path.join('result','validation','post_process',args.model[0],'_'.join([args.ground_truth,args.versions[1],str(args.steps),"classification.csv"])))
    
    if args.action == "test":
        inputs = {"source":args.source, "class_version":args.versions[1], "reg_version":args.versions[0], 'substitution': substitution, \
                    "class_threshold": args.class_threshold, "reg_threshold":args.reg_threshold, "reg_window":args.window}
        pp.test(inputs)
        
