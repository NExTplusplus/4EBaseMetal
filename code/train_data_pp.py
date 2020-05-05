import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
import warnings
from live.post_process_live import *
from itertools import product


if __name__ == '__main__':
    desc = 'the post process'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-c','--config',type=str,default="exp/classification_post_process.conf",
                        help='combination of ground truth and horizon')
    parser.add_argument('-d', '--dates', help = "the date is the final data's date which you want to use for testing",type=str,default = '2017-06-30,2017-12-31,2018-06-30,2018-12-31')
    parser.add_argument('-hier', '--hierarchical', help='hierarchical boolean',
                                            type=str, default='True')
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
                default = 'substitution > analyst '

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
    args = parser.parse_args()
    args.model = [x.strip() for x in args.model.split('>')]
    args.hierarchical = args.hierarchical == "True"
    with open(args.config) as f:
        comb = json.load(f)[args.model[0]][args.model[1]][args.versions]
    combinations = []
    for key in comb.keys():
        combinations.extend([list(x) for x in product([key],comb[key])])
    
    if args.action == "test":
        if args.model[0] == 'substitution':
            pp = Post_process_substitution(applied_comb = combinations ,dates = args.dates, version = args.model[1])
            results = pp.predict(args.sm_methods,args.ens_method,args.hierarchical,args.versions)



        