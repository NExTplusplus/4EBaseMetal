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
  parser.add_argument('-c','--config',type=str,default="",
            help='configuration file path')
  parser.add_argument('-s','--steps',type=int,default=5,
            help='steps in the future to be predicted')
  parser.add_argument('-gt', '--ground_truth', help='ground truth column',
            type=str, default="LME_Co_Spot")
  parser.add_argument('-max_iter','--max_iter',type=int,default=100,
            help='max number of iterations')
  parser.add_argument(
      '-sou','--source', help='source of data', type = str, default = "NExT")
  parser.add_argument(
      '-l','--lag', type=int, default = 5, help='lag')
  parser.add_argument('-v','--version', help='version', type = str, default = 'v10')
  parser.add_argument('-o', '--action', type=str, default='train',
            help='train, test, tune')
  parser.add_argument('-d', '--date', help = "the date is the final data's date which you want to use for testing",type=str)	
  parser.add_argument('-max_depth', '--max_depth', type=int)
  parser.add_argument('-learning_rate', '--learning_rate', type=float)
  parser.add_argument('-gamma', '--gamma', type=float)
  parser.add_argument('-min_child', '--min_child', type=int)
  parser.add_argument('-subsample', '--subsample', type=float)
  parser.add_argument('-')
  args = parser.parse_args()
  model = xgboost_online(lag = args.lag, horizon = args.steps, version = args.version, gt = args.ground_truth, date = args.date, source = args.source, path =args.config)
  if args.action=="tune":
  #model = Logistic_online(lag = 5, horizon = 5, version = 'v9', gt = 'LME_Co_Spot', date = '2016-05-01', source = 'NExT')
    model.choose_parameter()
  elif args.action=='train':
    model.train(max_depth = args.max_depth,learning_rate = args.learning_rate, gamma = args.gamma, min_child_weight = args.min_child, subsample = args.subsample)
  else:
    final = model.test()
    final.to_csv("_".join([args.ground_truth,args.date,str(args.steps),args.version])+".csv")