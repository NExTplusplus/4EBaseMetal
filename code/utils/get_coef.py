import argparse
from datetime import datetime
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tkinter
import time
import joblib
from copy import copy

if __name__ == '__main__':
  desc = 'the logistic regression model'
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('-s','--steps',type=int,default=1,
                    help='steps in the future to be predicted')
  parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                      type=str, default="LMCADY")
  parser.add_argument('-max_iter','--max_iter',type=int,default=100,
                    help='max number of iterations')
  parser.add_argument(
    '-min', '--model_path', help='path to load model',
    type=str, default='../../exp/log_reg/model'
  )
  parser.add_argument(
    '-v','--version', help='version', type = int, default = 1
  )
  parser.add_argument(
    '-mout', '--model_save_path', type=str, help='path to save model',
    default='../../exp/log_reg/model'
  )
  parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
  parser.add_argument('-o', '--action', type=str, default='train',
                    help='train, test, tune')

  args = parser.parse_args()

  gt = ""

  if args.ground_truth == "LMCADY":
    gt = "Co"
  if args.ground_truth == "LMAHDY":
    gt = "Al"
  if args.ground_truth == "LMPBDY":
    gt = "Le"
  if args.ground_truth == "LMZSDY":
    gt = "Zi"
  if args.ground_truth == "LMSNDY":
    gt = "Ti"
  if args.ground_truth == "LMNIDY":
    gt = "Ni"
  if args.ground_truth == "All":
    gt = "All"

  sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],'..')))
  
  directory = os.listdir(os.path.abspath(os.path.join(sys.path[0],"..","..","mingwei","NExT","4EBaseMetal","exp",str(args.steps)+"d",gt,"logistic_regression","v"+str(args.version))))

  for n in range(61):
    coef = None
    if "n"+str(n)+".joblib" in directory:
      model = joblib.load(os.path.abspath(os.path.join(sys.path[0],"..","..","mingwei","NExT","4EBaseMetal","exp",str(args.steps)+"d",gt,"logistic_regression","v"+str(args.version),"n"+str(n)+".joblib")))
      coef = model.coef_
    
    if coef is not None:
      col =""
      with open(os.path.abspath(os.path.join(sys.path[0],"..","..","mingwei","NExT","Results","LMCADY_h"+str(args.steps)+"_v"+str(args.version)+".csv"))) as f:
        lines = f.readlines()
        rel_line = lines[n].split(",")
        lag = rel_line[1]
        if args.version in [1]:
          col = ""
        elif args.version in [2]:
          col ='LME_Co_Spot,LME_Al_Spot,LME_Le_Spot,LME_Ni_Spot,LME_Zi_Spot,LME_Ti_Spot,LME_Al_Open,LME_Al_High,LME_Al_Low,LME_Al_Close,LME_Co_Open,LME_Co_High,LME_Co_Low,LME_Co_Close,LME_Ti_Open,LME_Ti_High,LME_Ti_Low,LME_Ti_Close,LME_Le_Open,LME_Le_High,LME_Le_Low,LME_Le_Close,LME_Ni_Open,LME_Ni_High,LME_Ni_Low,LME_Ni_Close,LME_Zi_Open,LME_Zi_High,LME_Zi_Low,LME_Zi_Close,LME_Co_nOI,LME_Al_nOI,LME_Le_nOI,LME_Ni_nOI,LME_Zi_nOI,LME_Ti_nOI,LME_Al_n3MSpread,LME_Al_nVolume,LME_Co_n3MSpread,LME_Co_nVolume,LME_Ti_n3MSpread,LME_Ti_nVolume,LME_Le_n3MSpread,LME_Le_nVolume,LME_Ni_n3MSpread,LME_Ni_nVolume,LME_Zi_n3MSpread,LME_Zi_nVolume,'
        elif args.version in [3]:
          col ='LME_Co_Spot,COMEX_GC_lag1_Open,COMEX_GC_lag1_High,COMEX_GC_lag1_Low,COMEX_GC_lag1_Close,COMEX_HG_lag1_Open,COMEX_HG_lag1_High,COMEX_HG_lag1_Low,COMEX_HG_lag1_Close,COMEX_PA_lag1_Open,COMEX_PA_lag1_High,COMEX_PA_lag1_Low,COMEX_PA_lag1_Close,COMEX_PL_lag1_Open,COMEX_PL_lag1_High,COMEX_PL_lag1_Low,COMEX_PL_lag1_Close,COMEX_SI_lag1_Close,DCE_AC_Open,DCE_AC_High,DCE_AC_Low,DCE_AC_Close,DCE_AK_Open,DCE_AK_High,DCE_AK_Low,DCE_AK_Close,DCE_AE_Open,DCE_AE_High,DCE_AE_Low,DCE_AE_Close,DXY,SPX,SX5E,UKX,VIX,HSI,NKY,SHCOMP,SHSZ300,LME_Al_Spot,LME_Le_Spot,LME_Ni_Spot,LME_Zi_Spot,LME_Ti_Spot,LME_Al_Open,LME_Al_High,LME_Al_Low,LME_Al_Close,LME_Co_Open,LME_Co_High,LME_Co_Low,LME_Co_Close,LME_Ti_Open,LME_Ti_High,LME_Ti_Low,LME_Ti_Close,LME_Le_Open,LME_Le_High,LME_Le_Low,LME_Le_Close,LME_Ni_Open,LME_Ni_High,LME_Ni_Low,LME_Ni_Close,LME_Zi_Open,LME_Zi_High,LME_Zi_Low,LME_Zi_Close,SHFE_Al_Open,SHFE_Al_High,SHFE_Al_Low,SHFE_Al_Close,SHFE_Co_Open,SHFE_Co_High,SHFE_Co_Low,SHFE_Co_Close,SHFE_Le_Open,SHFE_Le_High,SHFE_Le_Low,SHFE_Le_Close,SHFE_Zi_Open,SHFE_Zi_High,SHFE_Zi_Low,SHFE_Zi_Close,SHFE_RT_Open,SHFE_RT_High,SHFE_RT_Low,SHFE_RT_Close,COMEX_GC_lag1_nVolume,COMEX_GC_lag1_nOI,COMEX_HG_lag1_nVolume,COMEX_HG_lag1_nOI,COMEX_PA_lag1_nVolume,COMEX_PA_lag1_nOI,COMEX_PL_lag1_nVolume,COMEX_PL_lag1_nOI,COMEX_SI_lag1_nVolume,COMEX_SI_lag1_nOI,DCE_AC_nVolume,DCE_AC_nOI,DCE_AK_nVolume,DCE_AK_nOI,DCE_AE_nVolume,DCE_AE_nOI,LME_Co_nOI,LME_Al_nOI,LME_Le_nOI,LME_Ni_nOI,LME_Zi_nOI,LME_Ti_nOI,LME_Al_n3MSpread,LME_Al_nVolume,LME_Co_n3MSpread,LME_Co_nVolume,LME_Ti_n3MSpread,LME_Ti_nVolume,LME_Le_n3MSpread,LME_Le_nVolume,LME_Ni_n3MSpread,LME_Ni_nVolume,LME_Zi_n3MSpread,LME_Zi_nVolume,SHFE_Al_nEx3MSpread,SHFE_Al_nExSpread,SHFE_Al_nVolume,SHFE_Al_nOI,SHFE_Co_nEx3MSpread,SHFE_Co_nExSpread,SHFE_Co_nVolume,SHFE_Co_nOI,SHFE_Le_nEx3MSpread,SHFE_Le_nExSpread,SHFE_Le_nVolume,SHFE_Le_nOI,SHFE_Zi_nEx3MSpread,SHFE_Zi_nExSpread,SHFE_Zi_nVolume,SHFE_Zi_nOI,SHFE_RT_nVolume,SHFE_RT_nOI,'
        if args.ground_truth =="All":
          col = 'self,'+col
        with open(os.path.abspath(os.path.join(sys.path[0],"..","..","mingwei","NExT","Results","LMCADY_h"+str(args.steps)+"_v"+str(args.version)+"_n"+str(n)+"_coef.csv")),"w") as out:
          out.write(col+"\n")
          num = int((len(coef[0])/(int(lag))))
          
          for i in range(int(lag)):
            out.write(','.join(map(str,coef[0][i*num:(i+1)*num])))
            out.write('\n')

  for n in range(61):
    get_coef(n,1,os.path.abspath(os.path.join(sys.path[0],"..","..","mingwei","NExT","Results","LMCADY_h3_v1.csv")))

      
      