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
import statistics

from data.load_rnn import load_pure_log_reg




sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],'..')))
directory = os.listdir(os.path.abspath(os.path.join(sys.path[0],"4EBaseMetal","exp","log_reg")))

tra_date = '2005-01-10'
val_date = '2016-06-01'
tes_date = '2016-12-21'
split_dates = [tra_date, val_date, tes_date]
len_ma = 5
len_update = 30
tol = 1e-7

with open("../../NExT/4EBaseMetal/exp/LMCADY_v2.conf") as fin:
    fname_columns = json.load(fin)[0]

for h in [3]:
  for v in [2]:
    for n in range(60):
      model = joblib.load(os.path.abspath(os.path.join(sys.path[0],"4EBaseMetal","exp","log_reg","LME_Co_Spot_h"+str(h)+"_n"+str(n+1)+"_v"+str(v+1)+".joblib")))
      with open(os.path.abspath(os.path.join(sys.path[0],"Results","LMCADY_h"+str(h)+"_v"+str(v+1)+".csv"))) as f:
        lines = f.readlines()
        rel_line = lines[n+1].split(",")
        lag = rel_line[1]
        norm_volume = rel_line[2]
        norm_3m_spread = rel_line[3]
        norm_ex = rel_line[4]

        X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params= load_pure_log_reg(
            fname_columns, 'log_1d_return', split_dates, gt_column = "LME_Co_Spot", T = int(lag),
            S = h,
            vol_norm = norm_volume, ex_spread_norm = norm_ex, spot_spread_norm = norm_3m_spread
        )

        X_va = np.concatenate(X_va)
        y_va = np.concatenate(y_va).flatten()

        # print(model.predict_proba(X_tr))

        with open(os.path.abspath(os.path.join(sys.path[0],"Results","LMCADY_h3_v3","LMCADY_h"+str(h)+"_n"+str(n+1)+"_v"+str(v+1)+".csv")),"w") as out:
          out.write("Negative Prob,Positive Prob,Prob Diff,Pred,TrueVal\n")
          prob = model.predict_proba(X_va)
          prob_diff = [abs(p[1]-p[0]) for p in prob]
          pred = model.predict(X_va)
          med = statistics.median(prob_diff)
          top_pred = []
          top_true = []
          bot_pred = []
          bot_true = []
          for i in range(len(prob)):
            out.write(str(prob[i][0])+","+str(prob[i][1])+","+str(prob_diff[i])+","+str(pred[i])+","+str(y_va[i])+"\n")
            if (prob_diff[i] > med):
              top_pred.append(pred[i])
              top_true.append(y_va[i])
            else:
              bot_pred.append(pred[i])
              bot_true.append(y_va[i])
          top_pred = np.array(top_pred)
          top_true = np.array(top_true)
          bot_pred = np.array(bot_pred)
          bot_true = np.array(bot_true)


          out.write("\n\n\n")
          no_true = sum(np.equal(pred,y_va))
          no_TT = sum(np.multiply(pred + 1,y_va + 1))/4
          no_FF = sum(np.multiply(pred - 1,y_va - 1))/4
          no_TF = -sum(np.multiply(pred + 1,y_va - 1))/4
          no_FT = -sum(np.multiply(pred - 1,y_va + 1))/4
          top_no_true = sum(np.equal(top_pred,top_true))
          top_no_TT = sum(np.multiply(top_pred + 1,top_true + 1))/4
          top_no_FF = sum(np.multiply(top_pred - 1,top_true - 1))/4
          top_no_TF = -sum(np.multiply(top_pred + 1,top_true - 1))/4
          top_no_FT = -sum(np.multiply(top_pred - 1,top_true + 1))/4
          bot_no_true = sum(np.equal(bot_pred,bot_true))
          bot_no_TT = sum(np.multiply(bot_pred + 1,bot_true + 1))/4
          bot_no_FF = sum(np.multiply(bot_pred - 1,bot_true - 1))/4
          bot_no_TF = -sum(np.multiply(bot_pred + 1,bot_true - 1))/4
          bot_no_FT = -sum(np.multiply(bot_pred - 1,bot_true + 1))/4
          out.write(",Total,Top,Bot\n")
          out.write(",".join(["Acc",str(no_true/len(pred)),str(top_no_true/len(top_pred)),str(bot_no_true/len(top_pred))]))
          out.write("\n")
          out.write(",".join(["TT",str(no_TT),str(top_no_TT),str(bot_no_TT)]))
          out.write("\n")
          out.write(",".join(["TF",str(no_TF),str(top_no_TF),str(bot_no_TF)]))
          out.write("\n")
          out.write(",".join(["FT",str(no_FT),str(top_no_FT),str(bot_no_FT)]))
          out.write("\n")
          out.write(",".join(["FF",str(no_FF),str(top_no_FF),str(bot_no_FF)]))

          out.close()

    
