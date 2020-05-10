import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

print(os.curdir)
sys.path.insert(0,os.path.abspath(os.curdir))
final_ans = pd.DataFrame()
for v in ["v16","v26"]:
  print(v)
  ave_loss_path = os.path.join(sys.path[0],"data","lstm_lt_prediction","ave_loss",v)
  ave_acc_path = os.path.join(sys.path[0],"data","lstm_lt_prediction","ave_acc",v)
  best_loss_path = os.path.join(sys.path[0],"data","lstm_lt_prediction","best_loss",v)
  best_acc_path = os.path.join(sys.path[0],"data","lstm_lt_prediction","best_acc",v)
  labels = os.path.join(sys.path[0],"data","Label")
  files = os.listdir(ave_loss_path)
  print(files)
  for f in files:
    if ".txt" not in f:
      continue
    ave_loss_pred = np.loadtxt(os.path.join(ave_loss_path,f))
    ave_acc_pred = np.loadtxt(os.path.join(ave_acc_path,f))
    best_loss_pred = np.loadtxt(os.path.join(best_loss_path,f))
    best_acc_pred = np.loadtxt(os.path.join(best_acc_path,f))
    ground_truths_list = ['LME_Co_Spot','LME_Al_Spot','LME_Le_Spot','LME_Ni_Spot','LME_Zi_Spot','LME_Ti_Spot']
    all_length = len(ave_loss_pred)
    #print("the length of the ALSTM is {}".format(all_length))
    metal_length = int(all_length/6)
    for i,gt in enumerate(ground_truths_list):
      ans = {"version":[],"horizon":[],"method":[],"date":[],"length":[],"ground truth":[],"acc":[]}
      ave_loss = ave_loss_pred[i*metal_length:(i+1)*metal_length]
      ave_acc = ave_acc_pred[i*metal_length:(i+1)*metal_length]
      best_loss = best_loss_pred[i*metal_length:(i+1)*metal_length]
      best_acc = best_acc_pred[i*metal_length:(i+1)*metal_length]
      label = pd.read_csv(os.path.join(labels,"_".join([gt,"h"+f.split("_")[1],f.split("_")[0],"label.csv"])),index_col = 0).values.tolist()
      ans["version"] = [v]*4
      ans["horizon"] = [f.split("_")[1]]*4
      ans["method"] = ["ave_loss","ave_acc","best_loss","best_acc"]
      ans["date"] = [f.split("_")[0]]*4
      ans["length"] = [metal_length]*4
      ans['ground truth'] = [gt]*4
      ans['acc'] = [accuracy_score(label,ave_loss),accuracy_score(label,ave_acc),accuracy_score(label,best_loss),accuracy_score(label,best_acc)]
      final_ans = pd.concat([final_ans,pd.DataFrame(ans)],axis = 0)
      print(final_ans)
final_ans.to_csv("alstm.csv")




