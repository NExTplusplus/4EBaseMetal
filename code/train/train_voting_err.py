import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from sklearn import metrics
import math

np.random.seed(0)

for horizon in [1,3,5]:
  for ground_truth in ["LME_Co_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot","LME_Le_Spot"]:
      for window_size in [5,10,15,20,25,30]:
        for date in ["2014-07-01","2015-01-01","2015-07-01","2016-01-01","2016-07-01","2017-01-01","2017-07-01","2018-01-01","2018-07-01"]:
          lr_v3 = np.loadtxt("data/LR_probability/"+ground_truth+str(horizon)+"_"+date+"_lr_v3_probability.txt")
          lr_v5 = np.loadtxt("data/LR_probability/"+ground_truth+str(horizon)+"_"+date+"_lr_v5_probability.txt")
          lr_v7 = np.loadtxt("data/LR_probability/"+ground_truth+str(horizon)+"_"+date+"_lr_v7_probability.txt")
          lr_v9 = np.loadtxt("data/LR_probability/"+ground_truth+str(horizon)+"_"+date+"_lr_v9_probability.txt")
          if ground_truth=="LME_Co_Spot":
            lr_v10 = np.loadtxt("data/LR_probability/"+'LMCADY'+str(horizon)+"_"+date+"_lr_v10_probability.txt")
            lr_v12 = np.loadtxt("data/LR_probability/"+'LMCADY'+str(horizon)+"_"+date+"_lr_v12_probability.txt")
          elif ground_truth=='LME_Al_Spot':
            lr_v10 = np.loadtxt("data/LR_probability/"+'LMAHDY'+str(horizon)+"_"+date+"_lr_v10_probability.txt")
            lr_v12 = np.loadtxt("data/LR_probability/"+'LMAHDY'+str(horizon)+"_"+date+"_lr_v12_probability.txt")
          elif ground_truth=='LME_Le_Spot':
            lr_v10 = np.loadtxt("data/LR_probability/"+'LMPBDY'+str(horizon)+"_"+date+"_lr_v10_probability.txt")
            lr_v12 = np.loadtxt("data/LR_probability/"+'LMPBDY'+str(horizon)+"_"+date+"_lr_v12_probability.txt")
          elif ground_truth=='LME_Ni_Spot':
            lr_v10 = np.loadtxt("data/LR_probability/"+'LMNIDY'+str(horizon)+"_"+date+"_lr_v10_probability.txt")
            lr_v12 = np.loadtxt("data/LR_probability/"+'LMNIDY'+str(horizon)+"_"+date+"_lr_v12_probability.txt")
          elif ground_truth=='LME_Ti_Spot':
            lr_v10 = np.loadtxt("data/LR_probability/"+'LMSNDY'+str(horizon)+"_"+date+"_lr_v10_probability.txt")
            lr_v12 = np.loadtxt("data/LR_probability/"+'LMSNDY'+str(horizon)+"_"+date+"_lr_v12_probability.txt")
          elif ground_truth=='LME_Zi_Spot':
            lr_v10 = np.loadtxt("data/LR_probability/"+'LMZSDY'+str(horizon)+"_"+date+"_lr_v10_probability.txt")
            lr_v12 = np.loadtxt("data/LR_probability/"+'LMZSDY'+str(horizon)+"_"+date+"_lr_v12_probability.txt")
          lr_v23 = np.loadtxt("data/LR_probability/"+ground_truth+str(horizon)+"_"+date+"_lr_v23_probability.txt")
          y_va = pd.read_csv("data/Label/"+ground_truth+"_h"+str(horizon)+"_"+date+"_label"+".csv")
          y_va = list(y_va['Label'])
          length=0
          result_v3_error = []
          result_v5_error = []
          result_v7_error = []
          result_v9_error = []
          result_v10_error = []
          result_v12_error = []
          result_v23_error = []
          final_list_v3 = []
          final_list_v5 = []
          final_list_v7 = []
          final_list_v9 = []
          final_list_v10 = []
          final_list_v12 = []
          final_list_v23 = []
          results = []
          final_list_1 = []
          df = pd.DataFrame()
          for j in range(len(lr_v3)):
            if lr_v3[j]>0.5:
              final_list_v3.append(1)
            else:
              final_list_v3.append(0)

            if lr_v5[j]>0.5:
              final_list_v5.append(1)
            else:
              final_list_v5.append(0)

            if lr_v7[j]>0.5:
              final_list_v7.append(1)
            else:
              final_list_v7.append(0)

            if lr_v9[j]>0.5:
              final_list_v9.append(1)
            else:
              final_list_v9.append(0)

            if lr_v10[j]>0.5:
              final_list_v10.append(1)
            else:
              final_list_v10.append(0)

            if lr_v12[j]>0.5:
              final_list_v12.append(1)
            else:
              final_list_v12.append(0)

            if lr_v23[j]>0.5:
              final_list_v23.append(1)
            else:
              final_list_v23.append(0)        

            if final_list_v3[-1]+final_list_v5[-1]+final_list_v7[-1]+final_list_v9[-1]+final_list_v10[-1]+final_list_v12[-1]+final_list_v23[-1]>=3:
              results.append(1)
              if j < horizon:
                final_list_1.append(1)
            else:
              results.append(0)
              if j < horizon:
                final_list_1.append(0)
          # calculate the error
            if y_va[j]!=final_list_v3[j]:
              result_v3_error.append(1)
            else:
              result_v3_error.append(0)

            if y_va[j]!=final_list_v5[j]:
              result_v5_error.append(1)
            else:
              result_v5_error.append(0)

            if y_va[j]!=final_list_v7[j]:
              result_v7_error.append(1)
            else:
              result_v7_error.append(0)

            if y_va[j]!=final_list_v9[j]:
              result_v9_error.append(1)
            else:
              result_v9_error.append(0)

            if y_va[j]!=final_list_v10[j]:
              result_v10_error.append(1)
            else:
              result_v10_error.append(0)

            if y_va[j]!=final_list_v12[j]:
              result_v12_error.append(1)
            else:
              result_v12_error.append(0)

            if y_va[j]!=final_list_v23[j]:
              result_v23_error.append(1)
            else:
              result_v23_error.append(0)
          
          print("the voting result is {}".format(metrics.accuracy_score(y_va, results)))
          
          window = 1
          for i in range(horizon,len(y_va)):
            
            error_lr_v3 = np.sum(result_v3_error[length:length+window])+1e-06
            error_lr_v5 = np.sum(result_v5_error[length:length+window])+1e-06
            error_lr_v7 = np.sum(result_v7_error[length:length+window])+1e-06
            error_lr_v9 = np.sum(result_v9_error[length:length+window])+1e-06
            error_lr_v10 = np.sum(result_v10_error[length:length+window])+1e-06
            error_lr_v12 = np.sum(result_v12_error[length:length+window])+1e-06
            error_lr_v23 = np.sum(result_v23_error[length:length+window])+1e-06	
            
            result = 0
            fenmu =1/error_lr_v3+ 1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v23
            weight_lr_v3 = float(1/error_lr_v3)/fenmu
            result+=weight_lr_v3*final_list_v3[i]
            weight_lr_v5 = float(1/error_lr_v5)/fenmu
            result+=weight_lr_v5*final_list_v5[i]
            weight_lr_v7 = float(1/error_lr_v7)/fenmu
            result+=weight_lr_v7*final_list_v7[i]
            weight_lr_v9 = float(1/error_lr_v9)/fenmu
            result+=weight_lr_v9*final_list_v9[i]
            weight_lr_v10 = float(1/error_lr_v10)/fenmu
            result+=weight_lr_v10*final_list_v10[i]
            weight_lr_v12 = float(1/error_lr_v12)/fenmu
            result+=weight_lr_v12*final_list_v12[i]
            if result>0.5:
              final_list_1.append(1)
            else:
              final_list_1.append(0)
            
            if window==window_size:
              length+=1
            else:
              window+=1
            
            #print(length)
          print("the length of the y_test is {}".format(len(final_list_1)))
          print("the weight ensebmle for V5 V7 LR weight voting beta precision is {}".format(metrics.accuracy_score(y_va, final_list_1)))
          print("the horizon is {}".format(horizon))
          print("the window size is {}".format(window_size))
          #print("the beta is {}".format(beta))
          print("the metal is {}".format(ground_truth))
          print("the test date is {}".format(date))					
