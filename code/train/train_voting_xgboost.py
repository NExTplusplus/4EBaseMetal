import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from sklearn import metrics
import math
import argparse


np.random.seed(0)
if __name__ == '__main__':
  desc = 'the ensemble model'
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument(
        '-v',nargs='+',
        help='which version model is to be deleted',
        default='none'
    )
  args = parser.parse_args()
  version = args.v
  
  print(version)
  for horizon in [1,3,5]:
    for ground_truth in ["LME_Co_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot","LME_Le_Spot"]:
        for window_size in [5,10,15,20,25,30]:
          for date in ["2014-07-01","2015-01-01","2015-07-01","2016-01-01","2016-07-01","2017-01-01","2017-07-01","2018-01-01","2018-07-01"]:
            version_dict = {}
            xgboost_v3 = np.loadtxt("data/xgboost_folder/"+ground_truth+"_"+"h"+str(horizon)+"_"+date+"_xgboost_v3.txt")
            version_dict['v3']=xgboost_v3
            xgboost_v5 = np.loadtxt("data/xgboost_folder/"+ground_truth+"_"+"h"+str(horizon)+"_"+date+"_xgboost_v5.txt")
            version_dict['v5']=xgboost_v5
            xgboost_v7 = np.loadtxt("data/xgboost_folder/"+ground_truth+"_"+"h"+str(horizon)+"_"+date+"_xgboost_v7.txt")
            version_dict['v7']=xgboost_v7
            xgboost_v9 = np.loadtxt("data/xgboost_folder/"+ground_truth+"_"+"h"+str(horizon)+"_"+date+"_xgboost_v9.txt")
            version_dict['v9']=xgboost_v9
            xgboost_v10 = np.loadtxt("data/xgboost_folder/"+ground_truth+"_"+"h"+str(horizon)+"_"+date+"_xgboost_v10.txt")
            version_dict['v10']=xgboost_v10
            xgboost_v12 = np.loadtxt("data/xgboost_folder/"+ground_truth+"_"+"h"+str(horizon)+"_"+date+"_xgboost_v12.txt")
            version_dict['v12']=xgboost_v12
            xgboost_v23 = np.loadtxt("data/xgboost_folder/"+ground_truth+"_"+"h"+str(horizon)+"_"+date+"_xgboost_v23.txt")
            version_dict['v23']=xgboost_v23
            xgboost_v24 = np.loadtxt("data/xgboost_folder/"+ground_truth+"_"+"h"+str(horizon)+"_"+date+"_xgboost_v24.txt")
            version_dict['v24']=xgboost_v24
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
            result_v24_error = []
            final_list_v3 = []
            final_list_v5 = []
            final_list_v7 = []
            final_list_v9 = []
            final_list_v10 = []
            final_list_v12 = []
            final_list_v23 = []
            final_list_v24 = []
            results = []
            final_list_1 = []
            df = pd.DataFrame()
            for j in range(len(xgboost_v23)):
              
              count_1 = 0
              count_0 = 0
              for item in xgboost_v3[j]:
                if item>0.5:
                  count_1+=1
                else:
                  count_0+=1
              if count_1>count_0:
                final_list_v3.append(1)
              else:
                final_list_v3.append(0)
                    
              count_1 = 0
              count_0 = 0
              for item in xgboost_v5[j]:
                if item>0.5:
                  count_1+=1
                else:
                  count_0+=1
              if count_1>count_0:
                final_list_v5.append(1)
              else:
                final_list_v5.append(0)
              
              count_1 = 0
              count_0 = 0
              for item in xgboost_v7[j]:
                if item>0.5:
                  count_1+=1
                else:
                  count_0+=1
              if count_1>count_0:
                final_list_v7.append(1)
              else:
                final_list_v7.append(0)
              
              count_1 = 0
              count_0 = 0
              for item in xgboost_v9[j]:
                if item>0.5:
                  count_1+=1
                else:
                  count_0+=1
              if count_1>count_0:
                final_list_v9.append(1)
              else:
                final_list_v9.append(0)
              
              count_1 = 0
              count_0 = 0
              for item in xgboost_v10[j]:
                if item>0.5:
                  count_1+=1
                else:
                  count_0+=1
              if count_1>count_0:
                final_list_v10.append(1)
              else:
                final_list_v10.append(0)
              
              count_1 = 0
              count_0 = 0
              for item in xgboost_v12[j]:
                if item>0.5:
                  count_1+=1
                else:
                  count_0+=1
              if count_1>count_0:
                final_list_v12.append(1)
              else:
                final_list_v12.append(0)
              
              count_1 = 0
              count_0 = 0
              for item in xgboost_v23[j]:
                if item>0.5:
                  count_1+=1
                else:
                  count_0+=1
              if count_1>count_0:
                final_list_v23.append(1)
              else:
                final_list_v23.append(0)
              
              count_1 = 0
              count_0 = 0
              for item in xgboost_v24[j]:
                if item>0.5:
                  count_1+=1
                else:
                  count_0+=1
              if count_1>count_0:
                final_list_v24.append(1)
              else:
                final_list_v24.append(0)

              for key in version_dict.keys():
                if key not in version:
                  count+=version_dict[key][-1]
              if count>=(len(version_dict.keys())/2)+1:
                results.append(1)
                if j < horizon:
                  final_list_1.append(1)
                  #print("done")
              else:
                results.append(0)
                if j < horizon:
                  final_list_1.append(0)
                  #print("done")s
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
              
              if y_va[j]!=final_list_v24[j]:
                result_v24_error.append(1)
              else:
                result_v24_error.append(0)
              
            print("the voting result is {}".format(metrics.accuracy_score(y_va, results)))
            
            window = 1
            for i in range(horizon,len(y_va)):
              error_dict = {}
              error_xgb_v3 = np.sum(result_v3_error[length:length+window])+1e-06
              error_dict['v3']=error_xgb_v3
              error_xgb_v5 = np.sum(result_v5_error[length:length+window])+1e-06
              error_dict['v5']=error_xgb_v5
              error_xgb_v7 = np.sum(result_v7_error[length:length+window])+1e-06
              error_dict['v7']=error_xgb_v7
              error_xgb_v9 = np.sum(result_v9_error[length:length+window])+1e-06
              error_dict['v9']=error_xgb_v9
              error_xgb_v10 = np.sum(result_v10_error[length:length+window])+1e-06
              error_dict['v10']=error_xgb_v10
              error_xgb_v12 = np.sum(result_v12_error[length:length+window])+1e-06
              error_dict['v12']=error_xgb_v12
              error_xgb_v23 = np.sum(result_v23_error[length:length+window])+1e-06
              error_dict['v23']=error_xgb_v23	
              error_xgb_v24 = np.sum(result_v24_error[length:length+window])+1e-06
              error_dict['v24']=error_xgb_v24
              result = 0
              for key in error_dict.keys():
                if key not in version:
                  fenmu += 1/error_dict[key]
              if 'v3' not in version:
                weight_xgb_v3 = float(1/error_xgb_v3)/fenmu
                result+=weight_xgb_v3*final_list_v3[i]
              if 'v5' not in version:
                weight_xgb_v5 = float(1/error_xgb_v5)/fenmu
                result+=weight_xgb_v5*final_list_v5[i]
              if 'v7' not in version:
                weight_xgb_v7 = float(1/error_xgb_v7)/fenmu
                result+=weight_xgb_v7*final_list_v7[i]
              if 'v9' not in version:
                weight_xgb_v9 = float(1/error_xgb_v9)/fenmu
                result+=weight_xgb_v9*final_list_v9[i]
              if 'v10' not in version:
                weight_xgb_v10 = float(1/error_xgb_v10)/fenmu
                result+=weight_xgb_v10*final_list_v10[i]
              if 'v12' not in version:
                weight_xgb_v12 = float(1/error_xgb_v12)/fenmu
                result+=weight_xgb_v12*final_list_v12[i]
              if 'v23' not in version:
                weight_xgb_v23 = float(1/error_xgb_v23)/fenmu
                result+=weight_xgb_v23*final_list_v23[i]
              if 'v24' not in version:
                weight_xgb_v24 = float(1/error_xgb_v24)/fenmu
                result+=weight_xgb_v24*final_list_v24[i]            
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
            print("the weight ensebmle for weight voting beta precision is {}".format(metrics.accuracy_score(y_va, final_list_1)))
            print("the horizon is {}".format(horizon))
            print("the window size is {}".format(window_size))
            #print("the beta is {}".format(beta))
            print("the metal is {}".format(ground_truth))
            print("the test date is {}".format(date))
