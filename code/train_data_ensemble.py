import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
import warnings
from live.ensemble_live_new import Ensemble_online
from sklearn import metrics

if __name__ == '__main__':
  desc = 'the ensemble model'
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('-s','--steps',type=int,default=5,
            help='steps in the future to be predicted')
  parser.add_argument('-model','--model', help='which single model we want to ensemble', type = str, default = 'alstm')
  parser.add_argument('-method','--method', help='the ensemble if we want to put out the single model ensemble result', type = str, default = 'single')
  parser.add_argument('-d', '--date', help = "the date is the final data's date which you want to use for testing",type=str)
  parser.add_argument(
        '-v',nargs='+',
        help='which version model is to be deleted',
        default=[]
    )
  args = parser.parse_args()
  if args.method=='single':
      for horizon in [1,3,5]:
        for ground_truth in ["LME_Co_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot","LME_Le_Spot"]:
            for window in [5,10,15,20,25,30]:
              for date in ["2014-07-01","2015-01-01","2015-07-01","2016-01-01","2016-07-01","2017-01-01","2017-07-01","2018-01-01","2018-07-01"]:
                y_va = pd.read_csv("data/Label/"+ground_truth+"_h"+str(horizon)+"_"+date+"_label"+".csv")
                label = list(y_va['Label'])
                ensemble=Ensemble_online(horizon=horizon,gt=ground_truth,date=date,single_window=window,label=label,delete_model = args.v)
                ensemble = ensemble.single_model(args.model)
                print("the length of the y_test is {}".format(len(ensemble)))
                print("the weight ensebmle for weight voting beta precision is {}".format(metrics.accuracy_score(label[:], ensemble)))
                print("the horizon is {}".format(horizon))
                print("the window size is {}".format(window))
                print("the metal is {}".format(ground_truth))
                print("the test date is {}".format(date))
  
  if args.method=='multi':
      for horizon in [1,3,5]:
        for ground_truth in ["LME_Co_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot","LME_Le_Spot"]:
            #for window in [5,10,15,20,25,30]:
              for date in ["2017-01-01","2017-07-01","2018-01-01","2018-07-01"]:
                label = pd.read_csv("data/Label/"+ground_truth+"_h"+str(horizon)+"_"+date+"_label"+".csv")
                label_value = list(label['Label'])
                date_list=list(label['Unnamed: 0'])
                #label = list(y_va['Label'])
                #date = list(y_va['date'])
                #path = os.getcwd()
                #print(path)
                new_date = "".join(date.split("-"))
                
                if ground_truth=="LME_Al_Spot":
                    indicator = pd.read_csv('data/indicator/Al_'+new_date+"_"+str(horizon)+".csv")
                elif ground_truth=="LME_Co_Spot":
                    indicator = pd.read_csv('data/indicator/Cu_'+new_date+"_"+str(horizon)+".csv")
                elif ground_truth=="LME_Ni_Spot":
                    indicator = pd.read_csv('data/indicator/Ni_'+new_date+"_"+str(horizon)+".csv")
                elif ground_truth=="LME_Le_Spot":
                    indicator = pd.read_csv('data/indicator/Pb_'+new_date+"_"+str(horizon)+".csv")
                elif ground_truth=="LME_Ti_Spot":
                    indicator = pd.read_csv('data/indicator/Xi_'+new_date+"_"+str(horizon)+".csv")
                else:
                    indicator = pd.read_csv('data/indicator/Zn_'+new_date+"_"+str(horizon)+".csv")
                indicator_prediction = indicator[['date','discrete_score']][indicator['discrete_score']!=0.0]
                
                #print(indicator_prediction)
                #print(indicator_prediction[indicator_prediction['date']=='2017-01-04']['discrete_score'].values[0])
                indicator_list = list(indicator_prediction['date'])
                if horizon == 1:
                  ensemble=Ensemble_online(horizon=horizon,gt=ground_truth,date=date,single_window=15,label=copy(label_value), delete_model=['v16_loss'])
                  alstm_ensemble = ensemble.single_model('alstm')
                  ensemble=Ensemble_online(horizon=horizon,gt=ground_truth,date=date,single_window=15,label=copy(label_value))
                  xgb_ensemble = ensemble.single_model('xgb')
                  ensemble=Ensemble_online(horizon=horizon,gt=ground_truth,date=date,single_window=20,label=copy(label_value))
                  lr_ensemble = ensemble.single_model('lr')
                elif horizon == 3:
                  ensemble=Ensemble_online(horizon=horizon,gt=ground_truth,date=date,single_window=25,label=copy(label_value), delete_model=['v16_loss'])
                  alstm_ensemble = ensemble.single_model('alstm')
                  ensemble=Ensemble_online(horizon=horizon,gt=ground_truth,date=date,single_window=25,label=copy(label_value))
                  xgb_ensemble = ensemble.single_model('xgb')
                  ensemble=Ensemble_online(horizon=horizon,gt=ground_truth,date=date,single_window=20,label=copy(label_value))
                  lr_ensemble = ensemble.single_model('lr')
                elif horizon == 5:
                  ensemble=Ensemble_online(horizon=horizon,gt=ground_truth,date=date,single_window=15,label=copy(label_value), delete_model=['v16_loss'])
                  alstm_ensemble = ensemble.single_model('alstm')
                  ensemble=Ensemble_online(horizon=horizon,gt=ground_truth,date=date,single_window=15,label=copy(label_value))
                  xgb_ensemble = ensemble.single_model('xgb')
                  ensemble=Ensemble_online(horizon=horizon,gt=ground_truth,date=date,single_window=25,label=copy(label_value))
                  lr_ensemble = ensemble.single_model('lr')
                final_list = []
                for i, label_date in enumerate(date_list):
                  if label_date not in indicator_list:
                    if alstm_ensemble[i]+xgb_ensemble[i]+lr_ensemble[i]>=2:
                      final_list.append(1)
                    else:
                      final_list.append(0)
                #np.savetxt("/Users/changjiangeng/Desktop/4EBaseMetal/"+ground_truth+"_"+date+"_"+str(horizon)+"_"+"predict_label.txt",final_list)
                  else:
                    if indicator_prediction[indicator_prediction['date']==label_date]['discrete_score'].values[0]==-1:
                      if alstm_ensemble[i]+xgb_ensemble[i]+lr_ensemble[i]>=2:
                        final_list.append(1)
                      else:
                        final_list.append(0)
                    else:
                      if alstm_ensemble[i]+xgb_ensemble[i]+lr_ensemble[i]>=1:
                        final_list.append(1)
                      else:
                        final_list.append(0)
                np.savetxt("/Users/changjiangeng/Desktop/4EBaseMetal/"+ground_truth+"_"+date+"_"+str(horizon)+"_"+"predict_label.txt",final_list)
                print("the length of the y_test is {}".format(len(final_list)))
                print("the weight ensebmle for 4 models voting precision is {}".format(metrics.accuracy_score(label_value, final_list)))
                print("the horizon is {}".format(horizon))
                #print("the window size is {}".format(window))
                print("the metal is {}".format(ground_truth))
                print("the test date is {}".format(date))



    