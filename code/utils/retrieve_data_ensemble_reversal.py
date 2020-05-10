import pandas as pd
import numpy as np
from copy import deepcopy
path_list = sorted(['ensemble_v7_0.00.txt','ensemble_v7_0.10.txt','ensemble_v7_0.20.txt','ensemble_v7_0.30.txt','ensemble_v7_0.40.txt',
              'ensemble_v7_0.60.txt','ensemble_v7_0.70.txt','ensemble_v7_0.80.txt','ensemble_v7_0.90.txt','ensemble_v7_1.00.txt',
              'ensemble_v7_0.05.txt','ensemble_v7_0.15.txt','ensemble_v7_0.25.txt','ensemble_v7_0.35.txt',
              'ensemble_v7_0.65.txt','ensemble_v7_0.75.txt','ensemble_v7_0.85.txt','ensemble_v7_0.95.txt',
              'ensemble_v9_0.00.txt','ensemble_v9_0.10.txt','ensemble_v9_0.20.txt','ensemble_v9_0.30.txt','ensemble_v9_0.40.txt',
              'ensemble_v9_0.60.txt','ensemble_v9_0.70.txt','ensemble_v9_0.80.txt','ensemble_v9_0.90.txt','ensemble_v9_1.00.txt',
              'ensemble_v9_0.05.txt','ensemble_v9_0.15.txt','ensemble_v9_0.25.txt','ensemble_v9_0.35.txt',
              'ensemble_v9_0.65.txt','ensemble_v9_0.75.txt','ensemble_v9_0.85.txt','ensemble_v9_0.95.txt',
            ])
voting_weight_frame = pd.DataFrame(columns=['metal',
                                '2014-07-01',"length_1","reverse_1",
                                '2015-01-01',"length_2","reverse_2",
                                '2015-07-01',"length_3","reverse_3",
                                '2016-01-01',"length_4","reverse_4",
                                '2016-07-01',"length_5","reverse_5",
                                '2017-01-01',"length_6","reverse_6",
                                '2017-07-01',"length_7","reverse_7",
                                '2018-01-01',"length_8","reverse_8",
                                '2018-07-01',"length_9","reverse_9",
                                'validation','test','reverse percentage','horizon','window_size','acc','version'])
for new_path in path_list:
  all_ensemble_result = []
  all_ensemble_voting_weight = []
  result_list = []
  ensemble_str = 'the voting result is'
  

  with open(new_path,"r") as f:
    lines = f.readlines()
    j=1
    k = 0
    voting_weight_result_list = [0]*35
    for i,line in enumerate(lines):
      if ensemble_str.lower() in line.lower():
          voting_weight_result_list[j]=float(lines[i+2].strip("\n").split(" ")[-1])
          voting_weight_result_list[j+1]=int(lines[i+1].strip("\n").split(" ")[-1])
          voting_weight_result_list[j+2]=float(lines[i+7].strip("\n").split(" ")[-1])
          voting_weight_result_list[-3]=float(lines[i+4].strip("\n").split(" ")[-1])
          voting_weight_result_list[-4]=float(lines[i+3].strip("\n").split(" ")[-1])
          voting_weight_result_list[0]=lines[i+5].strip("\n").split(" ")[-1]
          j+=3
      if j == 28:
        item = deepcopy(voting_weight_result_list)
        voting_weight_result_list[j] = float(item[1]*item[2]+item[4]*item[5]+item[7]*item[8]+item[10]*item[11]+item[13]*item[14])/float(item[2]+item[5]+item[8]+item[11]+item[14])
        voting_weight_result_list[j+1] = float(item[16]*item[17]+item[19]*item[20]+item[22]*item[23]+item[25]*item[26])/float(item[17]+item[20]+item[23]+item[26])
        voting_weight_result_list[j+2] = float(item[3]*item[2]+item[6]*item[5]+item[9]*item[8]+item[12]*item[11]+item[15]*item[14]+item[18]*item[17]+item[21]*item[20]+item[24]*item[23]+item[27]*item[26])/float(item[2]+item[5]+item[8]+item[11]+item[14]+item[17]+item[10]+item[23]+item[26])
        voting_weight_result_list[-1] = new_path.strip(".txt").split("_")[1]
        voting_weight_result_list[-2] = new_path.strip(".txt").split("_")[2]
        voting_weight_frame = pd.concat([voting_weight_frame,pd.DataFrame(data = [voting_weight_result_list], columns = ['metal',
                                              '2014-07-01',"length_1","reverse_1",
                                              '2015-01-01',"length_2","reverse_2",
                                              '2015-07-01',"length_3","reverse_3",
                                              '2016-01-01',"length_4","reverse_4",
                                              '2016-07-01',"length_5","reverse_5",
                                              '2017-01-01',"length_6","reverse_6",
                                              '2017-07-01',"length_7","reverse_7",
                                              '2018-01-01',"length_8","reverse_8",
                                              '2018-07-01',"length_9","reverse_9",
                                              'validation','test','reverse percentage','horizon','window_size','acc','version'])],axis = 0)
        j = 1
        k+=1
      if k == 6:
        temp = [0]*35
        temp[0] = "Total_Average"
        df = voting_weight_frame.iloc[-6:,-7:-2].mean(axis = 0)
        temp[-7] = df[0]
        temp[-6] = df[1]
        temp[-5] = df[2]
        temp[-4] = df[3]
        temp[-3] = df[4]
        temp[-2] = new_path.strip(".txt").split("_")[2]
        temp[-1] = new_path.strip(".txt").split("_")[1]
        voting_weight_frame = pd.concat([voting_weight_frame,pd.DataFrame(data = [temp], columns = ['metal',
                                      '2014-07-01',"length_1","reverse_1",
                                      '2015-01-01',"length_2","reverse_2",
                                      '2015-07-01',"length_3","reverse_3",
                                      '2016-01-01',"length_4","reverse_4",
                                      '2016-07-01',"length_5","reverse_5",
                                      '2017-01-01',"length_6","reverse_6",
                                      '2017-07-01',"length_7","reverse_7",
                                      '2018-01-01',"length_8","reverse_8",
                                      '2018-07-01',"length_9","reverse_9",
                                      'validation','test','reverse percentage','horizon','window_size','acc','version'])],axis = 0)
        k = 0




        
  voting_weight_frame = pd.concat([voting_weight_frame,pd.DataFrame(data = [[None]*35], columns = ['metal',
                              '2014-07-01',"length_1","reverse_1",
                              '2015-01-01',"length_2","reverse_2",
                              '2015-07-01',"length_3","reverse_3",
                              '2016-01-01',"length_4","reverse_4",
                              '2016-07-01',"length_5","reverse_5",
                              '2017-01-01',"length_6","reverse_6",
                              '2017-07-01',"length_7","reverse_7",
                              '2018-01-01',"length_8","reverse_8",
                              '2018-07-01',"length_9","reverse_9",
                              'validation','test','reverse percentage','horizon','window_size','acc','version'])],axis = 0)
  # voting_weight_frame = voting_weight_frame.sort_values(by=['horizon','window_size','metal'],ascending=(True, True,False))
voting_weight_frame.to_csv("voting_weight.csv",index=False)


