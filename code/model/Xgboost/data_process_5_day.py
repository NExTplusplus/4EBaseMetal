#! /usr/bin/env python
#-*- coding: UTF-8 -*-
import pandas as pd
import xgboost as xgb
import os
import sys
import math
import scipy.stats as sct

predict_result_data = pd.read_csv('Comex_data_with_technical_indicator.csv')
predict_result_data = predict_result_data.drop(columns=['Unnamed: 0'])
for column in predict_result_data.columns:
    if column!='LMCADY' and column!='Index':
        predict_result_data[column+'_one_day']=None
        predict_result_data[column+'_two_day']=None
        predict_result_data[column+'_three_day']=None
        predict_result_data[column+'_four_day']=None
        predict_result_data[column+'_five_day']=None
        for i in range(5, len(predict_result_data['Index'])):
            predict_result_data[column+'_one_day'][i]=predict_result_data[column][i-1]
            predict_result_data[column+'_two_day'][i]=predict_result_data[column][i-2]
            predict_result_data[column+'_three_day'][i]=predict_result_data[column][i-3]
            predict_result_data[column+'_four_day'][i]=predict_result_data[column][i-4]
            predict_result_data[column+'_five_day'][i]=predict_result_data[column][i-5]
        predict_result_data=predict_result_data.drop(columns=[column])
        print(column+" is done")
predict_result_data.to_csv('Comex_data_with_technical_indicator_window_5_days.csv')
print("done")
