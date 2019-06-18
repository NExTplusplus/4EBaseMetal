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
        predict_result_data[column+'_six_day']=None
        predict_result_data[column+'_seven_day']=None
        predict_result_data[column+'_eight_day']=None
        predict_result_data[column+'_nine_day']=None
        predict_result_data[column+'_ten_day']=None
        for i in range(10, len(predict_result_data['Index'])):
            predict_result_data[column+'_six_day'][i]=predict_result_data[column][i-6]
            predict_result_data[column+'_seven_day'][i]=predict_result_data[column][i-7]
            predict_result_data[column+'_eight_day'][i]=predict_result_data[column][i-8]
            predict_result_data[column+'_nine_day'][i]=predict_result_data[column][i-9]
            predict_result_data[column+'_ten_day'][i]=predict_result_data[column][i-10]
        predict_result_data=predict_result_data.drop(columns=[column])
        print(column+" is done")
predict_result_data.to_csv('Comex_data_with_technical_indicator_window_5_10_days.csv')
print("done")
