# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:55:39 2020

@author: Kwoks
"""

import pandas as pd
import sys

def calcculate_accuracy_for_the_result(csv_path):
    
    try:
        target_df = pd.read_csv(csv_path)
    except:
        print('the path is wrong, please check')
        return
    
    name = csv_path.split('/')[-1]
    
    name_split = name.split('_')
    
    metal = name_split[0]
    
    start_date = name_split[1]
    
    end_date = name_split[2]
    
    threshold = name_split[3]
    
    horizon = name_split[4]
    
    without_zero_df = target_df[(target_df['discrete_{}d'.format(horizon)]!=0)&(target_df['discrete_score']!=0)]
    
    same_num = len(without_zero_df[without_zero_df['discrete_{}d'.format(horizon)]==without_zero_df['discrete_score']])
    
    total_num = len(without_zero_df)
    
    if total_num==0:
        acc = 0
    else:
        acc = same_num / total_num
        
    print('the accuracy for {} from {} to {} with threshold equals to {} is {}'.format(metal, start_date, end_date, threshold, acc))
    
if __name__ == '__main__':
    
    csv_path = sys.argv[1]
    
    calcculate_accuracy_for_the_result(csv_path)

