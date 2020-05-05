import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from utils.construct_data import rolling_half_year
from utils.general_functions import *
import warnings
from utils.version_control_functions import generate_version_params

if __name__ == '__main__':
    desc = 'the ensemble model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s','--steps',type=str,default="1,3,5,10,20,60",
                        help='steps in the future to be predicted')
    parser.add_argument('-model','--model', help='which single model we want to ensemble', type = str, default = 'alstm')
    parser.add_argument('-d', '--dates', help = "the date is the final data's date which you want to use for testing",type=str)
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                                            type=str, default="LME_Al_Spot,LME_Co_Spot,LME_Le_Spot,LME_Ni_Spot,LME_Ti_Spot,LME_Zi_Spot")
    parser.add_argument(
                '-v','--version', type = str,
                help='which version model is to be deleted',
                default="")
    args = parser.parse_args()
    args.steps = [int(x) for x in args.steps.split(",")]
    args.ground_truth = args.ground_truth.split(",")
    
    for horizon in args.steps:
        for version in args.version.split(','):
            if version in ['v3','v23','v24','v28','v30']:
                config = 'exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf'
            elif version in ['v5','v7']:
                config = 'exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
            else:
                config = 'exp/online_v10.conf'

            assert_version(version,config)
        
            time_series,LME_dates,config_length = read_data_with_specified_columns('NExT',config,"2003-11-12")

            today = '2017-06-30'
            length = 5
            if even_version(version) and horizon > 5:
                length = 4
            start_time,end_time = get_relevant_dates(today,length,"tune")
            split_dates = rolling_half_year(start_time,end_time,length)
            split_dates  =  split_dates[:]
            
            #generate the version
            version_params=generate_version_params(version)
            
            for s, split_date in enumerate(split_dates[1:-1]):
        
                print("the train date is {}".format(split_date[0]))
                print("the test date is {}".format(split_date[1]))
                for gt in args.ground_truth:
                    #toggle metal id
                    metal_id = False
                    ground_truth_list = [gt]
                    lag = 5
                    if even_version(version) and version not in ['v16','v26']:
                        metal_id = True
                        gt = "ALL"
                        ground_truth_list = ["LME_Co_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot","LME_Le_Spot"]
                    if version in ['v16','v26']:
                        metal_id = True
                        spot_list = copy(np.array(time_series[gt]))
                        time_series['spot_price'] = spot_list
                    if int(version[1:]) > 20:
                        lag = 1
                    #extract copy of data to process
                    ts = copy(time_series.loc[split_dates[s][0]:split_dates[s+2][2]])

                    #load data for use
                    final_X_tr, final_y_tr, final_X_va, final_y_va, val_dates, column_lag_list = prepare_data(ts,LME_dates,horizon,ground_truth_list,lag,copy(split_date),version_params,metal_id_bool = metal_id)

                    np.savetxt('code/feature_pytest/lagged_data/'+'_'.join(['lagged_data_X_tr',gt+str(horizon),'l'+str(lag),version,split_date[1]+'.txt']),final_X_tr) 
                    np.savetxt('code/feature_pytest/lagged_data/'+'_'.join(['lagged_data_y_tr',gt+str(horizon),'l'+str(lag),version,split_date[1]+'.txt']),final_y_tr)
                    np.savetxt('code/feature_pytest/lagged_data/'+'_'.join(['lagged_data_X_va',gt+str(horizon),'l'+str(lag),version,split_date[1]+'.txt']),final_X_va)
                    np.savetxt('code/feature_pytest/lagged_data/'+'_'.join(['lagged_data_y_va',gt+str(horizon),'l'+str(lag),version,split_date[1]+'.txt']),final_y_va)
                    if gt == "ALL":
                        break
