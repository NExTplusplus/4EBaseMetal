import os
import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from utils.general_functions import *
from utils.construct_data import rolling_half_year
from utils.version_control_functions import generate_version_params

"""
there are two functions below,
the test_one is for the v5 v7 v9 whose features are for the specific metal
the test_two is for the v10 v12 whose six metals features are reunited together 
"""
def test():
    for horizon in [1,3,5,10,20,60]:
        for version in ['v7','v9','v10','v12','v16','v23','v24','v26','v28','v30']:
            if version in ['v9','v12'] and horizon > 5:
                continue
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
                if s == 0:
                    continue
        
                print("the train date is {}".format(split_date[0]))
                print("the test date is {}".format(split_date[1]))
                for gt in ["LME_Co_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot","LME_Le_Spot"]:
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
                    true_data_x_tr = np.loadtxt(os.path.join(sys.path[0],'code','feature_pytest','lagged_data','lagged_data_X_tr_'+gt+str(horizon)+"_l"+str(lag)+"_"+version+"_"+split_date[1]+".txt"))
                    true_data_y_tr = np.loadtxt(os.path.join(sys.path[0],'code','feature_pytest','lagged_data','lagged_data_y_tr_'+gt+str(horizon)+"_l"+str(lag)+"_"+version+"_"+split_date[1]+".txt"))
                    true_data_x_va = np.loadtxt(os.path.join(sys.path[0],'code','feature_pytest','lagged_data','lagged_data_X_va_'+gt+str(horizon)+"_l"+str(lag)+"_"+version+"_"+split_date[1]+".txt"))
                    true_data_y_va = np.loadtxt(os.path.join(sys.path[0],'code','feature_pytest','lagged_data','lagged_data_y_va_'+gt+str(horizon)+"_l"+str(lag)+"_"+version+"_"+split_date[1]+".txt"))
                    ts = copy(time_series.loc[split_dates[s][0]:split_dates[s+2][2]])

                    #load data for use
                    final_X_tr, final_y_tr, final_X_va, final_y_va, val_dates, column_lag_list = prepare_data(ts,LME_dates,horizon,ground_truth_list,lag,copy(split_date),version_params,metal_id_bool = metal_id)



                    """
                    after loading the feature, we now check whether the accurate feature is same with the feature from the load data
                    """
                    print("version is {}".format(version))
                    print("the split date is {}".format(split_date))
                    print("the horizon is {}".format(horizon))
                    print("the metal is {}".format(gt))
                    #print(X_tr == true_data_x_tr)
                    np.savetxt('true.csv',true_data_x_tr,delimiter = ',')
                    np.savetxt('test.csv',final_X_tr-true_data_x_tr,delimiter = ',')
                    assert ((final_X_tr == true_data_x_tr).all())
                    assert ((final_y_tr.flatten() == true_data_y_tr).all())
                    assert ((final_X_va == true_data_x_va).all())
                    assert ((final_y_va.flatten() == true_data_y_va).all())

                    if gt == "ALL":
                        break
