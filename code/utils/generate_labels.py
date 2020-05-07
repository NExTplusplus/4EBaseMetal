import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from utils.construct_data import rolling_half_year
from utils.read_data import m2ar
from utils.general_functions import read_data_with_specified_columns

if __name__ == '__main__':
    desc = 'label generation'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s','--steps',type=str,default='1,3,5,10,20,60',
                        help='steps in the future to be predicted')
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                        type=str, default="LME_Al_Spot,LME_Co_Spot,LME_Le_Spot,LME_Ni_Spot,LME_Ti_Spot,LME_Zi_Spot")
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "NExT"
    )
    args = parser.parse_args()
    if args.ground_truth =='None':
        args.ground_truth = None
    os.chdir(os.path.abspath(sys.path[0]))
    
    args.ground_truth = args.ground_truth.split(",")
    args.steps = [int(i) for i in args.steps.split(",")]

    if args.source == "NExT":
        ts, dates, length = read_data_with_specified_columns("NExT","exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf","2003-11-12")        
    else:
        import rpy2.robjects as robjects
        robjects.r('.sourceAlfunction()')    
        ts = robjects.r('''merge(getSecurity(c("LMCADY Comdty","LMAHDY Comdty","LMPBDY Comdty","LMZSDY Comdty","LMNIDY Comdty","LMSNDY Comdty"), start = "'''+start_date+'''"), 
                        getSecurityOHLCV(c("LMCADS03 Comdty","LMPBDS03 Comdty","LMNIDS03 Comdty","LMSNDS03 Comdty","LMZSDS03 Comdty","LMAHDS03 Comdty"), start = "'''+start_date+'''")
                        )
                    ''')
        ts.colnames = robjects.vectors.StrVector(["LME_Co_Spot","LME_Al_Spot","LME_Le_Spot","LME_Zi_Spot","LME_Ni_Spot","LME_Ti_Spot"
                        ,"LME_Co_Open","LME_Co_High","LME_Co_Low","LME_Co_Close","LME_Co_Volume","LME_Co_OI"
                        ,"LME_Le_Open","LME_Le_High","LME_Le_Low","LME_Le_Close","LME_Le_Volume","LME_Le_OI"
                        ,"LME_Ni_Open","LME_Ni_High","LME_Ni_Low","LME_Ni_Close","LME_Ni_Volume","LME_Ni_OI"
                        ,"LME_Ti_Open","LME_Ti_High","LME_Ti_Low","LME_Ti_Close","LME_Ti_Volume","LME_Ti_OI"
                        ,"LME_Zi_Open","LME_Zi_High","LME_Zi_Low","LME_Zi_Close","LME_Zi_Volume","LME_Zi_OI"
                        ,"LME_Al_Open","LME_Al_High","LME_Al_Low","LME_Al_Close","LME_Al_Volume","LME_Al_OI"
                        ])
        ts = m2ar(ts)

    for gt in args.ground_truth:
        if args.source == "NExT":
            spot = copy(ts).loc[:,gt]
        else:
            spot = copy(ts[gt]).to_frame()
        split_dates = rolling_half_year("2009-07-01",spot.index[-1],5)
        for step in args.steps:
            class_label = ((copy(spot.shift(-step)) - spot > 0)*1).to_frame()
            reg_label = (copy(spot.shift(-step))).to_frame()
            class_label.columns = ["Label"]
            reg_label.columns = ["Label"]
            class_label.dropna(inplace = True)
            reg_label.dropna(inplace = True)
            for split_date in split_dates:
                class_label.loc[split_date[1]:split_date[2]].to_csv(os.path.join(os.getcwd(),'data','Label',"_".join([gt,'h'+str(step),split_date[1],"label.csv"])))
                reg_label.loc[split_date[1]:split_date[2]].to_csv(os.path.join(os.getcwd(),'data','Label',"_".join([gt,'h'+str(step),split_date[1],"reg_label.csv"])))

                
                




    