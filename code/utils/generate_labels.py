import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from utils.general_functions import rolling_half_year
from utils.read_data import m2ar
from utils.general_functions import read_data_with_specified_columns

if __name__ == '__main__':
    desc = 'script to generate labels'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s','--horizon',type=str,default='1,3,5,10,20,60',
                        help='horizon in the future to be predicted')
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                        type=str, default="LME_Al_Spot,LME_Cu_Spot,LME_Pb_Spot,LME_Ni_Spot,LME_Xi_Spot,LME_Zn_Spot")
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "NExT"
    )
    args = parser.parse_args()
    if args.ground_truth =='None':
        args.ground_truth = None
    os.chdir(os.path.abspath(sys.path[0]))
    
    args.ground_truth = args.ground_truth.split(",")
    args.horizon = [int(i) for i in args.horizon.split(",")]

    #read data from specified source
    if args.source == "NExT":
        ts, dates, length = read_data_with_specified_columns("NExT","exp/LMCADY_v3.conf","2003-11-12")        
    else:
        start_date = "2003-11-12"
        import rpy2.robjects as robjects
        robjects.r('.sourceQlib()')    
        ts = robjects.r('''merge(getSecurity(c("LMCADY Comdty","LMAHDY Comdty","LMPBDY Comdty","LMZSDY Comdty","LMNIDY Comdty","LMSNDY Comdty"), start = "'''+start_date+'''"), 
                        getSecurityOHLCV(c("LMCADS03 Comdty","LMPBDS03 Comdty","LMNIDS03 Comdty","LMSNDS03 Comdty","LMZSDS03 Comdty","LMAHDS03 Comdty"), start = "'''+start_date+'''")
                        )
                    ''')
        ts.colnames = robjects.vectors.StrVector(["LME_Cu_Spot","LME_Al_Spot","LME_Pb_Spot","LME_Zn_Spot","LME_Ni_Spot","LME_Xi_Spot"
                        ,"LME_Cu_Open","LME_Cu_High","LME_Cu_Low","LME_Cu_Close","LME_Cu_Volume","LME_Cu_OI"
                        ,"LME_Pb_Open","LME_Pb_High","LME_Pb_Low","LME_Pb_Close","LME_Pb_Volume","LME_Pb_OI"
                        ,"LME_Ni_Open","LME_Ni_High","LME_Ni_Low","LME_Ni_Close","LME_Ni_Volume","LME_Ni_OI"
                        ,"LME_Xi_Open","LME_Xi_High","LME_Xi_Low","LME_Xi_Close","LME_Xi_Volume","LME_Xi_OI"
                        ,"LME_Zn_Open","LME_Zn_High","LME_Zn_Low","LME_Zn_Close","LME_Zn_Volume","LME_Zn_OI"
                        ,"LME_Al_Open","LME_Al_High","LME_Al_Low","LME_Al_Close","LME_Al_Volume","LME_Al_OI"
                        ])
        ts = m2ar(ts)
        os.chdir("NEXT/4EBaseMetal")
    print(ts)
    #iterate over ground truth
    for gt in args.ground_truth:

        #case if the source of data is csv files
        if args.source == "NExT":
            spot = copy(ts).loc[:,gt]
        else:
            spot = copy(ts[gt]).to_frame()

        #tentatively set the begining of label generation to 2014-01-01
        split_dates = rolling_half_year("2014-01-01",spot.index[-1],5)
        print(split_dates)

        #iterate over horizon
        for step in args.horizon:
            class_label = ((copy(spot.shift(-step)) - spot > 0)*1)
            reg_label = (copy(spot.shift(-step)))
            class_label.columns = ["Label"]
            reg_label.columns = ["Label"]
            class_label.dropna(inplace = True)
            reg_label.dropna(inplace = True)

            #for each half-year
            for split_date in split_dates:
                temp_class_label = class_label.loc[split_date[2]:split_date[3]]
                temp_reg_label = reg_label.loc[split_date[2]:split_date[3]]
                print(split_date,temp_class_label)
                temp_class_label = temp_class_label if temp_class_label.index.values[-1] != split_date[3] else temp_class_label[:-1]
                temp_reg_label = temp_reg_label if temp_reg_label.index.values[-1] != split_date[3] else temp_reg_label[:-1]
                temp_class_label.to_frame("Label").to_csv(os.path.join(os.getcwd(),'data','Label',"_".join([gt,'h'+str(step),split_date[2],"label.csv"])))
                temp_reg_label.to_frame("Label").to_csv(os.path.join(os.getcwd(),'data','Label',"_".join([gt,'h'+str(step),split_date[2],"reg_label.csv"])))

            #for the final half year that the last day is in
            temp_class_label = class_label.loc[split_date[3]:split_date[4]]
            temp_reg_label = reg_label.loc[split_date[3]:split_date[4]]
            print(split_date,temp_class_label)
            temp_class_label = temp_class_label if temp_class_label.index.values[-1] != split_date[3] else temp_class_label[:-1]
            temp_reg_label = temp_reg_label if temp_reg_label.index.values[-1] != split_date[3] else temp_reg_label[:-1]
            temp_class_label.to_frame("Label").to_csv(os.path.join(os.getcwd(),'data','Label',"_".join([gt,'h'+str(step),split_date[3],"label.csv"])))
            temp_reg_label.to_frame("Label").to_csv(os.path.join(os.getcwd(),'data','Label',"_".join([gt,'h'+str(step),split_date[3],"reg_label.csv"])))
            
            
            
                
                




    
