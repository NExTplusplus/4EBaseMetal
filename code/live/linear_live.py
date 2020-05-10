import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
from data.load_data import load_data
from utils.transform_data import flatten
from utils.construct_data import rolling_half_year
from utils.log_reg_functions import objective_function, loss_function
from utils.read_data import read_data_NExT
from utils.general_functions import *
import warnings
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from utils.version_control_functions import generate_version_params
from sklearn.externals import joblib
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '4EBaseMetal')))


class Linear_online():
    """
    lag: the window size of the data feature
    horizon: the time horizon of the predict target
    version: the version of the feature
    gt: the ground_truth metal name
    date: the last date of the prediction
    source: the data source
    """
    def __init__(self,
                lag,
                horizon,
                version,
                gt,
                date,
                source,
                path):
        self.lag = lag
        self.horizon = horizon
        self.version = version
        self.gt = gt
        self.date = date
        self.source = source
        self.path = path
    """
    this function is used to train the model and save it
    """
    def train(self):
        print("begin to train")

        #assert that the configuration path is correct
        assert_version(self.version,self.path)

        #read the data from the 4E or NExT database
        time_series,LME_dates,config_length = read_data_with_specified_columns(self.source,self.path,"2003-11-12")

        for date in self.date.split(","):
            #generate list of dates for today's model training period
            today = date
            length = 5
            if even_version(self.version) and self.horizon > 5:
                length = 4
            start_time,train_time,evalidate_date = get_relevant_dates(today,length,"train")
            split_dates  =  [train_time,evalidate_date,str(today)]

            #generate the version
            version_params=generate_version_params(self.version)

            print("the train date is {}".format(split_dates[0]))
            print("the test date is {}".format(split_dates[1]))

            #toggle metal id
            metal_id = False
            ground_truth_list = [self.gt]
            if even_version(self.version):
                metal_id = True
                ground_truth_list = ["LME_Co_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot","LME_Le_Spot"]

            #extract copy of data to process
            ts = copy(time_series.loc[start_time:split_dates[2]])

            

            #load data for use
            final_X_tr, final_y_tr, final_X_va, final_y_va, val_dates, column_lag_list = prepare_data(ts,LME_dates,self.horizon,ground_truth_list,self.lag,copy(split_dates),version_params,metal_id_bool = metal_id)

            LR= LinearRegression(n_jobs = -1)
            LR.fit(final_X_tr,final_y_tr[:,0])
            if even_version(self.version):
                joblib.dump(LR,os.path.join(os.getcwd(),'result','model','linear',self.version+"_ALL_"+str(self.horizon)+"_"+str(self.lag)+"_"+evalidate_date+'.pkl'))
            else:
                joblib.dump(LR,os.path.join(os.getcwd(),'result','model','linear',self.version+"_"+self.gt+"_"+str(self.horizon)+"_"+str(self.lag)+"_"+evalidate_date+'.pkl'))

    #-------------------------------------------------------------------------------------------------------------------------------------#
    """
    this function is used to predict the date
    """
    def test(self):
        """
        split the date
        """
        #os.chdir(os.path.abspath(sys.path[0]))
        print("begin to test")
        #assert that the configuration path is correct
        assert_version(self.version,self.path)

        #read the data from the 4E or NExT database
        time_series,LME_dates,config_length = read_data_with_specified_columns(self.source,self.path,"2003-11-12")

        for date in self.date.split(","):
            #generate list of dates for today's model training period
            today = date
            length = 5
            if even_version(self.version) and self.horizon > 5:
                length = 4
            start_time,train_time,evalidate_date = get_relevant_dates(today,length,"test")
            split_dates  =  [train_time,evalidate_date,str(today)]
            
            if even_version(self.version):
                model = joblib.load(os.path.join(os.getcwd(),'result','model','linear',self.version+"_ALL_"+str(self.horizon)+"_"+str(self.lag)+"_"+evalidate_date+'.pkl'))
            else:
                model = joblib.load(os.path.join(os.getcwd(),'result','model','linear',self.version+"_"+self.gt+"_"+str(self.horizon)+"_"+str(self.lag)+"_"+evalidate_date+'.pkl'))

            #generate the version
            version_params=generate_version_params(self.version)

            metal_id = False
            if even_version(self.version):
                metal_id = True

            #extract copy of data to process
            ts = copy(time_series.loc[start_time:split_dates[2]])

            

            #load data for use
            final_X_tr, final_y_tr, final_X_va, final_y_va,val_dates, column_lag_list = prepare_data(ts,LME_dates,self.horizon,[self.gt],self.lag,copy(split_dates),version_params,metal_id_bool = metal_id,live = True)

            prob = (1+model.predict(final_X_va))*final_y_va[:,1]
            final_list = []
            piece_list = []
            for i,val_date in enumerate(val_dates):
                piece_list.append(val_date)
                piece_list.append(prob[i])
                final_list.append(piece_list)
                piece_list=[]
            final_dataframe = pd.DataFrame(prob, columns=['result'],index=val_dates)
            final_dataframe.to_csv(os.path.join("result","prediction","linear","_".join([self.gt,date,str(self.horizon),self.version])+".csv"))
