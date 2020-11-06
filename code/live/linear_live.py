import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
import warnings
import utils.general_functions as gn
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from utils.data_preprocess_version_control import generate_version_params
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
    #this function is used to train the model and save it
    def train(self):
        print("begin to train")

        #assert that the configuration path is correct
        self.path = gn.generate_config_path(self.version)

        #read the data from the 4E or NExT database
        time_series,LME_dates,config_length = gn.read_data_with_specified_columns(self.source,self.path,"2003-11-12")

        for date in self.date.split(","):
            #generate list of dates for today's model training period
            today = date
            length = 5
            if gn.even_version(self.version) and self.horizon > 5:
                length = 4
            start_time,train_time,evalidate_date = gn.get_relevant_dates(today,length,"train")
            split_dates  =  [train_time,evalidate_date,str(today)]

            #generate the version
            version_params = generate_version_params(self.version)

            print("the train date is {}".format(split_dates[0]))
            print("the test date is {}".format(split_dates[1]))

            #toggle metal id
            metal_id = False
            ground_truth_list = [self.gt]
            if gn.even_version(self.version):
                metal_id = True
                ground_truth_list = ["LME_Cu_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Xi_Spot","LME_Zn_Spot","LME_Pb_Spot"]

            #extract copy of data to process
            ts = copy(time_series.loc[start_time:split_dates[2]])

            #load data for use
            final_X_tr, final_y_tr, final_X_va, final_y_va, val_dates, column_lag_list = gn.prepare_data(ts,LME_dates,self.horizon,ground_truth_list,self.lag,copy(split_dates),version_params,metal_id_bool = metal_id)

            LR= LinearRegression(n_jobs = -1)
            LR.fit(final_X_tr,final_y_tr[:,0])
            if gn.even_version(self.version):
                joblib.dump(LR,os.path.join(os.getcwd(),'result','model','linear',self.version+"_ALL_"+str(self.horizon)+"_"+str(self.lag)+"_"+evalidate_date+'.pkl'))
            else:
                joblib.dump(LR,os.path.join(os.getcwd(),'result','model','linear',self.version+"_"+self.gt+"_"+str(self.horizon)+"_"+str(self.lag)+"_"+evalidate_date+'.pkl'))

    #-------------------------------------------------------------------------------------------------------------------------------------#

    #this function is used to predict the date
    def test(self):
        #split the date
        print("begin to test")
        #assert that the configuration path is correct
        self.path = gn.generate_config_path(self.version)

        #read the data from the 4E or NExT database
        time_series,LME_dates,config_length = gn.read_data_with_specified_columns(self.source,self.path,"2003-11-12")

        for date in self.date.split(","):
            #generate list of dates for today's model training period
            today = date
            length = 5
            if gn.even_version(self.version) and self.horizon > 5:
                length = 4
            start_time,train_time,evalidate_date = gn.get_relevant_dates(today,length,"test")
            split_dates  =  [train_time,evalidate_date,str(today)]
            
            if gn.even_version(self.version):
                model = joblib.load(os.path.join(os.getcwd(),'result','model','linear',self.version+"_ALL_"+str(self.horizon)+"_"+str(self.lag)+"_"+evalidate_date+'.pkl'))
            else:
                model = joblib.load(os.path.join(os.getcwd(),'result','model','linear',self.version+"_"+self.gt+"_"+str(self.horizon)+"_"+str(self.lag)+"_"+evalidate_date+'.pkl'))

            #generate the version
            version_params=generate_version_params(self.version)

            metal_id = False
            if gn.even_version(self.version):
                metal_id = True

            #extract copy of data to process
            ts = copy(time_series.loc[start_time:split_dates[2]])

            #load data for use
            final_X_tr, final_y_tr, final_X_va, final_y_va,val_dates, column_lag_list = gn.prepare_data(ts,LME_dates,self.horizon,[self.gt],self.lag,copy(split_dates),version_params,metal_id_bool = metal_id,live = True)

            prob = (1+model.predict(final_X_va))*final_y_va[:,1]
            final_list = []
            piece_list = []
            for i,val_date in enumerate(val_dates):
                piece_list.append(val_date)
                piece_list.append(prob[i])
                final_list.append(piece_list)
                piece_list=[]
            final_dataframe = pd.DataFrame(prob, columns=['prediction'],index=val_dates)
            final_dataframe.to_csv(os.path.join("result","prediction","linear",self.version,"_".join([self.gt,date,str(self.horizon),self.version])+".csv"))
