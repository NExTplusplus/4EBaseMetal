import os
import sys
import pandas as pd
import numpy as np
from copy import copy
from model.logistic_regression import LogReg
import utils.general_functions as gn
from utils.data_preprocess_version_control import generate_version_params
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '4EBaseMetal')))


class Logistic_online():
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
                source):
        self.lag = lag
        self.horizon = horizon
        self.version = version
        self.gt = gt
        self.date = date
        self.source = source
    
    #this function tunes the model using grid search over a defined set of hyperparameter values
    def tune(self,max_iter):
        print("begin to tune")

        #identify the configuration file for data based on version
        self.path = gn.generate_config_path(self.version)

        #read the data from the 4E or NExT database with configuration file to determine columns to that are required
        time_series,LME_dates,config_length = gn.read_data_with_specified_columns(self.source,self.path,"2003-11-12")

        #generate list of list of dates for rolling window
        today = self.date
        length = 5
        if gn.even_version(self.version) and self.horizon > 5:
            length = 4
        start_time,end_time = gn.get_relevant_dates(today,length,"tune")
        split_dates = gn.rolling_half_year(start_time,end_time,length)

        #generate the version parameters (parameters that control the preprocess)
        version_params = generate_version_params(self.version)

        #prepare holder for results
        ans = {"C":[]}
        
        #loop over each half year
        for s, split_date in enumerate(split_dates):

            print("the train date is {}".format(split_date[1]))
            print("the test date is {}".format(split_date[2]))

            #toggle metal id
            metal_id = False
            ground_truth_list = [self.gt]
            if gn.even_version(self.version):
                metal_id = True
                ground_truth_list = ["LME_Co_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot","LME_Le_Spot"]

            #extract copy of data to process
            ts = copy(time_series.loc[split_date[0]:split_date[-1]])
            tvt_date = split_date[1:-1]

            #prepare data according to model type and version parameters
            final_X_tr, final_y_tr, final_X_va, final_y_va, val_dates, column_lag_list = gn.prepare_data(ts,LME_dates,self.horizon,ground_truth_list,self.lag,copy(tvt_date),version_params,metal_id_bool = metal_id)
                
            #generate hyperparameters instances
            if self.horizon <=5:
                if self.version == "v23":
                    C_list = [0.01,0.1,1.0,10.0,100.0,1000.0]
                else:
                    C_list = [0.001,0.01,0.1,1.0,10.0,100.0] 
            else:
                if self.version == "v24":
                    C_list = [0.1,1.0,10.0,100.0,1000.0,10000.0]
                else:
                    C_list = [1e-5,0.0001,0.001,0.01,0.1,1.0,10.0]

            #generate model results for each hyperparameter instance for each half year
            for C in C_list:
                if C not in ans['C']:
                    ans["C"].append(C)
                if split_date[1]+"_acc" not in ans.keys():
                    ans[split_date[2]+"_acc"] = []
                    ans[split_date[2]+"_length"] = []

                pure_LogReg = LogReg(parameters={})
                max_iter = max_iter
                parameters = {"penalty":"l2", "C":C, "solver":"lbfgs", "tol":1e-7,"max_iter":6*4*config_length*max_iter, "verbose" : 0,"warm_start": False, "n_jobs": -1}
                pure_LogReg.train(final_X_tr,final_y_tr.flatten(), parameters)
                acc = pure_LogReg.test(final_X_va,final_y_va.flatten())
                ans[split_date[2]+"_acc"].append(acc)
                ans[split_date[2]+"_length"].append(len(final_y_va.flatten()))

        ans = pd.DataFrame(ans)
        ave = None
        length = None

        #generate total average across all half years
        for col in ans.columns.values.tolist():
            if "_acc" in col:
                if ave is None:
                    ave = ans.loc[:,col]*ans.loc[:,col[:-3]+"length"]
                    length = ans.loc[:,col[:-3]+"length"]
                else:
                    ave = ave + ans.loc[:,col]*ans.loc[:,col[:-3]+"length"]
                    length = length + ans.loc[:,col[:-3]+"length"]
        ave = ave/length
        ans = pd.concat([ans,pd.DataFrame({"average": ave})],axis = 1)
        
        #store results in csv
        pd.DataFrame(ans).to_csv(os.path.join(os.getcwd(),'result','validation','logistic',\
                                                        "_".join(["log_reg",self.gt,self.version,str(self.lag),str(self.horizon)+".csv"])))

    #-------------------------------------------------------------------------------------------------------------------------------------#
    
    #this function generates the model instances
    def train(self,C=0.01,tol=1e-7,max_iter=100):
        print("begin to train")

        #identify the configuration file for data based on version
        self.path = gn.generate_config_path(self.version)

        #read the data from the 4E or NExT database with configuration file to determine columns to that are required
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
            version_params=generate_version_params(self.version)

            print("the train date is {}".format(split_dates[0]))
            print("the test date is {}".format(split_dates[1]))

            #toggle metal id
            metal_id = False
            ground_truth_list = [self.gt]
            if gn.even_version(self.version):
                metal_id = True
                ground_truth_list = ["LME_Co_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot","LME_Le_Spot"]

            #extract copy of data to process
            ts = copy(time_series.loc[start_time:split_dates[2]])

            

            #load data for use
            final_X_tr, final_y_tr, final_X_va, final_y_va, val_dates, column_lag_list = gn.prepare_data(ts,LME_dates,self.horizon,ground_truth_list,self.lag,copy(split_dates),version_params,metal_id_bool = metal_id)

            pure_LogReg = LogReg(parameters={})
            parameters = {"penalty":"l2", "C":C, "solver":"lbfgs", "tol":tol,"max_iter":6*4*config_length*max_iter, "verbose" : 0,"warm_start": False, "n_jobs": -1}
            pure_LogReg.train(final_X_tr,final_y_tr.flatten(), parameters)	
            pure_LogReg.save(self.version, self.gt, self.horizon, self.lag,evalidate_date)

    #-------------------------------------------------------------------------------------------------------------------------------------#
    
    #this function is used to generate prediction 
    def test(self):
        print("begin to test")

        pure_LogReg = LogReg(parameters={})

        #identify the configuration file for data based on version
        self.path = gn.generate_config_path(self.version)

        #read the data from the 4E or NExT database with configuration file to determine columns to that are required
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
                model = pure_LogReg.load(self.version, "LME_All_Spot", self.horizon, self.lag,evalidate_date)
            else:
                model = pure_LogReg.load(self.version, self.gt, self.horizon, self.lag,evalidate_date)

            #generate the version
            version_params=generate_version_params(self.version)

            metal_id = False
            if gn.even_version(self.version):
                metal_id = True

            #extract copy of data to process
            ts = copy(time_series.loc[start_time:split_dates[2]])

            #load data for use
            final_X_tr, final_y_tr, final_X_va, final_y_va,val_dates, column_lag_list = gn.prepare_data(ts,LME_dates,self.horizon,[self.gt],self.lag,copy(split_dates),version_params,metal_id_bool = metal_id,live = True)

            prob = model.predict(final_X_va)
            probability = model.predict_proba(final_X_va)
            np.savetxt(os.path.join("result","probability","logistic","_".join([self.gt+str(self.horizon),date,"lr",self.version,"probability.txt"])),probability)
            final_list = []
            piece_list = []
            for i,val_date in enumerate(val_dates):
                piece_list.append(val_date)
                piece_list.append(prob[i])
                final_list.append(piece_list)
                piece_list=[]
            final_dataframe = pd.DataFrame(prob, columns=['prediction'],index=val_dates)
            final_dataframe.to_csv(os.path.join("result","prediction","logistic","_".join([self.gt,date,str(self.horizon),self.version])+".csv"))
