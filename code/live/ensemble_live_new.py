import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from sklearn import metrics
import math
import argparse
import json
from copy import copy
import multiprocessing
from multiprocessing import Pool as pl
from itertools import combinations,product

def read_config(model, version, config):
  string = ""
  with open(os.path.join(os.getcwd(),config)) as f:
    config = json.load(f)
  if model.split("_")[0] == "ensemble":
    for j,v in enumerate(version.split(":")):
      if j == 0:
        m = "lr"
      elif j == 1:
        m = "xgboost"
      elif j == 2:
        m = "alstm"
      string += config[m][v]
      if j != 2:
        string+=":"
  else:
    string = config[model][version]
  return string

def voting(df):
    """
    df: dataframe which holds the predictions across all combinations
    """
    threshold = np.ceil(len(df.columns)/2)
    votes = df.sum(axis = 1)
    res = (votes >= threshold) * 1.0
    return res

def weight(df,label,window,horizon):
    error = df.ne(label.to_numpy(),axis = 0)*1.0
    cum_error = error.rolling(window = window,min_periods = 1).sum()
    cum_error = cum_error.shift(horizon).fillna(0) + 1e-6
    rec = (1.0/cum_error).sum(axis = 1)
    pred = (((1.0/cum_error).div(rec.to_numpy(),axis = 0) * df).sum(axis = 1) > 0.5)*1.0
    return pred
    
class Ensemble_online():
    """
    horizon: the time horizon of the predict target
    gt: the ground_truth metal name
    date: the last date of the prediction
    window: size for the single model
    """
    def __init__(self,
                horizon,
                gt,
                dates,
                window = "0:0:0",
                version = "",
                config = os.path.join('exp','ensemble_tune.conf'),
                hierarchical = True):
        self.horizon = horizon
        self.gt = gt
        self.dates = dates
        self.window = [int(x) for x in window.split(":")]
        self.version = version
        self.config = config
        self.hierarchical = hierarchical
        

    """
    this function is to ensemble the single model result
    """
    def sm_predict(self, model, date, version, window, method):
        """
        model: the single model that you want to use ensemble
        """
        validation_date = date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" 
        total_df = pd.DataFrame() 
        df = pd.DataFrame()
        for v in version.split(","):
            if model in ['lr','xgboost']:
                filepath = "result/prediction/"+model+"/"+self.gt+"_"+date+"_"+str(self.horizon)+"_"+v+".csv"
            else:
                filepath = "result/prediction/"+model+"/"+v+"/"+self.gt+"_"+date+"_"+str(self.horizon)+"_"+v.split("_")[0]+".csv"
            if filepath.split('/')[-1] in os.listdir('/'.join(filepath.split('/')[:-1])):
                df = pd.read_csv(filepath,index_col = 0, names = [model+' '+v],header = 0)
                total_df = pd.concat([total_df,df],axis = 1)
        if self.hierarchical:
            if method == "vote":
                return voting(total_df)
            else:
                label = pd.read_csv("data/Label/"+self.gt+"_h"+str(self.horizon)+"_"+validation_date+"_label.csv",index_col = 0)[:len(total_df.index)]
                return weight(total_df,label,window,self.horizon)
        else:
            return total_df

    def choose_parameter(self):

        validation_dates = [date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" for date in self.dates.split(",")]
        
        ans = {"window":[],"model":[]}

        for date in validation_dates:
            ans[date+"_acc"] = []
            ans[date+"_len"] = []

        print("single model tuning")
        for i, version in enumerate(self.version.split(":")):

            if version == "":
                continue 
            
            if i == 0:
                model = "lr"
            elif i == 1:
                model = "xgboost"
            elif i == 2:
                model = "alstm"
            
            versions = read_config(model,version,self.config)

            for window in [10,15,20,25]:
                ans["window"].append(window)
                ans["model"].append(model)
                for i,date in enumerate(self.dates.split(",")):
                    weight = self.sm_predict(model, date, versions, window, 'weight')
                    label = pd.read_csv("data/Label/"+self.gt+"_h"+str(self.horizon)+"_"+validation_dates[i]+"_label.csv",index_col = 0)
                    acc = metrics.accuracy_score(weight,label[:len(weight)])
                    length = len(weight)
                    ans[validation_dates[i]+"_acc"].append(acc)
                    ans[validation_dates[i]+"_len"].append(length)
        df = pd.DataFrame(ans)
        average = np.zeros(len(df.index))
        length = np.zeros(len(df.index))
        for i, date in enumerate(self.dates.split(',')):
            average += df[validation_dates[i]+"_acc"]*df[validation_dates[i]+"_len"]
            length += df[validation_dates[i]+"_len"]
        df['average'] = average/length
        df = df.sort_values(by = ['model','average','window'],ascending = [True,False, True])
        self.window = [int(df.iloc[4,0]),int(df.iloc[8,0]),int(df.iloc[0,0]),0]
        print("ensemble tuning")
        for model in ['hensemble','nhensemble']:
            if model == 'hensemble':
                self.hierarchical = True
            if model == 'nhensemble':
                self.hierarchical = False
            for window in [10,15,20,25]:
                self.window[3] = window
                ans['window'].append(window)
                ans['window'].append(window)
                ans['model'].append('vote_'+model)
                ans['model'].append('weight_'+model)
                for i, date in enumerate(self.dates.split(',')):
                    vote = self.predict(date, self.version, 'vote:vote:vote', 'weight')
                    weight = self.predict(date, self.version, 'weight:weight:weight','weight')
                    label = pd.read_csv("data/Label/"+self.gt+"_h"+str(self.horizon)+"_"+validation_dates[i]+"_label.csv",index_col = 0)
                    vote_acc = metrics.accuracy_score(vote, label[:len(weight)])
                    weight_acc = metrics.accuracy_score(weight, label[:len(weight)])
                    length = len(vote)
                    ans[validation_dates[i]+"_acc"].append(vote_acc)
                    ans[validation_dates[i]+"_len"].append(length)
                    ans[validation_dates[i]+"_acc"].append(weight_acc)
                    ans[validation_dates[i]+"_len"].append(length)
        final_ans = pd.DataFrame(ans)
        average = np.zeros(len(final_ans.index))
        length = np.zeros(len(final_ans.index))
        for i, date in enumerate(self.dates.split(',')):
            average += final_ans[validation_dates[i]+"_acc"]*final_ans[validation_dates[i]+"_len"]
            length += final_ans[validation_dates[i]+"_len"]
        final_ans['average'] = average/length
                
        return pd.DataFrame(final_ans)
    
    def predict(self, date, versions_list, sm_methods, ens_method, direct = False):
        '''
        '''
        validation_date = date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" 

        total_df = pd.DataFrame()
        sm_methods = sm_methods.split(':')
        for i,version in enumerate(versions_list.split(":")):

            if version == "":
                continue 
            
            if i == 0:
                model = "lr"
            elif i == 1:
                model = "xgboost"
            elif i == 2:
                model = "alstm"
            if not direct:
                versions = read_config(model,version,self.config)
            else:
                versions = version
            df = self.sm_predict(model, date, versions, self.window[i], sm_methods[i])
            total_df = pd.concat([total_df,df],axis = 1)
        if ens_method == "vote":
            return voting(total_df)
            
        else:
            label = pd.read_csv("data/Label/"+self.gt+"_h"+str(self.horizon)+"_"+validation_date+"_label.csv",index_col = 0)[:len(total_df.index)]
            return weight(total_df,label,self.window[3],self.horizon)

    def predict_in_deletion(self,version,date,sm_methods,ens_method):
        versions = ""
        for i,v in enumerate(version):
            versions += v.split(' ')[1]+","
            if i != len(version) - 1 and version[i+1][:2] != version[i][:2] :
                versions = versions[:-1]+":"
            if i == len(version) - 1:
                versions = versions[:-1]
        pred = self.predict(date,versions,sm_methods,ens_method,direct = True)
        return pred
    
    def delete_model(self, date, versions_list, sm_methods, ens_method, length):

        validation_date = date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" 
        total_df = pd.DataFrame()
        name_list = []
        total_list = []
        
        for i,version in enumerate(versions_list.split(":")):

            if version == "":
                continue 
            
            if i == 0:
                model = "lr"
            elif i == 1:
                model = "xgboost"
            elif i == 2:
                model = "alstm"
            vers = read_config(model,version,self.config)
            total_list = total_list+[model+" "+v for v in vers.split(',')]

        
        version_list = list(combinations(total_list,len(total_list) - length))
        p = pl(multiprocessing.cpu_count())
        pred = p.starmap(self.predict_in_deletion,product(version_list,[date],[sm_methods],[ens_method]))
        p.close()
        total_df = pd.concat(pred, axis = 1)
        name_list = [str(set(total_list) - set(v)) for v in version_list]
        total_df.columns = name_list
        return total_df
