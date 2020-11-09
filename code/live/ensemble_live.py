import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from sklearn import metrics
import math
import argparse
import json
from copy import copy,deepcopy
from model.ensemble import Ensemble
import multiprocessing
from multiprocessing import Pool as pl
from itertools import combinations,product

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
                method = ['vote','vote'],
                window = [[0,0,0],[0]],
                feature_version = {"logistic":[],"xgboost":[],"alstm":[]}
                ):
        self.horizon = horizon
        self.gt = gt
        self.val_dates = dates.split('.')[0]
        self.method = method
        self.test_dates = dates.split('.')[1]
        self.window = window
        self.feature_version = feature_version
        
    #read predictions from a model type
    def read_prediction(self, model, feature_version, date):
        total_df = pd.DataFrame()
        for v in feature_version:
            #generate file name
            if model in ["logistic","xgboost"]:
                filepath = "result/prediction/"+model+"/"+self.gt+"_"+date+"_"+str(self.horizon)+"_"+v+".csv"
                print(filepath)
            else:
                filepath = "result/prediction/"+model+"/"+v+"/"+self.gt+"_"+date+"_"+str(self.horizon)+"_"+v.split("_")[0]+".csv"
            
            #read file if in folder
            if os.path.exists(os.path.abspath(filepath)):
            
                df = pd.read_csv(filepath,index_col = 0, names = [model+' '+v],header = 0)
                total_df = pd.concat([total_df,df],axis = 1)
        return total_df

    #generate prediction 
    def predict(self, date, feature_version_dc, method, full_label = None, window = None, uncertainty = False):

        #initialize values
        validation_date = date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" 
        total_df = []
        print(feature_version_dc)

        #load all included predictions
        for model, fv in feature_version_dc.items():
            print(model,fv)
            df = self.read_prediction(model, fv, date)
            total_df.append(df)
        print("number of models: "+str(len(total_df)))
        label = full_label[:total_df[0].shape[0]] if full_label is not None else None

        #non hierarchical 
        if len(method) == 1:
            total_df = pd.concat(total_df,axis = 1)
            ens = Ensemble(method[0])
            return ens.predict(total_df, label, window[1][0] if window is not None else None, self.horizon, uncertainty)
        
        #hierarchical
        elif len(method) == 2:
            layer_one_ens = []
            for i,df in enumerate(total_df):
                ens = Ensemble(method[0])
                df = ens.predict(df, label, window[0][i] if window is not None else None, self.horizon)
                layer_one_ens.append(df)
            layer_one_ens = pd.concat(layer_one_ens,axis = 1)
            ens = Ensemble(method[1])
            return ens.predict(layer_one_ens, label, window[1][0] if window is not None else None, self.horizon, uncertainty)

    #main tune function
    def tune(self, target):

        validation_dates = [date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" for date in self.val_dates.split(",")]

        ans = {"window":[],"model":[]}
        for date in validation_dates:
            ans[date+"_acc"] = []
            ans[date+"_len"] = []

        if target == "window":
            best_window = self.tune_sm(deepcopy(ans))
            self.window[0] = best_window
            self.tune_mm(deepcopy(ans), best_window).to_csv(os.path.join("result","validation","ensemble","_".join([self.gt,str(self.horizon)+".csv"])))
        
        elif target == "fv":
            self.tune_dm().to_csv(os.path.join("result","validation","ensemble","_".join([self.gt,str(self.horizon),"dm.csv"])))

    #tune for single model type (ie logistic regression, xgboost, alstm)
    def tune_sm(self, ans):

        validation_dates = [date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" for date in self.val_dates.split(",")]
        print("single model tuning")
        for method in ["vote","weight"]:
            for model, fv in self.feature_version[str(self.horizon)].items():
                for window in range(10,26,5):
                    ans["window"].append(window)
                    ans["model"].append(model+" "+method)
                    for i,date in enumerate(self.val_dates.split(",")):
#                         full_label = pd.DataFrame()
#                         full_pred = pd.DataFrame()
#                         set_gt = self.gt
#                         for gt in ["LME_Al_Spot","LME_Co_Spot","LME_Le_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot"]:
#                             self.gt = gt
#                             label = pd.read_csv("data/Label/"+gt+"_h"+str(self.horizon)+"_"+validation_dates[i]+"_label.csv",index_col = 0)
#                             predictions = self.predict(date, {model:fv}, [method], label, [None,[window]])
#                             label = label[:len(predictions)]
#                             full_label = pd.concat([full_label,label],axis = 0)
#                             full_pred = pd.concat([full_pred,predictions],axis = 0)
#                         self.gt = set_gt
#                         acc = metrics.accuracy_score(full_pred,full_label)
#                         length = len(full_pred)
                        label = pd.read_csv("data/Label/"+self.gt+"_h"+str(self.horizon)+"_"+validation_dates[i]+"_label.csv",index_col = 0)
                        predictions = self.predict(date, {model:fv}, [method], label, [None,[window]])
                        acc = metrics.accuracy_score(predictions,label[:len(predictions)])
                        length = len(predictions)
                        ans[validation_dates[i]+"_acc"].append(acc)
                        ans[validation_dates[i]+"_len"].append(length)
                    if method == "vote":
                        break

        df = pd.DataFrame(ans)
        average = np.zeros(len(df.index))
        length = np.zeros(len(df.index))
        for i, date in enumerate(self.val_dates.split(',')):
            average += df[validation_dates[i]+"_acc"]*df[validation_dates[i]+"_len"]
            length += df[validation_dates[i]+"_len"]
        df['average'] = average/length
        df = df.loc[["vote" not in name for name in df['model']],]
        df = df.sort_values(by = ['model','average','window'],ascending = [True,False, True])
        best_window = [int(df.iloc[4,0]),int(df.iloc[8,0]),int(df.iloc[0,0])] if len(df)> 8 else [int(df.iloc[4,0]),int(df.iloc[0,0]),0]
        return best_window


    #tune over multiple model types
    def tune_mm(self, ans, best_window):

        validation_dates = [date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" for date in self.val_dates.split(",")]
        print("multi model tuning")
        for method in [["vote"],["weight"],["vote","vote"],["vote","weight"],["weight","vote"],["weight","weight"]]:
            for window in range(10,26,5):
                ans["model"].append(','.join(method))
                ans["window"].append(','.join([str(i) for i in best_window])+','+str(window))
                for i,date in enumerate(self.val_dates.split(',')):
#                     full_label = pd.DataFrame()
#                     full_pred = pd.DataFrame()
#                     set_gt = self.gt
#                     for gt in ["LME_Al_Spot","LME_Co_Spot","LME_Le_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot"]:
#                         self.gt = gt
#                         label = pd.read_csv("data/Label/"+gt+"_h"+str(self.horizon)+"_"+validation_dates[i]+"_label.csv",index_col = 0)
#                         predictions = self.predict(date, self.feature_version[str(self.horizon)], method, label, [best_window,[window]])
#                         label = label[:len(predictions)]
#                         full_label = pd.concat([full_label,label],axis = 0)
#                         full_pred = pd.concat([full_pred,predictions],axis = 0)
#                     self.gt = set_gt
#                     acc = metrics.accuracy_score(full_pred,full_label)
#                     length = len(full_pred)
#                     ans[validation_dates[i]+"_acc"].append(acc)
#                     ans[validation_dates[i]+"_len"].append(length)
                    label = pd.read_csv("data/Label/"+self.gt+"_h"+str(self.horizon)+"_"+validation_dates[i]+"_label.csv",index_col = 0)
                    predictions = self.predict(date, self.feature_version[str(self.horizon)], method, label, [best_window,[window]])
                    acc = metrics.accuracy_score(predictions,label[:len(predictions)])
                    length = len(predictions)
                    ans[validation_dates[i]+"_acc"].append(acc)
                    ans[validation_dates[i]+"_len"].append(length)

        final_ans = pd.DataFrame(ans)
        average = np.zeros(len(final_ans.index))
        length = np.zeros(len(final_ans.index))
        for i, date in enumerate(self.val_dates.split(',')):
            average += final_ans[validation_dates[i]+"_acc"]*final_ans[validation_dates[i]+"_len"]
            length += final_ans[validation_dates[i]+"_len"]
        final_ans['average'] = average/length
        final_ans = final_ans.sort_values(by ="average", ascending = False)
        
        return final_ans
    
    #generate live prediction and uncertainty
    def test(self):
        dates  = self.val_dates + "," + self.test_dates if self.val_dates != "" else self.test_dates
        validation_dates = [date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01"  for date in dates.split(',')]

        for i, date in enumerate(dates.split(',')):
            #read validation for best window
            validation = pd.read_csv(os.path.join("result","validation","ensemble",self.gt+"_"+str(self.horizon)+".csv"), index_col = 0)
            method = validation.iloc[0,1].split(',')
            best_window = [int(i) for i in validation.iloc[0,0].split(',')]
            best_window = [best_window[:-1],[best_window[-1]]]
            label = pd.read_csv("data/Label/"+self.gt+"_h"+str(self.horizon)+"_"+validation_dates[i]+"_label.csv",index_col = 0)
            print(method,best_window)

            #generate predictions and uncertainties
            predictions,uncertainty = self.predict(date, self.feature_version[str(self.horizon)], method, label, best_window, uncertainty = True)
            predictions = pd.DataFrame(predictions,columns = ["result"])
            uncertainty = pd.DataFrame(uncertainty,columns = ["uncertainty"])
            predictions.to_csv(os.path.join("result","prediction","ensemble","_".join([self.gt,date,str(self.horizon),"ensemble.csv"])))
            uncertainty.to_csv(os.path.join("result","uncertainty","classification","_".join([self.gt,validation_dates[i],str(self.horizon),"ensemble.csv"])))


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                       No longer in use
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
    #tune for feature versions that should be included in final ensemble
    def tune_dm(self):

        #read all viable feature versions
        with open(os.path.join("exp","ensemble_tune_all.conf")) as f:
            fv_all = json.load(f)


        for horizon in fv_all.keys():
            for key in fv_all[horizon].keys():
                fv_all[horizon][key] = fv_all[horizon][key].split(',')
    
        assert fv_all == self.feature_version, "all feature version included"
        print("feature version selection")
        val_dates = self.val_dates
        self.val_dates = self.val_dates +','+ self.test_dates if self.test_dates != "" else self.val_dates
        validation_dates = [date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01"  for date in self.val_dates.split(',')]
        record = pd.DataFrame()
        max_acc = 0 
        final_dc = {}
        for horizon in [1,3,5,10,20,60]:
            self.feature_version = fv_all
            self.horizon = horizon
            while True:
                ans = pd.DataFrame()
                total_df = pd.DataFrame()
                for gt in "Al,Co,Le,Ni,Ti,Zi".split(','):
                    self.gt = "LME_"+gt+"_Spot"
                    name_list, df = self.delete_model()
                    total_df = pd.concat([total_df,df],axis = 0)

                for name in name_list: 
                    df = total_df.loc[total_df["version"]==name,]
                    for date in validation_dates:
                        df["final_average"] = df["final_average"] + df[date+"_acc"]*df[date+"_length"] if "final_average" in df.columns else df[date+"_acc"]*df[date+"_length"]
                        df["final_length"] = df["final_length"] + df[date+"_length"] if "final_length" in df.columns else df[date+"_length"]
                    df["final_average"] /= df["final_length"]
                    ans = pd.concat([ans,pd.DataFrame({"name":[name],"average":[np.average(df["final_average"])]})],axis = 0)

                ans.sort_values(by = "average", ascending = False, inplace = True)

                acc = ans.iloc[0,1]
                model = ans.iloc[0,0].split('_')[0]
                fv = '_'.join(ans.iloc[0,0].split('_')[1:])
                record = pd.concat([record,ans.iloc[0,:]],axis = 0)
                if ans.iloc[0,1] > max_acc:
                    max_acc = ans.iloc[0,1]
                    self.feature_version[str(self.horizon)][model].remove(fv)
                else:
                    print("break")
                    break

            dc = deepcopy(self.feature_version[str(self.horizon)])
            for key in dc.keys():
                dc[key] = ",".join(dc[key])
            final_dc[horizon] = dc
        with open(os.path.join("exp","ensemble_tune.conf"),"w") as out:
            json.dump(final_dc,out, indent = 4)

        return record
        

    #Implementation of deletion of feature version based on validation results
    def delete_model(self):

        validation_dates = [date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01"  for date in self.val_dates.split(',')]

        total_df = pd.DataFrame()
        name_list = []
        total_list = []
        
        fv_list = []


        ans = {'horizon':[],"ground_truth":[],"version":[]}
        for model, fv in self.feature_version[str(self.horizon)].items():
            for f in fv:
                dc = deepcopy(self.feature_version[str(self.horizon)])
                dc[model].remove(f)
                name_list.append(model+"_"+f)
                fv_list.append(dc)
        
        for col in name_list:
            ans['horizon'].append(self.horizon)
            ans['ground_truth'].append(self.gt)
            ans['version'].append(col)
            
        #loop through dates
        for i, date in enumerate(self.val_dates.split(',')):

            if validation_dates[i]+"_acc" not in ans.keys():
                ans[validation_dates[i]+"_acc"]=[]
                ans[validation_dates[i]+"_length"]= []

            #extract best results from validation
            validation = pd.read_csv(os.path.join("result","validation","ensemble",self.gt+"_"+str(self.horizon)+".csv"), index_col = 0)
            method = validation.iloc[0,1].split(',')
            best_window = validation.iloc[0,0].split(',')
            best_window = [[int(i) for i in best_window[:-1]], [int(best_window[-1])]]
            label = pd.read_csv("data/Label/"+self.gt+"_h"+str(self.horizon)+"_"+validation_dates[i]+"_label.csv",index_col = 0)

            p = pl(multiprocessing.cpu_count())
            pred = p.starmap(self.predict,product([date],fv_list,[method],[label],[best_window]))
            p.close()
            total_df = pd.concat(pred, axis = 1)
            label = label[:total_df.shape[0]]
            total_df.columns = name_list
            for col in total_df.columns:
                acc = metrics.accuracy_score(label, total_df[col])
                ans[validation_dates[i]+"_acc"].append(acc)
                ans[validation_dates[i]+"_length"].append(len(label))
        
        ans = pd.DataFrame(ans)
        return name_list, ans



    # def generate_uncertainty(self, date, versions_list, direct = False):
    #     validation_date = date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" 

    #     total_df = pd.DataFrame()
    #     for i,version in enumerate(versions_list.split(":")):

    #         if version == "":
    #             continue 
    #         if i == 0:
    #             model = "lr"
    #         elif i == 1:
    #             model = "xgboost"
    #         elif i == 2:
    #             model = "alstm"
    #         if not direct:
    #             versions = read_config(model,version,self.config)
    #         else:
    #             versions = version
    #         df = self.sm_predict(model, date, versions, self.window[i], "vote")
    #         total_df = pd.concat([total_df,df],axis = 1)
    #     ans = total_df.aggregate(np.sum,axis = 1)
    #     ans = (len(total_df.columns.values.tolist()) - ans) / len(total_df.columns.values.tolist())
    #     return ans