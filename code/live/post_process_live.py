import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from sklearn import metrics
from copy import copy
import multiprocessing
from multiprocessing import Pool as pl
from utils.general_functions import read_data_with_specified_columns
from utils.post_process import read_regression, read_classification, read_substitution_analyst, generate_final_signal, generate_class_signal, generate_reg_signal,read_uncertainty
from model.post_process import Post_process, Post_process_substitution, Post_process_filter
from itertools import product
import json

class Post_process_live():
    def __init__(self,
                ground_truth,
                horizon,
                dates,
                model,
                version):
            self.ground_truth = ground_truth
            self.horizon = horizon
            self.dates = dates
            self.model = model
            self.version = version

    def tune(self, inputs):
        #initialize parameters
        class_dict = {'threshold':[],'acc':[],'coverage':[], 'total_len' :[]}
        reg_dict = {'threshold':[],'mae':[], 'coverage':[], 'total_len' :[]}
        spot_price = read_data_with_specified_columns(inputs['source'],'exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf','2003-11-12')[0].loc[:,self.ground_truth].to_frame()

        for date in self.dates.split(","):
            class_dict[date+"_acc"] = []
            class_dict[date+"_coverage"] = []
            class_dict[date+"_total_len"] = []

            # reg_dict[date+"_acc"] = []
            reg_dict[date+"_mae"] = []
            reg_dict[date+"_coverage"] = []
            reg_dict[date+"_total_len"] = []

        #begin tuning with looping of date
        for date in self.dates.split(','):
            validation_date = date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" 
            if self.model == "Filter":
                #classification tune
                class_thresh = [i + 0.01 for i in np.arange(0.51,step = float(0.05))]
                class_combination = product([self.ground_truth], [self.horizon], [date], [self.version[0]], class_thresh)
                
                #generate classification signals
                p = pl(multiprocessing.cpu_count())
                class_signal = p.starmap(generate_class_signal, class_combination)
                p.close()
                class_signal = pd.concat(class_signal, axis = 1)
                class_signal.columns = [class_thresh]

                #begin analysis of classification results
                for c, col in enumerate(class_signal.columns):
                    class_pred = (read_classification(self.ground_truth, self.horizon, date, self.version[0],"ensemble")*2 - 1).multiply(class_signal[col]*1, axis = 0)
                    class_pred = class_pred.loc[class_pred['result'] != 0]
                    class_label = pd.read_csv(os.path.join("data","Label",'_'.join([self.ground_truth,"h"+str(self.horizon),validation_date,"label.csv"])),index_col = 0)*2 - 1
                    class_label = class_label.loc[class_pred.index,:]
                    if col not in class_dict['threshold']:
                        class_dict['threshold'].append(col)
                        class_dict['acc'].append(0)
                        class_dict['coverage'].append(0)
                        class_dict['total_len'].append(0)
                    if len(class_pred.index) > 0:
                        class_dict[date + "_acc"].append(metrics.accuracy_score(class_pred,class_label))
                        class_dict['acc'][c] += metrics.accuracy_score(class_pred, class_label)*len(class_label.index)
                    else:
                        class_dict[date + "_acc"].append(0)
                        class_dict['acc'][c] += 0
                    class_dict[date +"_coverage"].append(len(class_label.index)/len(class_signal.index))
                    class_dict[date +"_total_len"].append(len(class_signal.index))
                    class_dict['coverage'][c] += len(class_label.index)
                    class_dict['total_len'][c] += len(class_signal.index)
                
                #regression tuning
                if self.horizon  <= 5:
                    reg_thresh = np.arange(1.01, step = 0.1)
                else:
                    reg_thresh = np.arange(0.51, step = 0.05)
                reg_thresh = np.arange(0.05,0.31, step = 0.025)
                reg_window = [60]
                reg_combination = product([spot_price], [self.ground_truth], [self.horizon], [date], [self.version[1]], reg_thresh, reg_window)
                p = pl(multiprocessing.cpu_count())
                
                #generate regression signals
                reg_signal = p.starmap(generate_reg_signal, reg_combination)
                p.close()
                reg_signal = pd.concat(reg_signal, axis = 1)
                reg_signal.columns = [reg_thresh]

                #begin analysis of regression results
                for c,col in enumerate(reg_signal.columns):
                    reg_pred = (read_regression(spot_price, self.ground_truth, self.horizon, date, self.version[1])).multiply(reg_signal[col]*1, axis = 0)
                    reg_pred = reg_pred.loc[reg_pred['Prediction'] != 0]
                    class_pred = np.sign(reg_pred)
                    reg_label = spot_price.shift(-self.horizon).loc[reg_pred.index,:]
                    class_label = pd.read_csv(os.path.join("data","Label",'_'.join([self.ground_truth,"h"+str(self.horizon),validation_date,"label.csv"])),index_col = 0)*2 - 1
                    class_label = class_label.loc[reg_pred.index,:]
                    spot = spot_price.loc[reg_pred.index,:]
                    if col not in reg_dict['threshold']:
                        reg_dict['threshold'].append(col)
                        # reg_dict['acc'].append(0)
                        reg_dict['mae'].append(0)
                        reg_dict['coverage'].append(0)
                        reg_dict['total_len'].append(0)
                    if len(reg_pred.index) > 0:
                        # reg_dict[date+"_acc"].append(metrics.accuracy_score(class_pred,class_label))
                        reg_dict[date + "_mae"].append(metrics.mean_absolute_error(reg_pred/np.array(spot),reg_label/np.array(spot)))
                        reg_dict['mae'][c] += metrics.mean_absolute_error(reg_pred/np.array(spot),reg_label/np.array(spot))*len(reg_label.index)
                        # reg_dict["acc"][c] += metrics.accuracy_score(class_pred,class_label)*len(reg_label.index)
                    else:
                        # reg_dict[date + "_acc"].append(0)
                        reg_dict[date + "_mae"].append(0)
                        reg_dict['mae'][c] += 0
                        # reg_dict['acc'][c] += 0
                    reg_dict[date +"_coverage"].append(len(reg_label.index)/len(reg_signal.index))
                    reg_dict[date+"_total_len"].append(len(reg_signal.index))
                    reg_dict['coverage'][c] += len(reg_label.index)
                    reg_dict['total_len'][c] += len(reg_signal.index)
                    print(reg_dict)
        
        #compute average
        for i in range(len(class_dict['threshold'])):
            class_dict['acc'][i] = class_dict['acc'][i]/class_dict['coverage'][i] if class_dict['coverage'][i] > 0 else 0
            class_dict['coverage'][i] = class_dict['coverage'][i]/class_dict['total_len'][i]

        for i in range(len(reg_dict['threshold'])):
            # reg_dict['acc'][i] = reg_dict['acc'][i]/reg_dict['coverage'][i] if reg_dict['coverage'][i] > 0 else 0
            reg_dict['mae'][i] = reg_dict['mae'][i]/reg_dict['coverage'][i] if reg_dict['coverage'][i] > 0 else 0
            reg_dict['coverage'][i] = reg_dict['coverage'][i]/reg_dict['total_len'][i]
        class_df = pd.DataFrame(class_dict)
        reg_df = pd.DataFrame(reg_dict)
        reg_df = reg_df.loc[reg_df["coverage"] != 0].reset_index(drop = True)

        #generate ranking 
        class_df['acc_rank'] = class_df['acc'].rank(method = 'min', ascending = False)
        class_df['coverage_rank'] = class_df['coverage'].rank(method = 'min', ascending = False)
        class_df['rank'] = (class_df['acc_rank'] + class_df['coverage_rank'])/2
        # reg_df['acc_rank'] = reg_df['acc'].rank(method = 'min', ascending = False)
        reg_df['mae_rank'] = reg_df['mae'].rank(method = 'min', ascending = True)
        reg_df['coverage_rank'] = reg_df['coverage'].rank(method = 'min', ascending = False)
        reg_df['rank'] = (reg_df['mae_rank'] + reg_df['coverage_rank'])/2

        return class_df,reg_df

    def test(self, inputs):
        spot_price = read_data_with_specified_columns(inputs['source'],'exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf','2003-11-12')[0].loc[:,self.ground_truth].to_frame()
        for date in self.dates.split(','):

            #generate model specific arguments
            if self.model is None:
                model = Post_process()

            elif self.model == "Substitution":
                X = { 'Prediction' : read_classification(self.ground_truth,self.horizon,date,self.version[0],"ensemble") }
                if inputs['substitution'] == "analyst":
                    validation_date = date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01"
                    
                    with open("exp/substitution.conf","r") as f:
                        config = json.load(f)

                    if self.ground_truth not in config.keys() or self.horizon not in config[self.ground_truth]:
                        model = Post_process()
                        X["Uncertainty"] = read_uncertainty(self.ground_truth,self.horizon,date,"ensemble","classification") 
                        prediction = model.predict(X)
                        X["Uncertainty"].to_csv(os.path.join("result","uncertainty","classification",'_'.join([self.ground_truth,validation_date,str(self.horizon),"substitution.csv"])))
                    else:
                        model = Post_process_substitution()
                        validation_date = date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" 
                        X["Substitute"] = read_substitution_analyst(self.ground_truth, self.horizon, date)
                        X["Uncertainty"] = read_uncertainty(self.ground_truth,self.horizon,date,"ensemble","classification")
                        prediction, uncertainty = model.predict(X)
                        uncertainty.to_csv(os.path.join("result","uncertainty","classification",'_'.join([self.ground_truth,validation_date,str(self.horizon),"substitution.csv"])))

            elif self.model == "Filter":
                X = { 'Prediction' : read_classification(self.ground_truth,self.horizon,date,self.version[0],"Substitution") }
                model = Post_process_filter()
                X["Prediction"] = read_regression(spot_price, self.ground_truth, self.horizon, date, self.version[1])
                X["Filter"] = generate_final_signal(spot_price,self.ground_truth, self.horizon, date, self.version[0], self.version[1], inputs["class_threshold"], inputs["reg_threshold"], inputs["reg_window"])
                print(X["Filter"])
                prediction = model.predict(X)
            prediction.to_csv(os.path.join('result','prediction','post_process',self.model,'_'.join([self.ground_truth,date,str(self.horizon),self.model+".csv"])))
