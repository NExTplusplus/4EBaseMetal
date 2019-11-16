import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from data.load_data import load_data
from model.logistic_regression import LogReg
from utils.transform_data import flatten
from utils.construct_data import rolling_half_year
from utils.log_reg_functions import objective_function, loss_function
import warnings
import xgboost as xgb
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn import metrics
from sklearn.model_selection import KFold
from utils.version_control_functions import generate_version_params
from sklearn.externals import joblib


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
	"""
	this function is used to choose the parameter
	"""
	def choose_parameter(self,max_iter):
		if self.version in ['v5','v7','v9']:
			print("begin to choose the parameter")
			"""
			read the data from the 4E database
			"""
			#os.chdir(os.path.abspath(sys.path[0]))
			if self.source=='4E':
				from utils.read_data import read_data_v5_4E
				time_series, LME_dates = read_data_v5_4E("2003-11-12")
			elif self.source=='NExT':
				if self.version=='v5' or self.version=='v7':
					path = 'exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
					#if '4EBaseMetal' not in sys.path[0]:
					#	path = '4EBaseMetal/exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
					with open(os.path.join(os.getcwd(),path)) as fin:
						fname_columns = json.load(fin)
					from utils.read_data import read_data_NExT
					data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
					time_series = pd.concat(data_list, axis = 1, sort = True)
				elif self.version=='v9' or self.version=='v10' or self.version=='v12':
					path = 'exp/online_v10.conf'
					#if '4EBaseMetal' not in sys.path[0]:
					#	path = '4EBaseMetal/exp/online_v10.conf'
					with open(os.path.join(os.getcwd(),path)) as fin:
						fname_columns = json.load(fin)
					from utils.read_data import read_data_NExT
					data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
					time_series = pd.concat(data_list, axis = 1, sort = True)
			"""
			begin to split the train data
			"""
			today = self.date
			month = str(today).split("-")[1]
			year = str(today).split("-")[0]
			if month <="06":
				if month=="01":
					day = int(str(today).split("-")[2])
					if day <=self.horizon:
						print("the data is not enough for training")
						os.exit(0)
					else:
						start_year = int(year)-10
						start_time = str(start_year)+"-07-01"
						end_time = str(today)
				else:
					start_year = int(year)-10
					start_time = str(start_year)+"-07-01"
					end_time = str(today)
			else:
				if month=="07":
					day = int(str(today).split("-")[2])
					if day<=self.horizon:
						print("the data is not enough for training")
						os.exit(0)
					else:
						year = str(today).split("-")[0]
						start_year = int(year)-9
						end_year = int(year)
						start_time = str(start_year)+"-01-01"
						end_time = str(end_year)+"-07-01"
				else:
					year = str(today).split("-")[0]
					start_year = int(year)-9
					end_year = int(year)
					start_time = str(start_year)+"-01-01"
					end_time = str(end_year)+"-07-01"

			length = 5
			split_dates = rolling_half_year(start_time,end_time,length)
			split_dates  =  split_dates[:]
			print(split_dates)
			"""
			generate the version
			"""			
			version_params=generate_version_params(self.version)
			ans = {"C":[]}
			for s, split_date in enumerate(split_dates[:-1]):
				print("the train date is {}".format(split_date[0]))
				print("the test date is {}".format(split_date[1]))
				norm_volume = "v1"
				norm_3m_spread = "v1"
				norm_ex = "v1"
				len_ma = 5
				len_update = 30
				tol = 1e-7
				norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
								'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
				tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
												'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
				ts = copy(time_series.loc[split_date[0]:split_dates[s+1][2]])
				"""
				load_data
				"""
				X_tr, y_tr, X_va, y_va, X_te, y_te, norm_params,column_list = load_data(ts,LME_dates,self.horizon,[self.gt],self.lag,copy(split_date),norm_params,tech_params,version_params)
				
				X_tr = np.concatenate(X_tr)
				X_tr = X_tr.reshape(len(X_tr),self.lag*len(column_list[0]))
				y_tr = np.concatenate(y_tr)
				X_va = np.concatenate(X_va)
				y_va = np.concatenate(y_va)
				X_va = X_va.reshape(len(X_va),self.lag*len(column_list[0]))
				"""
				tune logistic regression hyper parameter
				"""
				for C in [0.0001,0.001,0.01,0.1,1.0,10.0,100.0]:
					if C not in ans['C']:
						ans["C"].append(C)
					if split_date[1]+"_acc" not in ans.keys():
						ans[split_date[1]+"_acc"] = []
						ans[split_date[1]+"_length"] = []

					pure_LogReg = LogReg(parameters={})
					max_iter = max_iter
					parameters = {"penalty":"l2", "C":C, "solver":"lbfgs", "tol":tol,"max_iter":6*4*len(fname_columns[0])*max_iter, "verbose" : 0,"warm_start": False, "n_jobs": -1}
					pure_LogReg.train(X_tr,y_tr.flatten(), parameters)
					acc = pure_LogReg.test(X_va,y_va.flatten())
					ans[split_date[1]+"_acc"].append(acc)
					ans[split_date[1]+"_length"].append(len(y_va))
			ans = pd.DataFrame(ans)
			ave = None
			length = None
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
			ans.sort_values(by= "average",ascending = False)
			pd.DataFrame(ans).to_csv("_".join(["log_reg",self.gt,self.version,str(self.lag),str(self.horizon)+".csv"]))
		else:
			print("begin to choose the parameter")
			"""
			read the data from the 4E database
			"""
			if self.source=='4E':
				from utils.read_data import read_data_v5_4E
				time_series, LME_dates = read_data_v5_4E("2003-11-12")
			elif self.source=='NExT':
				if self.version=='v5' or self.version=='v7':
					path = 'exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
					#if '4EBaseMetal' not in sys.path[0]:
					#	path = '4EBaseMetal/exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
					with open(os.path.join(os.getcwd(),path)) as fin:
						fname_columns = json.load(fin)
					from utils.read_data import read_data_NExT
					data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
					time_series = pd.concat(data_list, axis = 1, sort = True)
				elif self.version=='v9' or self.version=='v10' or self.version=='v12':
					path = 'exp/online_v10.conf'
					#if '4EBaseMetal' not in sys.path[0]:
					#	path = '4EBaseMetal/exp/online_v10.conf'
					with open(os.path.join(os.getcwd(),path)) as fin:
						fname_columns = json.load(fin)
					from utils.read_data import read_data_NExT
					data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
					time_series = pd.concat(data_list, axis = 1, sort = True)
			"""
			begin to split the train data
			"""
			today = self.date
			month = str(today).split("-")[1]
			if month <="06":
				if month=="01":
					day = int(str(today).split("-")[2])
					if day <=self.horizon:
						print("the data is not enough for training")
						os.exit(0)
					else:
						start_year = int(year)-10
						start_time = str(start_year)+"-07-01"
						end_time = str(today)
				else:
					start_year = int(year)-10
					start_time = str(start_year)+"-07-01"
					end_time = str(today)
			else:
				if month=="07":
					day = int(str(today).split("-")[2])
					if day<=self.horizon:
						print("the data is not enough for training")
						os.exit(0)
					else:
						year = str(today).split("-")[0]
						start_year = int(year)-9
						end_year = int(year)
						start_time = str(start_year)+"-01-01"
						end_time = str(end_year)+"-07-01"
				else:
					year = str(today).split("-")[0]
					start_year = int(year)-9
					end_year = int(year)
					start_time = str(start_year)+"-01-01"
					end_time = str(end_year)+"-07-01"
			length = 5
			split_dates = rolling_half_year("2009-07-01","2017-07-01",length)
			split_dates  =  split_dates[:]
			importance_list = []
			"""
			generate the version
			"""			
			version_params=generate_version_params(args.version)
			ans = {"C":[]}
			for s, split_date in enumerate(split_dates[:-1]):
                
				#generate parameters for load data
				horizon = args.steps
				norm_volume = "v1"
				norm_3m_spread = "v1"
				norm_ex = "v1"
				len_ma = 5
				len_update = 30
				tol = 1e-7
				norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
								'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
				final_X_tr = []
				final_y_tr = []
				final_X_va = []
				final_y_va = []
				final_X_te = []
				final_y_te = [] 
				tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
												'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
				ts = copy(time_series.loc[split_date[0]:split_dates[s+1][2]])
				i = 0
				#iterate over ground truths
				for ground_truth in ['LME_Co_Spot','LME_Al_Spot','LME_Ni_Spot','LME_Ti_Spot','LME_Zi_Spot','LME_Le_Spot']:
					print(ground_truth)
					metal_id = [0,0,0,0,0,0]
					metal_id[i] = 1
					#load data
					X_tr, y_tr, X_va, y_va, X_te, y_te, norm_check,column_list = load_data(copy(ts),LME_dates,horizon,[ground_truth],lag,copy(split_date),norm_params,tech_params,version_params)
					#post load processing and metal id extension
					X_tr = np.concatenate(X_tr)
					X_tr = X_tr.reshape(len(X_tr),lag*len(column_list[0]))
					X_tr = np.append(X_tr,[metal_id]*len(X_tr),axis = 1)
					y_tr = np.concatenate(y_tr)
					X_va = np.concatenate(X_va)
					X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
					X_va = np.append(X_va,[metal_id]*len(X_va),axis = 1)
					y_va = np.concatenate(y_va)
					final_X_tr.append(X_tr)
					final_y_tr.append(y_tr)
					final_X_va.append(X_va)
					final_y_va.append(y_va)
					i+=1
				#sort by time, not by metal
				final_X_tr = [np.transpose(arr) for arr in np.dstack(final_X_tr)]
				final_y_tr = [np.transpose(arr) for arr in np.dstack(final_y_tr)]
				final_X_va = [np.transpose(arr) for arr in np.dstack(final_X_va)]
				final_y_va = [np.transpose(arr) for arr in np.dstack(final_y_va)]
				final_X_tr = np.reshape(final_X_tr,[np.shape(final_X_tr)[0]*np.shape(final_X_tr)[1],np.shape(final_X_tr)[2]])
				final_y_tr = np.reshape(final_y_tr,[np.shape(final_y_tr)[0]*np.shape(final_y_tr)[1],np.shape(final_y_tr)[2]])
				final_X_va = np.reshape(final_X_va,[np.shape(final_X_va)[0]*np.shape(final_X_va)[1],np.shape(final_X_va)[2]])
				final_y_va = np.reshape(final_y_va,[np.shape(final_y_va)[0]*np.shape(final_y_va)[1],np.shape(final_y_va)[2]])
				#tune logistic regression hyper parameter
				for C in [0.0001,0.001,0.01,0.1,1.0,10,100]:
					if C not in ans['C']:
						ans["C"].append(C)
					if split_date[1]+"_acc" not in ans.keys():
						ans[split_date[1]+"_acc"] = []
						ans[split_date[1]+"_length"] = []
					n+=1
					pure_LogReg = LogReg(parameters={})
					max_iter = max_iter
					parameters = {"penalty":"l2", "C":C, "solver":"lbfgs", "tol":tol,"max_iter":6*4*len(fname_columns[0])*max_iter, "verbose" : 0,"warm_start": False, "n_jobs": -1}
					pure_LogReg.train(final_X_tr,final_y_tr.flatten(), parameters)
					acc = pure_LogReg.test(final_X_va,final_y_va.flatten())
					ans[split_date[1]+"_acc"].append(acc)
					ans[split_date[1]+"_length"].append(len(final_y_va))
			ans = pd.DataFrame(ans)
			ave = None
			length = None
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
			pd.DataFrame(ans).to_csv("_".join(["log_reg_all",self.version,str(self.lag),str(self.steps)+".csv"]))			
	"""
	this function is used to train the model and save it
	"""

	def train(self,C=0.01,tol=1e-7,max_iter=100):
		#os.chdir(os.path.abspath(sys.path[0]))
		if self.version in ['v5','v7','v9']:
			if self.source=='4E':
				from utils.read_data import read_data_v5_4E
				time_series, LME_dates = read_data_v5_4E("2003-11-12")
			elif self.source=='NExT':
				if self.version=='v5' or self.version=='v7':
					path = 'exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
					#if '4EBaseMetal' not in sys.path[0]:
					#	path = '4EBaseMetal/exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
					with open(os.path.join(os.getcwd(),path)) as fin:
						fname_columns = json.load(fin)
					from utils.read_data import read_data_NExT
					data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
					time_series = pd.concat(data_list, axis = 1, sort = True)
				elif self.version=='v9' or self.version=='v10' or self.version=='v12':
					path = 'exp/online_v10.conf'
					#if '4EBaseMetal' not in sys.path[0]:
					#	path = '4EBaseMetal/exp/online_v10.conf'
					with open(os.path.join(os.getcwd(),path)) as fin:
						fname_columns = json.load(fin)
					from utils.read_data import read_data_NExT
					data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
					time_series = pd.concat(data_list, axis = 1, sort = True)
			print("begin to train")
			norm_volume = "v1"
			norm_3m_spread = "v1"
			norm_ex = "v1"
			len_ma = 5
			len_update = 30
			tol = 1e-7
			pure_LogReg = LogReg(parameters={})
			if self.version in ['v5','v7','v9']:
				"""
				begin to split the train data
				"""
				today = self.date
				month = str(today).split("-")[1]
				year = str(today).split("-")[0]
				if month <="06":
					if month=="01":
						day = int(str(today).split("-")[2])
						if day <=self.horizon:
							print("the data is not enough for training")
							os.exit(0)
						else:
							start_year = int(year)-5
							start_time = str(start_year)+"-01-01"
							evalidate_year = int(year)
							evalidate_date = str(evalidate_year)+"-01-01"
							end_time = str(today)
					else:
						start_year = int(year)-5
						start_time = str(start_year)+"-01-01"
						evalidate_year = int(year)
						evalidate_date = str(evalidate_year)+"-01-01"
						end_time = str(today)
				else:
					if month=="07":
						day = int(str(today).split("-")[2])
						if day<=self.horizon:
							print("the data is not enough for training")
							os.exit(0)
						else:
							year = str(today).split("-")[0]
							start_year = int(year)-5
							end_year = int(year)
							start_time = str(start_year)+"-07-01"
							evalidate_year = int(year)
							evalidate_date = str(evalidate_year)+"-07-01"
							end_time = str(today)
					else:
						year = str(today).split("-")[0]
						start_year = int(year)-5
						end_year = int(year)
						start_time = str(start_year)+"-07-01"
						evalidate_year = int(year)
						evalidate_date = str(evalidate_year)+"-07-01"
						end_time = str(today)
				split_dates  =  [start_time,evalidate_date,str(today)]
				"""
				generate the version
				"""			
				version_params=generate_version_params(self.version)
				
				
				norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
								'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
				tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
												'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
				ts = copy(time_series.loc[split_dates[0]:split_dates[2]])

				X_tr, y_tr, X_va, y_va, X_te, y_te, norm_params,column_list = load_data(ts,LME_dates,self.horizon,[self.gt],self.lag,copy(split_dates),norm_params,tech_params,version_params)

				X_tr = np.concatenate(X_tr)
				X_tr = X_tr.reshape(len(X_tr),self.lag*len(column_list[0]))
				y_tr = np.concatenate(y_tr)
				X_va = np.concatenate(X_va)
				X_va = X_va.reshape(len(X_va),self.lag*len(column_list[0]))
				y_va = np.concatenate(y_va)
				max_iter = max_iter
				parameters = {"penalty":"l2", "C":C, "solver":"lbfgs", "tol":tol,"max_iter":6*4*len(fname_columns[0])*max_iter, "verbose" : 0,"warm_start": False, "n_jobs": -1}
				pure_LogReg.train(X_tr,y_tr.flatten(), parameters)
				pure_LogReg.save(version=self.version, gt = self.gt, horizon=self.horizon, lag = self.lag)
		else:
			"""
			begin to split the train data
			"""
			if self.source=='4E':
				from utils.read_data import read_data_v5_4E
				time_series, LME_dates = read_data_v5_4E("2003-11-12")
			elif self.source=='NExT':
				if self.version=='v5' or self.version=='v7':
					path = 'exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
					#if '4EBaseMetal' not in sys.path[0]:
					#	path = '4EBaseMetal/exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
					with open(os.path.join(os.getcwd(),path)) as fin:
						fname_columns = json.load(fin)
					from utils.read_data import read_data_NExT
					data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
					time_series = pd.concat(data_list, axis = 1, sort = True)
				elif self.version=='v9' or self.version=='v10' or self.version=='v12':
					path = 'exp/online_v10.conf'
					#if '4EBaseMetal' not in sys.path[0]:
					#	path = '4EBaseMetal/exp/online_v10.conf'
					with open(os.path.join(os.getcwd(),path)) as fin:
						fname_columns = json.load(fin)
					from utils.read_data import read_data_NExT
					data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
					time_series = pd.concat(data_list, axis = 1, sort = True)			
			today = self.date
			month = str(today).split("-")[1]
			if month <="06":
				if month=="01":
					day = int(str(today).split("-")[2])
					if day <=self.horizon:
						print("the data is not enough for training")
						os.exit(0)
					else:
						start_year = int(year)-5
						start_time = str(start_year)+"-01-01"
						evalidate_year = int(year)
						evalidate_date = str(evalidate_year)+"-01-01"
						end_time = str(today)
				else:
					start_year = int(year)-5
					start_time = str(start_year)+"-01-01"
					evalidate_year = int(year)
					evalidate_date = str(evalidate_year)+"-01-01"
					end_time = str(today)
			else:
				if month=="07":
					day = int(str(today).split("-")[2])
					if day<=self.horizon:
						print("the data is not enough for training")
						os.exit(0)
					else:
						year = str(today).split("-")[0]
						start_year = int(year)-5
						end_year = int(year)
						start_time = str(start_year)+"-07-01"
						evalidate_year = int(year)
						evalidate_date = str(evalidate_year)+"-07-01"
						end_time = str(today)
				else:
					year = str(today).split("-")[0]
					start_year = int(year)-5
					end_year = int(year)
					start_time = str(start_year)+"-07-01"
					evalidate_year = int(year)
					evalidate_date = str(evalidate_year)+"-07-01"
					end_time = str(today)
			split_dates  =  [start_time,evalidate_date,str(today)]
			"""
			generate the version
			"""			
			version_params=generate_version_params(self.version)

			norm_volume = "v1"
			norm_3m_spread = "v1"
			norm_ex = "v1"
			len_ma = 5
			len_update = 30
			tol = 1e-7
			norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
							'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
			tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
											'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
			ts = copy(time_series.loc[split_dates[0]:split_dates[2]])
			for ground_truth in ['LME_Co_Spot','LME_Al_Spot','LME_Ni_Spot','LME_Ti_Spot','LME_Zi_Spot','LME_Le_Spot']:
				print(ground_truth)
				metal_id = [0,0,0,0,0,0]
				metal_id[i] = 1
				#load data
				X_tr, y_tr, X_va, y_va, X_te, y_te, norm_check,column_list = load_data(copy(ts),LME_dates,horizon,[ground_truth],lag,copy(split_dates),norm_params,tech_params,version_params)
				#post load processing and metal id extension
				X_tr = np.concatenate(X_tr)
				X_tr = X_tr.reshape(len(X_tr),lag*len(column_list[0]))
				X_tr = np.append(X_tr,[metal_id]*len(X_tr),axis = 1)
				y_tr = np.concatenate(y_tr)
				X_va = np.concatenate(X_va)
				X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
				X_va = np.append(X_va,[metal_id]*len(X_va),axis = 1)
				y_va = np.concatenate(y_va)
				final_X_tr.append(X_tr)
				final_y_tr.append(y_tr)
				final_X_va.append(X_va)
				final_y_va.append(y_va)
				i+=1
			#sort by time, not by metal
			final_X_tr = [np.transpose(arr) for arr in np.dstack(final_X_tr)]
			final_y_tr = [np.transpose(arr) for arr in np.dstack(final_y_tr)]
			final_X_va = [np.transpose(arr) for arr in np.dstack(final_X_va)]
			final_y_va = [np.transpose(arr) for arr in np.dstack(final_y_va)]
			final_X_tr = np.reshape(final_X_tr,[np.shape(final_X_tr)[0]*np.shape(final_X_tr)[1],np.shape(final_X_tr)[2]])
			final_y_tr = np.reshape(final_y_tr,[np.shape(final_y_tr)[0]*np.shape(final_y_tr)[1],np.shape(final_y_tr)[2]])
			final_X_va = np.reshape(final_X_va,[np.shape(final_X_va)[0]*np.shape(final_X_va)[1],np.shape(final_X_va)[2]])
			final_y_va = np.reshape(final_y_va,[np.shape(final_y_va)[0]*np.shape(final_y_va)[1],np.shape(final_y_va)[2]])

			max_iter = args.max_iter
			parameters = {"penalty":"l2", "C":C, "solver":"lbfgs", "tol":tol,"max_iter":6*4*len(fname_columns[0])*max_iter, "verbose" : 0,"warm_start": False, "n_jobs": -1}
			pure_LogReg.train(final_X_tr,final_y_tr.flatten(), parameters)	
		pure_LogReg.save(self.version, self.gt, self.horizon, self.lag)
	"""
	this function is used to predict the date
	"""
	def test(self):
		"""
		split the date
		"""
		#os.chdir(os.path.abspath(sys.path[0]))
		print("begin to test")
		pure_LogReg = LogReg(parameters={})
		model = pure_LogReg.load(self.version, self.gt, self.horizon, self.lag)
		if self.version in ['v5','v7','v9']:
			if self.source=='4E':
				from utils.read_data import read_data_v5_4E
				time_series, LME_dates = read_data_v5_4E("2003-11-12")
			elif self.source=='NExT':
				if self.version=='v5' or self.version=='v7':
					path = 'exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
					#if '4EBaseMetal' not in sys.path[0]:
					#	path = '4EBaseMetal/exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
					with open(os.path.join(os.getcwd(),path)) as fin:
						fname_columns = json.load(fin)
					from utils.read_data import read_data_NExT
					data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
					time_series = pd.concat(data_list, axis = 1, sort = True)
				elif self.version=='v9' or self.version=='v10' or self.version=='v12':
					path = 'exp/online_v10.conf'
					#if '4EBaseMetal' not in sys.path[0]:
					#	path = '4EBaseMetal/exp/online_v10.conf'
					with open(os.path.join(os.getcwd(),path)) as fin:
						fname_columns = json.load(fin)
					from utils.read_data import read_data_NExT
					data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
					time_series = pd.concat(data_list, axis = 1, sort = True)
			#pure_LogReg = LogReg(parameters={})
			if self.version in ['v5','v7','v9']:
				
				"""
				begin to split the train data
				"""
				today = self.date
				month = str(today).split("-")[1]
				year = str(today).split("-")[0]
				if month <="06":
					if month=="01":
						day = int(str(today).split("-")[2])
						if day <=self.horizon:
							print("the data is not enough for training")
							os.exit(0)
						else:
							start_year = int(year)-5
							start_time = str(start_year)+"-01-01"
							evalidate_year = int(year)
							evalidate_date = str(evalidate_year)+"-01-01"
							end_time = str(today)
					else:
						start_year = int(year)-5
						start_time = str(start_year)+"-01-01"
						evalidate_year = int(year)
						evalidate_date = str(evalidate_year)+"-01-01"
						end_time = str(today)
				else:
					if month=="07":
						day = int(str(today).split("-")[2])
						if day<=self.horizon:
							print("the data is not enough for training")
							os.exit(0)
						else:
							year = str(today).split("-")[0]
							start_year = int(year)-5
							end_year = int(year)
							start_time = str(start_year)+"-07-01"
							evalidate_year = int(year)
							evalidate_date = str(evalidate_year)+"-07-01"
							end_time = str(today)
					else:
						year = str(today).split("-")[0]
						start_year = int(year)-5
						end_year = int(year)
						start_time = str(start_year)+"-07-01"
						evalidate_year = int(year)
						evalidate_date = str(evalidate_year)+"-07-01"
						end_time = str(today)
				split_dates  =  [start_time,evalidate_date,str(today)]
				"""
				generate the version
				"""
				version_params=generate_version_params(self.version)

				norm_volume = "v1"
				norm_3m_spread = "v1"
				norm_ex = "v1"
				len_ma = 5
				len_update = 30
				tol = 1e-7
				norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
								'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
				tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
												'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
				ts = copy(time_series.loc[split_dates[0]:split_dates[2]])
				date_list = time_series.loc[split_dates[1]:split_dates[2]].index.values.tolist()
				print(len(date_list))
				X_tr, y_tr, X_va, y_va, X_te, y_te, norm_params,column_list = load_data(ts,LME_dates,self.horizon,[self.gt],self.lag,copy(split_dates),norm_params,tech_params,version_params)
				"""
				begin to load the model
				"""
				#model = pure_LogReg.load(self.version, self.gt, self.horizon, self.lag)
				X_va = np.concatenate(X_va)
				X_va = X_va.reshape(len(X_va),self.lag*len(column_list[0]))
				y_va = np.concatenate(y_va)
				prob = model.predict(X_va)
				final_list = []
				piece_list = []
				for i,date in enumerate(date_list):
					piece_list.append(date)
					piece_list.append(prob[i])
					final_list.append(piece_list)
					piece_list=[]
				final_dataframe = pd.DataFrame(final_list, columns=['date','result'])
				return final_dataframe
		else:
			"""
			begin to split the train data
			"""
			if self.source=='4E':
				from utils.read_data import read_data_v5_4E
				time_series, LME_dates = read_data_v5_4E("2003-11-12")
			elif self.source=='NExT':
				if self.version=='v5' or self.version=='v7':
					path = 'exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
					#if '4EBaseMetal' not in sys.path[0]:
					#	path = '4EBaseMetal/exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
					with open(os.path.join(os.getcwd(),path)) as fin:
						fname_columns = json.load(fin)
					from utils.read_data import read_data_NExT
					data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
					time_series = pd.concat(data_list, axis = 1, sort = True)
				elif self.version=='v9' or self.version=='v10' or self.version=='v12':
					path = 'exp/online_v10.conf'
					#if '4EBaseMetal' not in sys.path[0]:
					#	path = '4EBaseMetal/exp/online_v10.conf'
					with open(os.path.join(os.getcwd(),path)) as fin:
						fname_columns = json.load(fin)
					from utils.read_data import read_data_NExT
					data_list, LME_dates = read_data_NExT(fname_columns[0], "2003-11-12")
					time_series = pd.concat(data_list, axis = 1, sort = True)			
			today = self.date
			month = str(today).split("-")[1]
			year = str(today).split("-")[0]
			if month <="06":
				if month=="01":
					day = int(str(today).split("-")[2])
					if day <=self.horizon:
						print("the data is not enough for training")
						os.exit(0)
					else:
						start_year = int(year)-5
						start_time = str(start_year)+"-01-01"
						evalidate_year = int(year)
						evalidate_date = str(evalidate_year)+"-01-01"
						end_time = str(today)
				else:
					start_year = int(year)-5
					start_time = str(start_year)+"-01-01"
					evalidate_year = int(year)
					evalidate_date = str(evalidate_year)+"-01-01"
					end_time = str(today)
			else:
				if month=="07":
					day = int(str(today).split("-")[2])
					if day<=self.horizon:
						print("the data is not enough for training")
						os.exit(0)
					else:
						year = str(today).split("-")[0]
						start_year = int(year)-5
						end_year = int(year)
						start_time = str(start_year)+"-07-01"
						evalidate_year = int(year)
						evalidate_date = str(evalidate_year)+"-07-01"
						end_time = str(today)
				else:
					year = str(today).split("-")[0]
					start_year = int(year)-5
					end_year = int(year)
					start_time = str(start_year)+"-07-01"
					evalidate_year = int(year)
					evalidate_date = str(evalidate_year)+"-07-01"
					end_time = str(today)
			split_dates  =  [start_time,evalidate_date,str(today)]
			version_params=generate_version_params(self.version)

			norm_volume = "v1"
			norm_3m_spread = "v1"
			norm_ex = "v1"
			len_ma = 5
			len_update = 30
			tol = 1e-7
			norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
							'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
			tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
											'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
			ts = copy(time_series.loc[split_dates[0]:split_dates[2]])
			date_list = time_series.loc[split_dates[1]:split_dates[2]].index.values.tolist()
			for ground_truth in ['LME_Co_Spot','LME_Al_Spot','LME_Ni_Spot','LME_Ti_Spot','LME_Zi_Spot','LME_Le_Spot']:
				print(ground_truth)
				metal_id = [0,0,0,0,0,0]
				metal_id[i] = 1
				#load data
				X_tr, y_tr, X_va, y_va, X_te, y_te, norm_check,column_list = load_data(copy(ts),LME_dates,horizon,[ground_truth],lag,copy(split_dates),norm_params,tech_params,version_params)
				#post load processing and metal id extension
				X_tr = np.concatenate(X_tr)
				X_tr = X_tr.reshape(len(X_tr),lag*len(column_list[0]))
				X_tr = np.append(X_tr,[metal_id]*len(X_tr),axis = 1)
				y_tr = np.concatenate(y_tr)
				X_va = np.concatenate(X_va)
				X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
				X_va = np.append(X_va,[metal_id]*len(X_va),axis = 1)
				y_va = np.concatenate(y_va)
				final_X_tr.append(X_tr)
				final_y_tr.append(y_tr)
				final_X_va.append(X_va)
				final_y_va.append(y_va)
				i+=1
			#sort by time, not by metal
			final_X_tr = [np.transpose(arr) for arr in np.dstack(final_X_tr)]
			final_y_tr = [np.transpose(arr) for arr in np.dstack(final_y_tr)]
			final_X_va = [np.transpose(arr) for arr in np.dstack(final_X_va)]
			final_y_va = [np.transpose(arr) for arr in np.dstack(final_y_va)]
			final_X_tr = np.reshape(final_X_tr,[np.shape(final_X_tr)[0]*np.shape(final_X_tr)[1],np.shape(final_X_tr)[2]])
			final_y_tr = np.reshape(final_y_tr,[np.shape(final_y_tr)[0]*np.shape(final_y_tr)[1],np.shape(final_y_tr)[2]])
			final_X_va = np.reshape(final_X_va,[np.shape(final_X_va)[0]*np.shape(final_X_va)[1],np.shape(final_X_va)[2]])
			final_y_va = np.reshape(final_y_va,[np.shape(final_y_va)[0]*np.shape(final_y_va)[1],np.shape(final_y_va)[2]])
			"""
			begin to load the model
			"""
			for i,gt in enumerate(["LMCADY","LMAHDY","LMNIDY","LMSNDY","LMZSDY","LMPBDY"]):
				if gt==self.gt:
					#model = pure_LogReg.load(self.version, self.gt, self.horizon, self.lag)
					prob = model.predict(final_X_va[i])
					final_list = []
					piece_list = []
					for i,date in enumerate(date_list):
						piece_list.append(date)
						piece_list.append(prob[i])
						final_list.append(piece_list)
						piece_list=[]
					final_dataframe = pd.DataFrame(final_list, columns=['date','result'])
					return final_dataframe
