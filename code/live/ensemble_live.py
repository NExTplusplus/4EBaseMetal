import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from sklearn import metrics
import math
import argparse

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
				date,
				label,
				single_window = 0,
				delete_model = []):
		self.horizon = horizon
		self.gt = gt
		self.date = date
		self.single_window = single_window
		self.delete_model = delete_model
		self.label=label

	"""
	this function is to ensemble the single model result
	"""
	def single_model(self, model):
		"""
		model: the single model that you want to use ensemble
		"""
		
		"""
		this function is to ensemble the LR model for different versions
		"""
		length = 0
		if model == 'lr':
			lr_v3 = np.loadtxt("data/LR_probability/"+self.gt+str(self.horizon)+"_"+self.date+"_lr_v3_probability.txt")
			lr_v5 = np.loadtxt("data/LR_probability/"+self.gt+str(self.horizon)+"_"+self.date+"_lr_v5_probability.txt")
			lr_v7 = np.loadtxt("data/LR_probability/"+self.gt+str(self.horizon)+"_"+self.date+"_lr_v7_probability.txt")
			lr_v9 = np.loadtxt("data/LR_probability/"+self.gt+str(self.horizon)+"_"+self.date+"_lr_v9_probability.txt")
			lr_v23 = np.loadtxt("data/LR_probability/"+self.gt+str(self.horizon)+"_"+self.date+"_lr_v23_probability.txt")
			if self.gt=="LME_Co_Spot":
				lr_v10 = np.loadtxt("data/LR_probability/"+'LMCADY'+str(self.horizon)+"_"+self.date+"_lr_v10_probability.txt")
				lr_v12 = np.loadtxt("data/LR_probability/"+'LMCADY'+str(self.horizon)+"_"+self.date+"_lr_v12_probability.txt")
			elif self.gt=='LME_Al_Spot':
				lr_v10 = np.loadtxt("data/LR_probability/"+'LMAHDY'+str(self.horizon)+"_"+self.date+"_lr_v10_probability.txt")
				lr_v12 = np.loadtxt("data/LR_probability/"+'LMAHDY'+str(self.horizon)+"_"+self.date+"_lr_v12_probability.txt")
			elif self.gt=='LME_Le_Spot':
				lr_v10 = np.loadtxt("data/LR_probability/"+'LMPBDY'+str(self.horizon)+"_"+self.date+"_lr_v10_probability.txt")
				lr_v12 = np.loadtxt("data/LR_probability/"+'LMPBDY'+str(self.horizon)+"_"+self.date+"_lr_v12_probability.txt")
			elif self.gt=='LME_Ni_Spot':
				lr_v10 = np.loadtxt("data/LR_probability/"+'LMNIDY'+str(self.horizon)+"_"+self.date+"_lr_v10_probability.txt")
				lr_v12 = np.loadtxt("data/LR_probability/"+'LMNIDY'+str(self.horizon)+"_"+self.date+"_lr_v12_probability.txt")
			elif self.gt=='LME_Ti_Spot':
				lr_v10 = np.loadtxt("data/LR_probability/"+'LMSNDY'+str(self.horizon)+"_"+self.date+"_lr_v10_probability.txt")
				lr_v12 = np.loadtxt("data/LR_probability/"+'LMSNDY'+str(self.horizon)+"_"+self.date+"_lr_v12_probability.txt")
			elif self.gt=='LME_Zi_Spot':
				lr_v10 = np.loadtxt("data/LR_probability/"+'LMZSDY'+str(self.horizon)+"_"+self.date+"_lr_v10_probability.txt")
				lr_v12 = np.loadtxt("data/LR_probability/"+'LMZSDY'+str(self.horizon)+"_"+self.date+"_lr_v12_probability.txt")
			# self.label = pd.read_csv("data/Label/"+self.gt+"_h"+str(self.horizon)+"_"+date+"_label"+".csv")
			# self.label = list(self.label['Label'])
			result_v3_error = []
			result_v5_error = []
			result_v7_error = []
			result_v9_error = []
			result_v10_error = []
			result_v12_error = []
			result_v23_error = []
			final_list_v3 = []
			final_list_v5 = []
			final_list_v7 = []
			final_list_v9 = []
			final_list_v10 = []
			final_list_v12 = []
			final_list_v23 = []
			results = []
			final_list_1 = []
			df = pd.DataFrame()
			for j in range(len(lr_v3)):
				if lr_v3[j]>0.5:
					final_list_v3.append(1)
				else:
					final_list_v3.append(0)

				if lr_v5[j]>0.5:
					final_list_v5.append(1)
				else:
					final_list_v5.append(0)

				if lr_v7[j]>0.5:
					final_list_v7.append(1)
				else:
					final_list_v7.append(0)

				if lr_v9[j]>0.5:
					final_list_v9.append(1)
				else:
					final_list_v9.append(0)

				if lr_v10[j]>0.5:
					final_list_v10.append(1)
				else:
					final_list_v10.append(0)

				if lr_v12[j]>0.5:
					final_list_v12.append(1)
				else:
					final_list_v12.append(0)

				if lr_v23[j]>0.5:
					final_list_v23.append(1)
				else:
					final_list_v23.append(0)        

				if final_list_v3[-1]+final_list_v5[-1]+final_list_v7[-1]+final_list_v9[-1]+final_list_v10[-1]+final_list_v12[-1]+final_list_v23[-1]>=4:
					results.append(1)
					if j < self.horizon:
						final_list_1.append(1)
				else:
					results.append(0)
					if j < self.horizon:
						final_list_1.append(0)
				# calculate the error
				if self.label[j]!=final_list_v3[j]:
					result_v3_error.append(1)
				else:
					result_v3_error.append(0)

				if self.label[j]!=final_list_v5[j]:
					result_v5_error.append(1)
				else:
					result_v5_error.append(0)

				if self.label[j]!=final_list_v7[j]:
					result_v7_error.append(1)
				else:
					result_v7_error.append(0)

				if self.label[j]!=final_list_v9[j]:
					result_v9_error.append(1)
				else:
					result_v9_error.append(0)

				if self.label[j]!=final_list_v10[j]:
					result_v10_error.append(1)
				else:
					result_v10_error.append(0)

				if self.label[j]!=final_list_v12[j]:
					result_v12_error.append(1)
				else:
					result_v12_error.append(0)

				if self.label[j]!=final_list_v23[j]:
					result_v23_error.append(1)
				else:
					result_v23_error.append(0)

			print("the voting result is {}".format(metrics.accuracy_score(self.label, results)))

			window = 1
			for i in range(self.horizon,len(self.label)):
				error_lr_v3 = np.sum(result_v3_error[length:length+window])+1e-06
				error_lr_v5 = np.sum(result_v5_error[length:length+window])+1e-06
				error_lr_v7 = np.sum(result_v7_error[length:length+window])+1e-06
				error_lr_v9 = np.sum(result_v9_error[length:length+window])+1e-06
				error_lr_v10 = np.sum(result_v10_error[length:length+window])+1e-06
				error_lr_v12 = np.sum(result_v12_error[length:length+window])+1e-06
				error_lr_v23 = np.sum(result_v23_error[length:length+window])+1e-06 

				result = 0
				fenmu =1/error_lr_v3+ 1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v23
				weight_lr_v3 = float(1/error_lr_v3)/fenmu
				result+=weight_lr_v3*final_list_v3[i]
				weight_lr_v5 = float(1/error_lr_v5)/fenmu
				result+=weight_lr_v5*final_list_v5[i]
				weight_lr_v7 = float(1/error_lr_v7)/fenmu
				result+=weight_lr_v7*final_list_v7[i]
				weight_lr_v9 = float(1/error_lr_v9)/fenmu
				result+=weight_lr_v9*final_list_v9[i]
				weight_lr_v10 = float(1/error_lr_v10)/fenmu
				result+=weight_lr_v10*final_list_v10[i]
				weight_lr_v12 = float(1/error_lr_v12)/fenmu
				result+=weight_lr_v12*final_list_v12[i]
				weight_lr_v23 = float(1/error_lr_v23)/fenmu
				result+=weight_lr_v23*final_list_v23[i]				
				if result>0.5:
					final_list_1.append(1)
				else:
					final_list_1.append(0)

				if window==self.single_window:
					length+=1
				else:
					window+=1
			"""
			this function is to ensemble the xgboost model for different versions
			"""
		elif model == 'xgb':
			version_dict = {}
			xgboost_v3 = np.loadtxt("data/xgboost_folder/"+self.gt+"_"+"h"+str(self.horizon)+"_"+self.date+"_xgboost_v3.txt")
			#version_dict['v3']=xgboost_v3
			xgboost_v5 = np.loadtxt("data/xgboost_folder/"+self.gt+"_"+"h"+str(self.horizon)+"_"+self.date+"_xgboost_v5.txt")
			#version_dict['v5']=xgboost_v5
			xgboost_v7 = np.loadtxt("data/xgboost_folder/"+self.gt+"_"+"h"+str(self.horizon)+"_"+self.date+"_xgboost_v7.txt")
			#version_dict['v7']=xgboost_v7
			xgboost_v9 = np.loadtxt("data/xgboost_folder/"+self.gt+"_"+"h"+str(self.horizon)+"_"+self.date+"_xgboost_v9.txt")
			#version_dict['v9']=xgboost_v9
			xgboost_v10 = np.loadtxt("data/xgboost_folder/"+self.gt+"_"+"h"+str(self.horizon)+"_"+self.date+"_xgboost_v10.txt")
			#version_dict['v10']=xgboost_v10
			xgboost_v12 = np.loadtxt("data/xgboost_folder/"+self.gt+"_"+"h"+str(self.horizon)+"_"+self.date+"_xgboost_v12.txt")
			#version_dict['v12']=xgboost_v12
			xgboost_v23 = np.loadtxt("data/xgboost_folder/"+self.gt+"_"+"h"+str(self.horizon)+"_"+self.date+"_xgboost_v23.txt")
			#version_dict['v23']=xgboost_v23
			xgboost_v24 = np.loadtxt("data/xgboost_folder/"+self.gt+"_"+"h"+str(self.horizon)+"_"+self.date+"_xgboost_v24.txt")
			#version_dict['v24']=xgboost_v24
			xgboost_v28 = np.loadtxt("data/xgboost_folder/"+self.gt+"_"+"h"+str(self.horizon)+"_"+self.date+"_xgboost_v28.txt")
			xgboost_v30 = np.loadtxt("data/xgboost_folder/"+self.gt+"_"+"h"+str(self.horizon)+"_"+self.date+"_xgboost_v30.txt")
			length=0
			result_v3_error = []
			result_v5_error = []
			result_v7_error = []
			result_v9_error = []
			result_v10_error = []
			result_v12_error = []
			result_v23_error = []
			result_v24_error = []
			result_v28_error = []
			result_v30_error = []
			final_list_v3 = []
			final_list_v5 = []
			final_list_v7 = []
			final_list_v9 = []
			final_list_v10 = []
			final_list_v12 = []
			final_list_v23 = []
			final_list_v24 = []
			final_list_v28 = []
			final_list_v30 = []
			results = []
			final_list_1 = []
			df = pd.DataFrame()
			for j in range(len(xgboost_v23)):
				count_1 = 0
				count_0 = 0
				for item in xgboost_v3[j]:
					if item>0.5:
						count_1+=1
					else:
						count_0+=1
				if count_1>count_0:
					final_list_v3.append(1)
				else:
					final_list_v3.append(0)
				version_dict['v3']=final_list_v3
				count_1 = 0
				count_0 = 0
				for item in xgboost_v5[j]:
					if item>0.5:
						count_1+=1
					else:
						count_0+=1
				if count_1>count_0:
					final_list_v5.append(1)
				else:
					final_list_v5.append(0)
				version_dict['v5']=final_list_v5
				count_1 = 0
				count_0 = 0
				for item in xgboost_v7[j]:
					if item>0.5:
						count_1+=1
					else:
						count_0+=1
				if count_1>count_0:
					final_list_v7.append(1)
				else:
					final_list_v7.append(0)
				version_dict['v7']=final_list_v7
				count_1 = 0
				count_0 = 0
				for item in xgboost_v9[j]:
					if item>0.5:
						count_1+=1
					else:
						count_0+=1
				if count_1>count_0:
					final_list_v9.append(1)
				else:
					final_list_v9.append(0)
				version_dict['v9']=final_list_v9
				count_1 = 0
				count_0 = 0
				for item in xgboost_v10[j]:
					if item>0.5:
						count_1+=1
					else:
						count_0+=1
				if count_1>count_0:
					final_list_v10.append(1)
				else:
					final_list_v10.append(0)
				version_dict['v10']=final_list_v10
				count_1 = 0
				count_0 = 0
				for item in xgboost_v12[j]:
					if item>0.5:
						count_1+=1
					else:
						count_0+=1
				if count_1>count_0:
					final_list_v12.append(1)
				else:
					final_list_v12.append(0)
				version_dict['v12']=final_list_v12
				count_1 = 0
				count_0 = 0
				for item in xgboost_v23[j]:
					if item>0.5:
						count_1+=1
					else:
						count_0+=1
				if count_1>count_0:
					final_list_v23.append(1)
				else:
					final_list_v23.append(0)
				version_dict['v23']=final_list_v23
				count_1 = 0
				count_0 = 0
				for item in xgboost_v24[j]:
					if item>0.5:
						count_1+=1
					else:
						count_0+=1
				if count_1>count_0:
					final_list_v24.append(1)
				else:
					final_list_v24.append(0)
				count=0
				version_dict['v24']=final_list_v24
				count_1 = 0
				count_0 = 0
				for item in xgboost_v28[j]:
					if item>0.5:
						count_1+=1
					else:
						count_0+=1
				if count_1>count_0:
					final_list_v28.append(1)
				else:
					final_list_v28.append(0)
				count=0
				version_dict['v28']=final_list_v28
				count_1 = 0
				count_0 = 0
				for item in xgboost_v30[j]:
					if item>0.5:
						count_1+=1
					else:
						count_0+=1
				if count_1>count_0:
					final_list_v30.append(1)
				else:
					final_list_v30.append(0)
				count=0
				version_dict['v30']=final_list_v30								
				#print(len(version))
				gap=(len(version_dict.keys())-len(self.delete_model))//2+1
				#print(length)
				for key in version_dict.keys():
					if key not in self.delete_model:
						count+=version_dict[key][-1]
				#print(np.sum(count))
				if count >= gap:
					results.append(1)
					if j < self.horizon:
						final_list_1.append(1)
				  #print("done")
				else:
					results.append(0)
					if j < self.horizon:
				  		final_list_1.append(0)
				  #print("done")s
				# calculate the error

				if self.label[j]!=final_list_v3[j]:
					result_v3_error.append(1)
				else:
					result_v3_error.append(0)

				if self.label[j]!=final_list_v5[j]:
					result_v5_error.append(1)
				else:
					result_v5_error.append(0)

				if self.label[j]!=final_list_v7[j]:
					result_v7_error.append(1)
				else:
					result_v7_error.append(0)

				if self.label[j]!=final_list_v9[j]:
					result_v9_error.append(1)
				else:
					result_v9_error.append(0)

				if self.label[j]!=final_list_v10[j]:
					result_v10_error.append(1)
				else:
					result_v10_error.append(0)

				if self.label[j]!=final_list_v12[j]:
					result_v12_error.append(1)
				else:
					result_v12_error.append(0)

				if self.label[j]!=final_list_v23[j]:
					result_v23_error.append(1)
				else:
					result_v23_error.append(0)

				if self.label[j]!=final_list_v24[j]:
					result_v24_error.append(1)
				else:
					result_v24_error.append(0)
				if self.label[j]!=final_list_v28[j]:
					result_v28_error.append(1)
				else:
					result_v28_error.append(0)
				if self.label[j]!=final_list_v30[j]:
					result_v30_error.append(1)
				else:
					result_v30_error.append(0)
			print("the voting result is {}".format(metrics.accuracy_score(self.label, results)))
			
			window = 1
			for i in range(self.horizon,len(self.label)):
				error_dict = {}
				error_xgb_v3 = np.sum(result_v3_error[length:length+window])+1e-06
				error_dict['v3']=error_xgb_v3
				error_xgb_v5 = np.sum(result_v5_error[length:length+window])+1e-06
				error_dict['v5']=error_xgb_v5
				error_xgb_v7 = np.sum(result_v7_error[length:length+window])+1e-06
				error_dict['v7']=error_xgb_v7
				error_xgb_v9 = np.sum(result_v9_error[length:length+window])+1e-06
				error_dict['v9']=error_xgb_v9
				error_xgb_v10 = np.sum(result_v10_error[length:length+window])+1e-06
				error_dict['v10']=error_xgb_v10
				error_xgb_v12 = np.sum(result_v12_error[length:length+window])+1e-06
				error_dict['v12']=error_xgb_v12
				error_xgb_v23 = np.sum(result_v23_error[length:length+window])+1e-06
				error_dict['v23']=error_xgb_v23 
				error_xgb_v24 = np.sum(result_v24_error[length:length+window])+1e-06
				error_dict['v24']=error_xgb_v24
				error_xgb_v28 = np.sum(result_v28_error[length:length+window])+1e-06
				error_dict['v28']=error_xgb_v28
				error_xgb_v30 = np.sum(result_v30_error[length:length+window])+1e-06
				error_dict['v30']=error_xgb_v30								
				result = 0
				fenmu = 0
				for key in error_dict.keys():
					if key not in self.delete_model:
						fenmu += 1/error_dict[key]
				#print(fenmu)
				if 'v3' not in self.delete_model:
					weight_xgb_v3 = float(1/error_xgb_v3)/fenmu
					result+=weight_xgb_v3*final_list_v3[i]
				if 'v5' not in self.delete_model:
					weight_xgb_v5 = float(1/error_xgb_v5)/fenmu
					result+=weight_xgb_v5*final_list_v5[i]
				if 'v7' not in self.delete_model:
					weight_xgb_v7 = float(1/error_xgb_v7)/fenmu
					result+=weight_xgb_v7*final_list_v7[i]
				if 'v9' not in self.delete_model:
					weight_xgb_v9 = float(1/error_xgb_v9)/fenmu
					result+=weight_xgb_v9*final_list_v9[i]
				if 'v10' not in self.delete_model:
					weight_xgb_v10 = float(1/error_xgb_v10)/fenmu
					result+=weight_xgb_v10*final_list_v10[i]
				if 'v12' not in self.delete_model:
					weight_xgb_v12 = float(1/error_xgb_v12)/fenmu
					result+=weight_xgb_v12*final_list_v12[i]
				if 'v23' not in self.delete_model:
					weight_xgb_v23 = float(1/error_xgb_v23)/fenmu
					result+=weight_xgb_v23*final_list_v23[i]
				if 'v24' not in self.delete_model:
					weight_xgb_v24 = float(1/error_xgb_v24)/fenmu
					result+=weight_xgb_v24*final_list_v24[i]
				if 'v28' not in self.delete_model:
					weight_xgb_v28 = float(1/error_xgb_v28)/fenmu
					result+=weight_xgb_v28*final_list_v28[i]
				if 'v30' not in self.delete_model:
					weight_xgb_v30 = float(1/error_xgb_v30)/fenmu
					result+=weight_xgb_v30*final_list_v30[i] 					             
				if result>0.5:
					final_list_1.append(1)
				else:
					final_list_1.append(0)

				if window==self.single_window:
					length+=1
				else:
					window+=1
		elif model == "alstm":
			version_dict = {}
			if self.horizon==1:
				alstm_v16_loss = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.4_30_5_4_v16_prediction.txt")
				alstm_v16_accu = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.6_30_20_3_v16_prediction.txt")
				alstm_v26_choose = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.6_30_20_4_v26_prediction.txt")
				alstm_v26_accu = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.6_30_20_5_v26_prediction.txt")
				alstm_v26_loss = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.6_20_20_3_v26_prediction.txt")
			elif self.horizon==3:
				alstm_v16_loss = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.4_50_10_3_v16_prediction.txt")
				alstm_v16_accu = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.6_30_20_3_v16_prediction.txt")
				alstm_v26_choose = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.6_30_20_3_v26_prediction.txt")
				alstm_v26_accu = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.6_20_20_5_v26_prediction.txt")
				alstm_v26_loss = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.6_20_20_3_v26_prediction.txt")
			elif self.horizon==5:
				alstm_v16_loss = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.6_30_10_3_v16_prediction.txt")
				alstm_v16_accu = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.6_50_10_3_v16_prediction.txt")
				alstm_v26_choose = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.6_20_5_4_v26_prediction.txt")
				alstm_v26_accu = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.4_30_20_5_v26_prediction.txt")
				alstm_v26_loss = np.loadtxt("data/ALSTM_prediction/"+str(self.date)+"_"+str(self.horizon)+"_0.6_20_20_3_v26_prediction.txt")								
			ground_truths_list = ['LME_Co_Spot','LME_Al_Spot','LME_Le_Spot','LME_Ni_Spot','LME_Zi_Spot','LME_Ti_Spot']
			all_length = len(alstm_v16_loss)
			#print("the length of the ALSTM is {}".format(all_length))
			metal_length = all_length/6
			if self.gt == ground_truths_list[0]:
				start_index = 0
				end_index = int(metal_length)
			elif self.gt == ground_truths_list[1]:
				start_index = int(metal_length)
				end_index = 2*int(metal_length)
			elif self.gt == ground_truths_list[2]:
				start_index = 2*int(metal_length)
				end_index = 3*int(metal_length)
			elif self.gt == ground_truths_list[3]:
				start_index = 3*int(metal_length)
				end_index = 4*int(metal_length)
			elif self.gt == ground_truths_list[4]:
				start_index = 4*int(metal_length)
				end_index = 5*int(metal_length)
			elif self.gt == ground_truths_list[5]:
				start_index = 5*int(metal_length)
				end_index = 6*int(metal_length)
			alstm_v16_loss_metal = alstm_v16_loss[start_index:end_index]
			version_dict['v16_loss']=alstm_v16_loss_metal
			alstm_v16_accu_metal = alstm_v16_accu[start_index:end_index]
			version_dict['v16_accu']=alstm_v16_accu_metal
			alstm_v26_choose_metal = alstm_v26_choose[start_index:end_index]
			version_dict['v26_choose']=alstm_v26_choose_metal
			alstm_v26_accu_metal = alstm_v26_accu[start_index:end_index]
			version_dict['v26_accu']=alstm_v26_accu_metal
			alstm_v26_loss_metal = alstm_v26_loss[start_index:end_index]
			version_dict['v26_loss']=alstm_v26_loss_metal
			#print()
			result_alstm_v16_loss_metal_error = []
			result_alstm_v16_accu_metal_error = []
			result_alstm_v26_choose_metal_error = []
			result_alstm_v26_accu_metal_error = []
			result_alstm_v26_loss_metal_error = []
			results = []
			final_list_1 = []
			df = pd.DataFrame()
			for j in range(len(alstm_v16_loss_metal)):
				count=0        
				gap=(len(version_dict.keys())-len(self.delete_model))//2+1
				#print(length)
				for key in version_dict.keys():
					if key not in self.delete_model:
						count+=version_dict[key][-1]
				#print(np.sum(count))
				if count >= gap:
					results.append(1)
					if j < self.horizon:
						final_list_1.append(1)
				  #print("done")
				else:
					results.append(0)
					if j < self.horizon:
				  		final_list_1.append(0)
				# calculate the error
				if self.label[j]!=alstm_v16_loss_metal[j]:
					result_alstm_v16_loss_metal_error.append(1)
				else:
					result_alstm_v16_loss_metal_error.append(0)

				if self.label[j]!=alstm_v16_accu_metal[j]:
					result_alstm_v16_accu_metal_error.append(1)
				else:
					result_alstm_v16_accu_metal_error.append(0)

				if self.label[j]!=alstm_v26_choose_metal[j]:
					result_alstm_v26_choose_metal_error.append(1)
				else:
					result_alstm_v26_choose_metal_error.append(0)

				if self.label[j]!=alstm_v26_accu_metal[j]:
					result_alstm_v26_accu_metal_error.append(1)
				else:
					result_alstm_v26_accu_metal_error.append(0)

				if self.label[j]!=alstm_v26_loss_metal[j]:
					result_alstm_v26_loss_metal_error.append(1)
				else:
					result_alstm_v26_loss_metal_error.append(0)

			print("the voting result is {}".format(metrics.accuracy_score(self.label[:], results)))

			window = 1
			for i in range(self.horizon,len(alstm_v16_loss_metal)):
				error_dict = {}
				error_alstm_v16_loss = np.sum(result_alstm_v16_loss_metal_error[length:length+window])+1e-06
				error_dict['v16_loss']=error_alstm_v16_loss
				error_alstm_v16_accu = np.sum(result_alstm_v16_accu_metal_error[length:length+window])+1e-06
				error_dict['v16_accu']=error_alstm_v16_accu
				error_alstm_v26_choose = np.sum(result_alstm_v26_choose_metal_error[length:length+window])+1e-06
				error_dict['v26_choose']=error_alstm_v26_choose
				error_alstm_v26_accu = np.sum(result_alstm_v26_accu_metal_error[length:length+window])+1e-06
				error_dict['v26_accu']=error_alstm_v26_accu
				error_alstm_v26_loss = np.sum(result_alstm_v26_loss_metal_error[length:length+window])+1e-06
				error_dict['v26_loss']=error_alstm_v26_loss

				result = 0
				fenmu = 0
				for key in error_dict.keys():
					if key not in self.delete_model:
						fenmu += 1/error_dict[key]
				if 'v16_loss' not in self.delete_model: 
					weight_alstm_v16_loss = float(1/error_alstm_v16_loss)/fenmu
					result+=weight_alstm_v16_loss*alstm_v16_loss_metal[i]
				if 'v16_accu' not in self.delete_model:
					weight_alstm_v16_accu = float(1/error_alstm_v16_accu)/fenmu
					result+=weight_alstm_v16_accu*alstm_v16_accu_metal[i]
				if 'v26_choose' not in self.delete_model:
					weight_alstm_v26_choose = float(1/error_alstm_v26_choose)/fenmu
					result+=weight_alstm_v26_choose*alstm_v26_choose_metal[i]
				if 'v26_accu' not in self.delete_model:
					weight_alstm_v26_accu = float(1/error_alstm_v26_accu)/fenmu
					result+=weight_alstm_v26_accu*alstm_v26_accu_metal[i]
				if 'v26_loss' not in self.delete_model:
					weight_alstm_v26_loss = float(1/error_alstm_v26_loss)/fenmu
					result+=weight_alstm_v26_loss*alstm_v26_loss_metal[i]
				if result>0.5:
				  final_list_1.append(1)
				else:
				  final_list_1.append(0)

				if window==self.single_window:
				  length+=1
				else:
				  window+=1					

		return final_list_1



