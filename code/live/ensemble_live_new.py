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
			length = 0
			file_list = os.listdir("data/LR_probability/")
			all_list = []
			version_list = []
			for file in file_list:
				 if (self.gt+str(self.horizon)) in file:
				 	print(file)
				 	version = file.split("_")[-2]
				 	if version not in version_list:
				 		version_list.append(version)
			version_list.sort()
			for version in version_list:
				data = np.loadtxt("data/LR_probability/"+self.gt+str(self.horizon)+"_"+self.date+"_lr_"+version+"_probability.txt")
				all_list.append(data)
			results = []
			final_list_1 = []
			df = pd.DataFrame()
			list_length = len(all_list)
			error_list = [[]]*list_length
			final_version_list = [[]]*list_length
			for j in range(len(all_list[0])):
				count=0
				for i in range(list_length):
					if all_list[i][j]>0.5:
						final_version_list[i].append(1)
						count+=1
						# calculate the error
						if self.label[j]!=1:
							error_list[i].append(1)
						else:
							error_list[i].append(0)
					else:
						final_version_list[i].append(0)
						count+=0
						# calculate the error
						if self.label[j]!=0:
							error_list[i].append(1)
						else:
							error_list[i].append(0)
				if list_length%2!=0:
					if count>(list_length//2+1):
						results.append(1)
						if j < self.horizon:
							final_list_1.append(1)
					else:
						results.append(0)
						if j < self.horizon:
							final_list_1.append(0)
			print("the voting result is {}".format(metrics.accuracy_score(self.label, results)))
			#ensemble the data
			window = 1

			for i in range(self.horizon,len(self.label)):
				result = 0
				fenmu = 0
				for j in range(len(error_list)):
					error = np.sum(error_list[j][length:length+window])+1e-06
					fenmu+=1/error
				for j in range(len(error_list)):
					error = np.sum(error_list[j][length:length+window])+1e-06
					result+=((1/error)/fenmu) * final_version_list[j][i]			
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
			length = 0
			file_list = os.listdir("data/xgboost_folder/")
			all_list = []
			version_list = []
			for file in file_list:
				 if (self.gt+"_h"+str(self.horizon)) in file:
				 	print(file)
				 	version = file.split("_")[-1].split(".")[0]
				 	if version not in version_list:
				 		version_list.append(version)
			version_list.sort()
			for version in version_list:
				data = np.loadtxt("data/xgboost_folder/"+self.gt+"_"+"h"+str(self.horizon)+"_"+self.date+"_xgboost_"+version".txt")
				all_list.append(data)
			results = []
			final_list_1 = []
			df = pd.DataFrame()
			list_length = len(all_list)
			error_list = [[]]*list_length
			final_version_list = [[]]*list_length
			for j in range(len(all_list[0])):
				count=0
				for i in range(list_length):
					if all_list[i][j]>0.5:
						final_version_list[i].append(1)
						count+=1
						# calculate the error
						if self.label[j]!=1:
							error_list[i].append(1)
						else:
							error_list[i].append(0)
					else:
						final_version_list[i].append(0)
						count+=0
						# calculate the error
						if self.label[j]!=0:
							error_list[i].append(1)
						else:
							error_list[i].append(0)
				if list_length%2!=0:
					if count>(list_length//2+1):
						results.append(1)
						if j < self.horizon:
							final_list_1.append(1)
					else:
						results.append(0)
						if j < self.horizon:
							final_list_1.append(0)
			print("the voting result is {}".format(metrics.accuracy_score(self.label, results)))			
			window = 1
			for i in range(self.horizon,len(self.label)):
				result = 0
				fenmu = 0
				for j in range(len(error_list)):
					error = np.sum(error_list[j][length:length+window])+1e-06
					fenmu+=1/error
				for j in range(len(error_list)):
					error = np.sum(error_list[j][length:length+window])+1e-06
					result+=((1/error)/fenmu) * final_version_list[j][i]			
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