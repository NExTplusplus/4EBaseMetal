import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from sklearn import metrics
import math
import argparse
from copy import copy

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
			#error_list = [[]]*list_length
			final_version_list = []
			error_list = []
			for i in range(len(all_list)):
				final_list = []
				new_error_list = []
				for j in range(len(all_list[0])):
					if all_list[i][j]>0.5:
						final_list.append(1)
						# calculate the error
						if self.label[j]!=1:
							new_error_list.append(1)
						else:
							new_error_list.append(0)
					else:
						final_list.append(0)
						# calculate the error
						if self.label[j]!=0:
							new_error_list.append(1)
						else:
							new_error_list.append(0)
				error_list.append(new_error_list)
				final_version_list.append(final_list)
			for j in range(len(all_list[0])):
				count=0
				for i in range(list_length):
					if all_list[i][j]>0.5:
						count+=1
					else:
						count+=0
				if list_length%2!=0:
					if count>=(list_length//2+1):
						results.append(1)
						if j < self.horizon:
							final_list_1.append(1)
					else:
						results.append(0)
						if j < self.horizon:
							final_list_1.append(0)
				else:
					if count>(list_length/2):
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
			read_list = []
			version_list = []
			for file in file_list:
				 if (self.gt+"_h"+str(self.horizon)) in file:
				 	version = file.split("_")[-1].split(".")[0]
				 	if version not in version_list:
				 		version_list.append(version)
			version_list.sort()
			for version in version_list:
				#print("data/xgboost_folder/"+self.gt+"_h"+str(self.horizon)+"_"+self.date+"_xgboost_"+version+".txt")
				data = np.loadtxt("data/xgboost_folder/"+self.gt+"_h"+str(self.horizon)+"_"+self.date+"_xgboost_"+version+".txt")
				read_list.append(data)
			#print(all_list)
			results = []
			final_list_1 = []
			df = pd.DataFrame()
			list_length = len(read_list)
			all_list = []
			#error_list = [[]]*list_length
			final_version_list = []
			#print(len(self.label))
			#print(len(read_list))
			#print(len(read_list[0][0]))
			for i in range(len(read_list)):
				#print(i)
				#print(len(all_list[1]))
				new_list = []
				for j in range(len(read_list[i])):
					count_1 = 0
					count_0 = 0
					for item in read_list[i][j]:
						if item > 0.5:
							count_1+=1
						else:
							count_0+=1
					if count_1>count_0:
						#print(i)
						new_list.append(1)
					else:
						new_list.append(0)
				all_list.append(copy(new_list))
			#print(all_list[0])
			error_list = []
			for i in range(len(all_list)):
				final_list = []
				new_error_list = []
				for j in range(len(all_list[0])):
					if all_list[i][j]>0.5:
						final_list.append(1)
						# calculate the error
						if self.label[j]!=1:
							new_error_list.append(1)
						else:
							new_error_list.append(0)
					else:
						final_list.append(0)
						# calculate the error
						if self.label[j]!=0:
							new_error_list.append(1)
						else:
							new_error_list.append(0)
				error_list.append(new_error_list)
				final_version_list.append(final_list)
			for j in range(len(all_list[0])):
				count=0
				for i in range(list_length):
					if all_list[i][j]>0.5:
						count+=1
					else:
						count+=0
				if list_length%2!=0:
					#print(count)
					if count>=(list_length//2+1):
						results.append(1)
						if j < self.horizon:
							final_list_1.append(1)
					else:
						results.append(0)
						if j < self.horizon:
							final_list_1.append(0)
				else:
					if count>(list_length/2):
						results.append(1)
						if j < self.horizon:
							final_list_1.append(1)
					else:
						results.append(0)
						if j < self.horizon:
							final_list_1.append(0)
									
			print("the voting result is {}".format(metrics.accuracy_score(self.label, results)))
			#print(len(error_list))			
			window = 1
			for i in range(self.horizon,len(self.label)):
				result = 0
				fenmu = 0
				for j in range(list_length):
					error = copy(np.sum(error_list[j][length:length+window]))+1e-06
					#print(1/error)
					fenmu+=1/error
				#print(fenmu)
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
			folder_list = os.listdir("data/ALSTM_prediction")
			#print(folder_list)
			read_list = []
			for i in range(len(folder_list)):
				single_file_path = "data/ALSTM_prediction/"+folder_list[i]
				file_list = os.listdir(single_file_path)
				#print(file_list)
				for file in file_list:
					if (str(self.date)+"_"+str(self.horizon)) in file:
						#print(file)
						data = np.loadtxt("data/ALSTM_prediction/"+folder_list[i]+"/"+file)
						read_list.append(data)
			all_length = len(read_list[0])					
			ground_truths_list = ['LME_Co_Spot','LME_Al_Spot','LME_Le_Spot','LME_Ni_Spot','LME_Zi_Spot','LME_Ti_Spot']			
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
			all_list = []			
			for item in read_list:
				all_list.append(copy(item[start_index:end_index]))
			#print(all_list)
			results = []
			final_list_1 = []
			df = pd.DataFrame()
			list_length = len(all_list)
			final_version_list = []
			error_list = []
			#print(all_list)
			for i in range(len(all_list)):
				final_list = []
				new_error_list = []
				for j in range(len(all_list[0])):
					if all_list[i][j]==1:
						final_list.append(1)
						# calculate the error
						if self.label[j]!=1:
							new_error_list.append(1)
						else:
							new_error_list.append(0)
					else:
						final_list.append(0)
						# calculate the error
						if self.label[j]!=0:
							new_error_list.append(1)
						else:
							new_error_list.append(0)
				error_list.append(new_error_list)
				final_version_list.append(final_list)
			for j in range(len(all_list[0])):
				count=0

				for i in range(list_length):
					if all_list[i][j]==1:
						count+=1
					else:
						count+=0
				if list_length%2!=0:
					#print(list_length//2+1)
					print(count)
					if count>=(list_length//2+1):
						results.append(1)
						if j < self.horizon:
							final_list_1.append(1)
					else:
						results.append(0)
						if j < self.horizon:
							final_list_1.append(0)
				else:
					if count>(list_length/2):
						results.append(1)
						if j < self.horizon:
							final_list_1.append(1)
					else:
						results.append(0)
						if j < self.horizon:
							final_list_1.append(0)
			#print(results)
			print("the voting result is {}".format(metrics.accuracy_score(self.label[:], results)))

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

		return final_list_1