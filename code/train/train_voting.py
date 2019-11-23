import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from sklearn import metrics
import math

for horizon in [1,3,5]:
	for ground_truth in ["LME_Co_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot","LME_Le_Spot"]:
		#for date in ["2014-07-01","2015-01-01","2015-07-01","2016-01-01","2016-07-01","2017-01-01","2017-07-01","2018-01-01","2018-07-01"]:
			for window_size in [5,10,15,20,25,30]:
			#for beta in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
				#result_v3_error = []
				#result_v5_error = []
				#result_v7_error = []
				#result_v9_error = []
				#result_v10_error = []
				#result_v12_error = []
				#result_v24_error = []
				#length=0
				#window_size = 5
				for date in ["2014-07-01","2015-01-01","2015-07-01","2016-01-01","2016-07-01","2017-01-01","2017-07-01","2018-01-01","2018-07-01"]:
					lr_v3 = np.loadtxt("data/LR_probability/"+ground_truth+str(horizon)+"_"+date+"_lr_v3_probability.txt")
					lr_v5 = np.loadtxt("data/LR_probability/"+ground_truth+str(horizon)+"_"+date+"_lr_v5_probability.txt")
					lr_v7 = np.loadtxt("data/LR_probability/"+ground_truth+str(horizon)+"_"+date+"_lr_v7_probability.txt")
					lr_v9 = np.loadtxt("data/LR_probability/"+ground_truth+str(horizon)+"_"+date+"_lr_v9_probability.txt")
					if ground_truth=="LME_Co_Spot":
						lr_v10 = np.loadtxt("data/LR_probability/"+'LMCADY'+str(horizon)+"_"+date+"_lr_v10_probability.txt")
						lr_v12 = np.loadtxt("data/LR_probability/"+'LMCADY'+str(horizon)+"_"+date+"_lr_v12_probability.txt")
					elif ground_truth=='LME_Al_Spot':
						lr_v10 = np.loadtxt("data/LR_probability/"+'LMAHDY'+str(horizon)+"_"+date+"_lr_v10_probability.txt")
						lr_v12 = np.loadtxt("data/LR_probability/"+'LMAHDY'+str(horizon)+"_"+date+"_lr_v12_probability.txt")
					elif ground_truth=='LME_Le_Spot':
						lr_v10 = np.loadtxt("data/LR_probability/"+'LMPBDY'+str(horizon)+"_"+date+"_lr_v10_probability.txt")
						lr_v12 = np.loadtxt("data/LR_probability/"+'LMPBDY'+str(horizon)+"_"+date+"_lr_v12_probability.txt")
					elif ground_truth=='LME_Ni_Spot':
						lr_v10 = np.loadtxt("data/LR_probability/"+'LMNIDY'+str(horizon)+"_"+date+"_lr_v10_probability.txt")
						lr_v12 = np.loadtxt("data/LR_probability/"+'LMNIDY'+str(horizon)+"_"+date+"_lr_v12_probability.txt")
					elif ground_truth=='LME_Ti_Spot':
						lr_v10 = np.loadtxt("data/LR_probability/"+'LMSNDY'+str(horizon)+"_"+date+"_lr_v10_probability.txt")
						lr_v12 = np.loadtxt("data/LR_probability/"+'LMSNDY'+str(horizon)+"_"+date+"_lr_v12_probability.txt")
					elif ground_truth=='LME_Zi_Spot':
						lr_v10 = np.loadtxt("data/LR_probability/"+'LMZSDY'+str(horizon)+"_"+date+"_lr_v10_probability.txt")
						lr_v12 = np.loadtxt("data/LR_probability/"+'LMZSDY'+str(horizon)+"_"+date+"_lr_v12_probability.txt")
					lr_v24 = np.loadtxt("data/LR_probability/"+ground_truth+str(horizon)+"_"+date+"_lr_v23_probability.txt")
					length=0
					result_v3_error = []
					result_v5_error = []
					result_v7_error = []
					result_v9_error = []
					result_v10_error = []
					result_v12_error = []
					result_v24_error = []
					final_list_v3 = []
					for j in range(len(lr_v3)):
						if lr_v3[j]>0.5:
							final_list_v3.append(1)
						else:
							final_list_v3.append(0)

					final_list_v5 = []
					for j in range(len(lr_v5)):
						if lr_v5[j]>0.5:
							final_list_v5.append(1)
						else:
							final_list_v5.append(0)
					
					final_list_v7 = []
					for j in range(len(lr_v7)):
						if lr_v7[j]>0.5:
							final_list_v7.append(1)
						else:
							final_list_v7.append(0)

					final_list_v9 = []
					for j in range(len(lr_v9)):
						if lr_v9[j]>0.5:
							final_list_v9.append(1)
						else:
							final_list_v9.append(0)

					final_list_v10 = []
					for j in range(len(lr_v10)):
						if lr_v10[j]>0.5:
							final_list_v10.append(1)
						else:
							final_list_v10.append(0)

					final_list_v12 = []
					for j in range(len(lr_v12)):
						if lr_v12[j]>0.5:
							final_list_v12.append(1)
						else:
							final_list_v12.append(0)

					final_list_v24 = []
					for j in range(len(lr_v24)):
						if lr_v24[j]>0.5:
							final_list_v24.append(1)
						else:
							final_list_v24.append(0)
					"""
					get the label from the file
					"""
					"""
					if ground_truth.split("_")[1]=="Co":
						if date>='2017-01-03':
							y_va = pd.read_csv("data/Label/"+'LMCADY'+str(horizon)+"_"+date+"_label"+".csv")
							y_va = list(y_va['LMCADY'])
						else:
							y_va = pd.read_csv("data/Label/"+'LMCADY'+str(horizon)+"_"+date+"_label"+".csv")
							y_va = list(y_va['LMCADY'])
					elif ground_truth.split("_")[1]=="Al":
						if date>='2017-01-03':
							y_va = pd.read_csv("data/Label/"+'LMAHDY'+str(horizon)+"_"+date+"_label"+".csv")
							y_va = list(y_va['LMAHDY'])
						else:
							y_va = pd.read_csv("data/Label/"+'LMAHDY'+str(horizon)+"_"+date+"_label"+".csv")
							y_va = list(y_va['LMAHDY'])
					elif ground_truth.split("_")[1]=="Ni":
						if date>='2017-01-03':
							y_va = pd.read_csv("data/Label/"+'LMNIDY'+str(horizon)+"_"+date+"_label"+".csv")
							y_va = list(y_va['LMNIDY'])
						else:
							y_va = pd.read_csv("data/Label/"+'LMNIDY'+str(horizon)+"_"+date+"_label"+".csv")
							y_va = list(y_va['LMNIDY'])
					elif ground_truth.split("_")[1]=="Ti":
						if date>='2017-01-03':
							y_va = pd.read_csv("data/Label/"+'LMSNDY'+str(horizon)+"_"+date+"_label"+".csv")
							y_va = list(y_va['LMSNDY'])
						else:
							y_va = pd.read_csv("data/Label/"+'LMSNDY'+str(horizon)+"_"+date+"_label"+".csv")
							y_va = list(y_va['LMSNDY'])
					elif ground_truth.split("_")[1]=="Zi":
						if date>='2017-01-03':
							y_va = pd.read_csv("data/Label/"+'LMZSDY'+str(horizon)+"_"+date+"_label"+".csv")
							y_va = list(y_va['LMZSDY'])
						else:
							y_va = pd.read_csv("data/Label/"+'LMZSDY'+str(horizon)+"_"+date+"_label"+".csv")
							y_va = list(y_va['LMZSDY'])
					elif ground_truth.split("_")[1]=="Le":
						if date>='2017-01-03':
							y_va = pd.read_csv("data/Label/"+'LMPBDY'+str(horizon)+"_"+date+"_label"+".csv")
							y_va = list(y_va['LMPBDY'])
						else:
							y_va = pd.read_csv("data/Label/"+'LMPBDY'+str(horizon)+"_"+date+"_label"+".csv")
							y_va = list(y_va['LMPBDY'])
					"""
					y_va = pd.read_csv("data/Label/"+ground_truth+"_h"+str(horizon)+"_"+date+"_label"+".csv")
					y_va = list(y_va['Label'])
					result = []
					for i in range(len(final_list_v3)):
						if final_list_v3[i]+final_list_v5[i]+final_list_v7[i]+final_list_v9[i]+final_list_v10[i]+final_list_v12[i]+final_list_v24[i]>=4:
							result.append(1)
						else:
							result.append(0)
					print("the voting result is {}".format(metrics.accuracy_score(y_va, result)))
					final_list_1 = []
					#true_result = []
					for i in range(horizon):
						if final_list_v3[i]+final_list_v5[i]+final_list_v7[i]+final_list_v9[i]+final_list_v10[i]+final_list_v12[i]+final_list_v24[i]>=4:
							final_list_1.append(1)
						else:
							final_list_1.append(0)
					# calculate the error
					for i in range(len(final_list_v3)):
						if y_va[i]!=final_list_v3[i]:
							result_v3_error.append(1)
						else:
							result_v3_error.append(0)

					for i in range(len(final_list_v5)):
						if y_va[i]!=final_list_v5[i]:
							result_v5_error.append(1)
						else:
							result_v5_error.append(0)

					for i in range(len(final_list_v7)):
						if y_va[i]!=final_list_v7[i]:
							result_v7_error.append(1)
						else:
							result_v7_error.append(0)

					for i in range(len(final_list_v9)):
						if y_va[i]!=final_list_v9[i]:
							result_v9_error.append(1)
						else:
							result_v9_error.append(0)

					for i in range(len(final_list_v10)):
						if y_va[i]!=final_list_v10[i]:
							result_v10_error.append(1)
						else:
							result_v10_error.append(0)

					for i in range(len(final_list_v12)):
						if y_va[i]!=final_list_v12[i]:
							result_v12_error.append(1)
						else:
							result_v12_error.append(0)

					for i in range(len(final_list_v24)):
						if y_va[i]!=final_list_v24[i]:
							result_v24_error.append(1)
						else:
							result_v24_error.append(0)
					window = 1
					for i in range(horizon,len(y_va)):
						#true_result.append(y_va[i])
						"""
						error_lr_v3 = ((np.sum(result_v3_error[length:length+window]))/window)+1e-06
						error_lr_v5 = (np.sum(result_v5_error[length:length+window]))/window+1e-06
						error_lr_v7 = (np.sum(result_v7_error[length:length+window]))/window+1e-06
						error_lr_v9 = (np.sum(result_v9_error[length:length+window]))/window+1e-06
						error_lr_v10 = (np.sum(result_v10_error[length:length+window]))/window+1e-06
						error_lr_v12 = (np.sum(result_v12_error[length:length+window]))/window+1e-06
						error_lr_v24 = (np.sum(result_v24_error[length:length+window]))/window+1e-06
						"""
						#print(result_v3_error)
						#print(length)
						#print(window)
						#print(result_v3_error[length:length+window])
						#print(error_lr_v3)
						
						error_lr_v3 = np.sum(result_v3_error[length:length+window])+1e-06
						error_lr_v5 = np.sum(result_v5_error[length:length+window])+1e-06
						error_lr_v7 = np.sum(result_v7_error[length:length+window])+1e-06
						error_lr_v9 = np.sum(result_v9_error[length:length+window])+1e-06
						error_lr_v10 = np.sum(result_v10_error[length:length+window])+1e-06
						error_lr_v12 = np.sum(result_v12_error[length:length+window])+1e-06
						error_lr_v24 = np.sum(result_v24_error[length:length+window])+1e-06	
						
									
						accuracy_list = []
						accuracy_list.append((1,error_lr_v3))
						accuracy_list.append((2,error_lr_v5))
						accuracy_list.append((3,error_lr_v7))
						accuracy_list.append((4,error_lr_v9))
						accuracy_list.append((5,error_lr_v10))
						accuracy_list.append((6,error_lr_v12))
						accuracy_list.append((7,error_lr_v24))
						accuracy_list.sort(reverse=False, key=lambda x: x[1])

						
						#accuracy = ((np.sum(result_v7_error[length:length+window]))/window)
						#if accuracy>0.4:
						#	final_list_1.append(final_list_v7[i])
						#else:
						#	if final_list_v7[i]==0:
						#		final_list_1.append(1)
						#	else:
						#		final_list_1.append(0)
						#print(accuracy_list)
						#print(accuracy_list)
						minor = accuracy_list[-1][1]
						#print(minor)
						#print(minor)
						delete_list = []
						for item in accuracy_list:
							if item[1]==minor:
								delete_list.append(item)
						#print(delete_list)
						delete_length = len(delete_list)
						number = np.random.randint(0,delete_length)-1
						#print(number)
						delete_number = delete_list[number][0]
						#print(delete_number)
						result = 0
						#print(delete_number)
						"""
						if delete_number==1:
							fenmu = error_lr_v5+error_lr_v7+error_lr_v9+error_lr_v10+error_lr_v12+error_lr_v24
							weight_lr_v5 = float(error_lr_v5)/fenmu
							result+=weight_lr_v5*final_list_v5[i]
							weight_lr_v7 = float(error_lr_v7)/fenmu
							result+=weight_lr_v7*final_list_v7[i]
							weight_lr_v9 = float(error_lr_v9)/fenmu
							result+=weight_lr_v9*final_list_v9[i]
							weight_lr_v10 = float(error_lr_v10)/fenmu
							result+=weight_lr_v10*final_list_v10[i]
							weight_lr_v12 = float(error_lr_v12)/fenmu
							result+=weight_lr_v12*final_list_v12[i]
							weight_lr_v24 = float(error_lr_v24)/fenmu
							result+=weight_lr_v24*final_list_v24[i]																												
						elif delete_number==2:
							fenmu = error_lr_v3+error_lr_v7+error_lr_v9+error_lr_v10+error_lr_v12+error_lr_v24
							weight_lr_v3 = float(error_lr_v3)/fenmu
							result+=weight_lr_v3*final_list_v3[i]
							weight_lr_v7 = float(error_lr_v7)/fenmu
							result+=weight_lr_v7*final_list_v7[i]
							weight_lr_v9 = float(error_lr_v9)/fenmu
							result+=weight_lr_v9*final_list_v9[i]
							weight_lr_v10 = float(error_lr_v10)/fenmu
							result+=weight_lr_v10*final_list_v10[i]
							weight_lr_v12 = float(error_lr_v12)/fenmu
							result+=weight_lr_v12*final_list_v12[i]
							weight_lr_v24 = float(error_lr_v24)/fenmu
							result+=weight_lr_v24*final_list_v24[i]								
						elif delete_number==3:
							fenmu = error_lr_v3+error_lr_v5+error_lr_v9+error_lr_v10+error_lr_v12+error_lr_v24
							weight_lr_v3 = float(error_lr_v3)/fenmu
							result+=weight_lr_v3*final_list_v3[i]
							weight_lr_v5 = float(error_lr_v5)/fenmu
							result+=weight_lr_v5*final_list_v5[i]
							weight_lr_v9 = float(error_lr_v9)/fenmu
							result+=weight_lr_v9*final_list_v9[i]
							weight_lr_v10 = float(error_lr_v10)/fenmu
							result+=weight_lr_v10*final_list_v10[i]
							weight_lr_v12 = float(error_lr_v12)/fenmu
							result+=weight_lr_v12*final_list_v12[i]
							weight_lr_v24 = float(error_lr_v24)/fenmu
							result+=weight_lr_v24*final_list_v24[i]
						elif delete_number==4:
							fenmu = error_lr_v3+error_lr_v5+error_lr_v7+error_lr_v10+error_lr_v12+error_lr_v24
							weight_lr_v3 = float(error_lr_v3)/fenmu
							result+=weight_lr_v3*final_list_v3[i]
							weight_lr_v5 = float(error_lr_v5)/fenmu
							result+=weight_lr_v5*final_list_v5[i]
							weight_lr_v7 = float(error_lr_v7)/fenmu
							result+=weight_lr_v7*final_list_v7[i]
							weight_lr_v10 = float(error_lr_v10)/fenmu
							result+=weight_lr_v10*final_list_v10[i]
							weight_lr_v12 = float(error_lr_v12)/fenmu
							result+=weight_lr_v12*final_list_v12[i]
							weight_lr_v24 = float(error_lr_v24)/fenmu
							result+=weight_lr_v24*final_list_v24[i]	
						elif delete_number==5:
							fenmu = error_lr_v3+error_lr_v5+error_lr_v7+error_lr_v9+error_lr_v12+error_lr_v24
							weight_lr_v3 = float(error_lr_v3)/fenmu
							result+=weight_lr_v3*final_list_v3[i]
							weight_lr_v5 = float(error_lr_v5)/fenmu
							result+=weight_lr_v5*final_list_v5[i]
							weight_lr_v7 = float(error_lr_v7)/fenmu
							result+=weight_lr_v7*final_list_v7[i]
							weight_lr_v9 = float(error_lr_v9)/fenmu
							result+=weight_lr_v9*final_list_v9[i]
							weight_lr_v12 = float(error_lr_v12)/fenmu
							result+=weight_lr_v12*final_list_v12[i]
							weight_lr_v24 = float(error_lr_v24)/fenmu
							result+=weight_lr_v24*final_list_v24[i]	
						elif delete_number==6:
							fenmu = error_lr_v3+error_lr_v5+error_lr_v7+error_lr_v9+error_lr_v10+error_lr_v24
							weight_lr_v3 = float(error_lr_v3)/fenmu
							result+=weight_lr_v3*final_list_v3[i]
							weight_lr_v5 = float(error_lr_v5)/fenmu
							result+=weight_lr_v5*final_list_v5[i]
							weight_lr_v7 = float(error_lr_v7)/fenmu
							result+=weight_lr_v7*final_list_v7[i]
							weight_lr_v9 = float(error_lr_v9)/fenmu
							result+=weight_lr_v9*final_list_v9[i]
							weight_lr_v10 = float(error_lr_v10)/fenmu
							result+=weight_lr_v10*final_list_v10[i]
							weight_lr_v24 = float(error_lr_v24)/fenmu
							result+=weight_lr_v24*final_list_v24[i]
						elif delete_number==7:
							fenmu = error_lr_v3+error_lr_v5+error_lr_v7+error_lr_v9+error_lr_v10+error_lr_v12
							weight_lr_v3 = float(error_lr_v3)/fenmu
							result+=weight_lr_v3*final_list_v3[i]
							weight_lr_v5 = float(error_lr_v5)/fenmu
							result+=weight_lr_v5*final_list_v5[i]
							weight_lr_v7 = float(error_lr_v7)/fenmu
							result+=weight_lr_v7*final_list_v7[i]
							weight_lr_v9 = float(error_lr_v9)/fenmu
							result+=weight_lr_v9*final_list_v9[i]
							weight_lr_v10 = float(error_lr_v10)/fenmu
							result+=weight_lr_v10*final_list_v10[i]
							weight_lr_v12 = float(error_lr_v12)/fenmu
							result+=weight_lr_v12*final_list_v12[i]	
						"""
						if delete_number==1:
							fenmu = 1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24
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
							weight_lr_v24 = float(1/error_lr_v24)/fenmu
							result+=weight_lr_v24*final_list_v24[i]																												
						elif delete_number==2:
							fenmu = 1/error_lr_v3+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24
							weight_lr_v3 = float(1/error_lr_v3)/fenmu
							result+=weight_lr_v3*final_list_v3[i]
							weight_lr_v7 = float(1/error_lr_v7)/fenmu
							result+=weight_lr_v7*final_list_v7[i]
							weight_lr_v9 = float(1/error_lr_v9)/fenmu
							result+=weight_lr_v9*final_list_v9[i]
							weight_lr_v10 = float(1/error_lr_v10)/fenmu
							result+=weight_lr_v10*final_list_v10[i]
							weight_lr_v12 = float(1/error_lr_v12)/fenmu
							result+=weight_lr_v12*final_list_v12[i]
							weight_lr_v24 = float(1/error_lr_v24)/fenmu
							result+=weight_lr_v24*final_list_v24[i]								
						elif delete_number==3:
							fenmu = 1/error_lr_v3+1/error_lr_v5+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24
							weight_lr_v3 = float(1/error_lr_v3)/fenmu
							result+=weight_lr_v3*final_list_v3[i]
							weight_lr_v5 = float(1/error_lr_v5)/fenmu
							result+=weight_lr_v5*final_list_v5[i]
							weight_lr_v9 = float(1/error_lr_v9)/fenmu
							result+=weight_lr_v9*final_list_v9[i]
							weight_lr_v10 = float(1/error_lr_v10)/fenmu
							result+=weight_lr_v10*final_list_v10[i]
							weight_lr_v12 = float(1/error_lr_v12)/fenmu
							result+=weight_lr_v12*final_list_v12[i]
							weight_lr_v24 = float(1/error_lr_v24)/fenmu
							result+=weight_lr_v24*final_list_v24[i]
						elif delete_number==4:
							fenmu = 1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24
							weight_lr_v3 = float(1/error_lr_v3)/fenmu
							result+=weight_lr_v3*final_list_v3[i]
							weight_lr_v5 = float(1/error_lr_v5)/fenmu
							result+=weight_lr_v5*final_list_v5[i]
							weight_lr_v7 = float(1/error_lr_v7)/fenmu
							result+=weight_lr_v7*final_list_v7[i]
							weight_lr_v10 = float(1/error_lr_v10)/fenmu
							result+=weight_lr_v10*final_list_v10[i]
							weight_lr_v12 = float(1/error_lr_v12)/fenmu
							result+=weight_lr_v12*final_list_v12[i]
							weight_lr_v24 = float(1/error_lr_v24)/fenmu
							result+=weight_lr_v24*final_list_v24[i]	
						elif delete_number==5:
							fenmu = 1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v12+1/error_lr_v24
							weight_lr_v3 = float(1/error_lr_v3)/fenmu
							result+=weight_lr_v3*final_list_v3[i]
							weight_lr_v5 = float(1/error_lr_v5)/fenmu
							result+=weight_lr_v5*final_list_v5[i]
							weight_lr_v7 = float(1/error_lr_v7)/fenmu
							result+=weight_lr_v7*final_list_v7[i]
							weight_lr_v9 = float(1/error_lr_v9)/fenmu
							result+=weight_lr_v9*final_list_v9[i]
							weight_lr_v12 = float(1/error_lr_v12)/fenmu
							result+=weight_lr_v12*final_list_v12[i]
							weight_lr_v24 = float(1/error_lr_v24)/fenmu
							result+=weight_lr_v24*final_list_v24[i]	
						elif delete_number==6:
							fenmu = 1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v24
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
							weight_lr_v24 = float(1/error_lr_v24)/fenmu
							result+=weight_lr_v24*final_list_v24[i]
						elif delete_number==7:
							fenmu = 1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12
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
						#error_lr_v7 = (np.sum(result_v7_error[length:length+window]))/window+1e-06

						#print(accuracy_list)
						#minor = accuracy_list[-1]
						#print(minor)
						#print(minor)
						#weight_xgb_v5_1 = result_v5_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v5)/(1/error_xgb_v5+1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
						#weight_xgb_v7_1 = result_v7_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v7)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
						#weight_xgb_v10_1 = result_v10_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v10)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
						#weight_lr_1 = result_lr_previous_weight_1[length]*beta+(1-beta)*float(1/error_lr)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
						#validate_value = []
						#fenmu=0
						#for accuracy in accuracy_list:
						#	if accuracy!=minor:
						#		validate_value.append(accuracy)
						#		fenmu += accuracy+1e-06

						#result = 0
						#if window == window_size:
						#	weight_lr_v7 = float(error_lr_v7)
						#	result+=weight_lr_v7*final_list_v7[i]
						#else:
						#	result = final_list_v7[i]
						#if error_lr_v3 in validate_value:
						#	weight_lr_v3 = float(error_lr_v3)/fenmu
						#	result+=weight_lr_v3*final_list_v3[i]
							#print("v3")
						#if error_lr_v5 in validate_value:
						#	weight_lr_v5 = float(error_lr_v5)/fenmu
						#	result+=weight_lr_v5*final_list_v5[i]
							#print("v5")
						#if error_lr_v7 in validate_value:
						#	weight_lr_v7 = float(error_lr_v7)/fenmu
						#	result+=weight_lr_v7*final_list_v7[i]
							#print("v7")
						#if error_lr_v9 in validate_value:
						#	weight_lr_v9 = float(error_lr_v9)/fenmu
						#	result+=weight_lr_v9*final_list_v9[i]
							#print("v9")
						#if error_lr_v10 in validate_value:
						#	weight_lr_v10 = float(error_lr_v10)/fenmu
						#	result+=weight_lr_v10*final_list_v10[i]
							#print("v10")
						#if error_lr_v12 in validate_value:
						#	weight_lr_v12 = float(error_lr_v12)/fenmu
						#	result+=weight_lr_v12*final_list_v12[i]
							#print("v12")
						#if error_lr_v24 in validate_value:
						#	weight_lr_v24 = float(error_lr_v24)/fenmu
						#	result+=weight_lr_v24*final_list_v24[i]
							#print("v24")							
						#result_v5_previous_weight_1.append(weight_xgb_v5_1)
						#result_v7_previous_weight_1.append(weight_xgb_v7_1)
						#result_v10_previous_weight_1.append(weight_xgb_v10_1)
						#result_lr_previous_weight_1.append(weight_lr_1)
						
						#result=0
						'''v5_item=1
						v7_item=1
						v10_item=1
						for item in v5_voting_prob_list[i]:
							v5_item*=item
						v5_item = math.pow(v5_item,1/len(v5_voting_prob_list[i]))
						for item in v7_voting_prob_list[i]:
							v7_item*=item
						v7_item = math.pow(v7_item,1/len(v7_voting_prob_list[i]))
						for item in v10_voting_prob_list[i]:
							v10_item*=item
						v10_item = math.pow(v10_item,1/len(v10_voting_prob_list[i]))'''
						"""
						result+=weight_lr_v3*final_list_v3[i]
						result+=weight_lr_v5*final_list_v5[i]
						result+=weight_lr_v7*final_list_v7[i]
						result+=weight_lr_v9*final_list_v9[i]
						result+=weight_lr_v10*final_list_v10[i]
						result+=weight_lr_v12*final_list_v12[i]
						result+=weight_lr_v24*final_list_v24[i]
						"""
						#result+=weight_xgb_v10_1*final_list_v10[i]
						#result+=weight_lr_1*result_lr[i]
						
						#probal.append(result)
						#print(result)
						
						if result>0.5:
							final_list_1.append(1)
						else:
							final_list_1.append(0)
						
						if window==window_size:
							length+=1
						else:
							window+=1
						
						#print(length)
					print("the length of the y_test is {}".format(len(final_list_1)))
					print("the weight ensebmle for V5 V7 LR weight voting beta precision is {}".format(metrics.accuracy_score(y_va, final_list_1)))
					print("the horizon is {}".format(horizon))
					print("the window size is {}".format(window_size))
					#print("the beta is {}".format(beta))
					print("the metal is {}".format(ground_truth))
					print("the test date is {}".format(date))					
					"""
					if len(result_v3_error)==0:

						for i in range(len(final_list_v3)):
							if y_va[i]!=final_list_v3[i]:
								result_v3_error.append(1)
							else:
								result_v3_error.append(0)

						for i in range(len(final_list_v5)):
							if y_va[i]!=final_list_v5[i]:
								result_v5_error.append(1)
							else:
								result_v5_error.append(0)

						for i in range(len(final_list_v7)):
							if y_va[i]!=final_list_v7[i]:
								result_v7_error.append(1)
							else:
								result_v7_error.append(0)

						for i in range(len(final_list_v9)):
							if y_va[i]!=final_list_v9[i]:
								result_v9_error.append(1)
							else:
								result_v9_error.append(0)

						for i in range(len(final_list_v10)):
							if y_va[i]!=final_list_v10[i]:
								result_v10_error.append(1)
							else:
								result_v10_error.append(0)

						for i in range(len(final_list_v12)):
							if y_va[i]!=final_list_v12[i]:
								result_v12_error.append(1)
							else:
								result_v12_error.append(0)

						for i in range(len(final_list_v24)):
							if y_va[i]!=final_list_v24[i]:
								result_v24_error.append(1)
							else:
								result_v24_error.append(0)


						
						final_list_1 = []
						final_list_2 = []

						true_result = []
						probal = []
	                    
	                    # we choose a specific window size to calculate the precision weight to ensemble the models results together
						for i in range(window_size+horizon-1,len(y_va)):
							true_result.append(y_va[i])
							error_lr_v3 = np.sum(result_v3_error[length:length+window_size])+1e-05
							error_lr_v5 = np.sum(result_v5_error[length:length+window_size])+1e-05
							error_lr_v7 = np.sum(result_v7_error[length:length+window_size])+1e-05
							error_lr_v9 = np.sum(result_v9_error[length:length+window_size])+1e-05
							error_lr_v10 = np.sum(result_v10_error[length:length+window_size])+1e-05
							error_lr_v12 = np.sum(result_v12_error[length:length+window_size])+1e-05
							error_lr_v24 = np.sum(result_v24_error[length:length+window_size])+1e-05
							
							
							#weight_xgb_v5_1 = result_v5_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v5)/(1/error_xgb_v5+1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							#weight_xgb_v7_1 = result_v7_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v7)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							#weight_xgb_v10_1 = result_v10_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v10)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							#weight_lr_1 = result_lr_previous_weight_1[length]*beta+(1-beta)*float(1/error_lr)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							weight_lr_v3 = float(1/error_lr_v3)/(1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24)
							weight_lr_v5 = float(1/error_lr_v5)/(1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24)
							weight_lr_v7 = float(1/error_lr_v7)/(1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24)
							weight_lr_v9 = float(1/error_lr_v9)/(1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24)
							weight_lr_v10 = float(1/error_lr_v10)/(1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24)
							weight_lr_v12 = float(1/error_lr_v12)/(1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24)
							weight_lr_v24 = float(1/error_lr_v24)/(1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24)							
							#result_v5_previous_weight_1.append(weight_xgb_v5_1)
							#result_v7_previous_weight_1.append(weight_xgb_v7_1)
							#result_v10_previous_weight_1.append(weight_xgb_v10_1)
							#result_lr_previous_weight_1.append(weight_lr_1)
							
							result=0
							'''v5_item=1
							v7_item=1
							v10_item=1
							for item in v5_voting_prob_list[i]:
								v5_item*=item
							v5_item = math.pow(v5_item,1/len(v5_voting_prob_list[i]))
							for item in v7_voting_prob_list[i]:
								v7_item*=item
							v7_item = math.pow(v7_item,1/len(v7_voting_prob_list[i]))
							for item in v10_voting_prob_list[i]:
								v10_item*=item
							v10_item = math.pow(v10_item,1/len(v10_voting_prob_list[i]))'''
							result+=weight_lr_v3*final_list_v3[i]
							result+=weight_lr_v5*final_list_v5[i]
							result+=weight_lr_v7*final_list_v7[i]
							result+=weight_lr_v9*final_list_v9[i]
							result+=weight_lr_v10*final_list_v10[i]
							result+=weight_lr_v12*final_list_v12[i]
							result+=weight_lr_v24*final_list_v24[i]
							#result+=weight_xgb_v10_1*final_list_v10[i]
							#result+=weight_lr_1*result_lr[i]
							
							probal.append(result)
							if result>0.5:
								final_list_1.append(1)
							else:
								final_list_1.append(0)

							length+=1
						print("the length of the y_test is {}".format(len(true_result)))
						print("the weight ensebmle for V5 V7 LR weight voting beta precision is {}".format(metrics.accuracy_score(true_result, final_list_1)))
						print("the horizon is {}".format(horizon))
						print("the window size is {}".format(window_size))
						#print("the beta is {}".format(beta))
						print("the metal is {}".format(ground_truth))
						print("the test date is {}".format(date))
					"""
					"""
					else:
						for i in range(len(final_list_v3)):
							if y_va[i]!=final_list_v3[i]:
								result_v3_error.append(1)
							else:
								result_v3_error.append(0)

						for i in range(len(final_list_v5)):
							if y_va[i]!=final_list_v5[i]:
								result_v5_error.append(1)
							else:
								result_v5_error.append(0)

						for i in range(len(final_list_v7)):
							if y_va[i]!=final_list_v7[i]:
								result_v7_error.append(1)
							else:
								result_v7_error.append(0)

						for i in range(len(final_list_v9)):
							if y_va[i]!=final_list_v9[i]:
								result_v9_error.append(1)
							else:
								result_v9_error.append(0)

						for i in range(len(final_list_v10)):
							if y_va[i]!=final_list_v10[i]:
								result_v10_error.append(1)
							else:
								result_v10_error.append(0)

						for i in range(len(final_list_v12)):
							if y_va[i]!=final_list_v12[i]:
								result_v12_error.append(1)
							else:
								result_v12_error.append(0)

						for i in range(len(final_list_v24)):
							if y_va[i]!=final_list_v24[i]:
								result_v24_error.append(1)
							else:
								result_v24_error.append(0)
						final_list_1 = []
						probal = []
	                    # the same as above
						for i in range(len(y_va)):
							error_lr_v3 = np.sum(result_v3_error[length:length+window_size])+1e-05
							error_lr_v5 = np.sum(result_v5_error[length:length+window_size])+1e-05
							error_lr_v7 = np.sum(result_v7_error[length:length+window_size])+1e-05
							error_lr_v9 = np.sum(result_v9_error[length:length+window_size])+1e-05
							error_lr_v10 = np.sum(result_v10_error[length:length+window_size])+1e-05
							error_lr_v12 = np.sum(result_v12_error[length:length+window_size])+1e-05
							error_lr_v24 = np.sum(result_v24_error[length:length+window_size])+1e-05
							
							
							#weight_xgb_v5_1 = result_v5_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v5)/(1/error_xgb_v5+1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							#weight_xgb_v7_1 = result_v7_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v7)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							#weight_xgb_v10_1 = result_v10_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v10)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							#weight_lr_1 = result_lr_previous_weight_1[length]*beta+(1-beta)*float(1/error_lr)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							weight_lr_v3 = float(1/error_lr_v3)/(1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24)
							weight_lr_v5 = float(1/error_lr_v5)/(1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24)
							weight_lr_v7 = float(1/error_lr_v7)/(1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24)
							weight_lr_v9 = float(1/error_lr_v9)/(1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24)
							weight_lr_v10 = float(1/error_lr_v10)/(1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24)
							weight_lr_v12 = float(1/error_lr_v12)/(1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24)
							weight_lr_v24 = float(1/error_lr_v24)/(1/error_lr_v3+1/error_lr_v5+1/error_lr_v7+1/error_lr_v9+1/error_lr_v10+1/error_lr_v12+1/error_lr_v24)							
							#result_v5_previous_weight_1.append(weight_xgb_v5_1)
							#result_v7_previous_weight_1.append(weight_xgb_v7_1)
							#result_v10_previous_weight_1.append(weight_xgb_v10_1)
							#result_lr_previous_weight_1.append(weight_lr_1)
							
							result=0
							'''v5_item=1
							v7_item=1
							v10_item=1
							for item in v5_voting_prob_list[i]:
								v5_item*=item
							v5_item = math.pow(v5_item,1/len(v5_voting_prob_list[i]))
							for item in v7_voting_prob_list[i]:
								v7_item*=item
							v7_item = math.pow(v7_item,1/len(v7_voting_prob_list[i]))
							for item in v10_voting_prob_list[i]:
								v10_item*=item
							v10_item = math.pow(v10_item,1/len(v10_voting_prob_list[i]))'''
							result+=weight_lr_v3*final_list_v3[i]
							result+=weight_lr_v5*final_list_v5[i]
							result+=weight_lr_v7*final_list_v7[i]
							result+=weight_lr_v9*final_list_v9[i]
							result+=weight_lr_v10*final_list_v10[i]
							result+=weight_lr_v12*final_list_v12[i]
							result+=weight_lr_v24*final_list_v24[i]
							#result+=weight_xgb_v10_1*final_list_v10[i]
							#result+=weight_lr_1*result_lr[i]
							
							probal.append(result)
							if result>0.5:
								final_list_1.append(1)
							else:
								final_list_1.append(0)
							length+=1
						print("the length of the y_test is {}".format(len(y_va)))
						print("the weight ensebmle for V5 V7 LR weight voting precision is {}".format(metrics.accuracy_score(y_va, final_list_1)))
						print("the horizon is {}".format(horizon))
						print("the window size is {}".format(window_size))
						#print("the beta is {}".format(beta))
						print("the metal is {}".format(ground_truth))
						print("the test date is {}".format(date))"""
