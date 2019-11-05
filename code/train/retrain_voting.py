import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
os.chdir("D://Next++//4E//Wanying//4EBaseMetal")
from sklearn import metrics
import math

for horizon in [1,3,5]:
	for ground_truth in ["LME_Co_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot","LME_Le_Spot"]:
		for window_size in [5,10,15,20,25,30]:
			#for beta in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
				result_v5_error = []
				result_v7_error = []
				result_v10_error = []
				result_lr_error = []
				result_lr_v10_error = []
				
				result_v5_previous_weight_1 = [0]
				result_v7_previous_weight_1 = [0]
				result_v10_previous_weight_1 = [0]
				result_lr_previous_weight_1 = [0]
				result_lr_v10_previous_weight_1 = [0]

				result_v5_previous_weight_2 = [0]
				result_v7_previous_weight_2 = [0]
				result_v10_previous_weight_2 = [0]
				result_lr_previous_weight_2 = [0]
				result_lr_v10_previous_weight_2 = [0]
				length=0
				#window_size = 5
				for date in ["2014-07-01","2015-01-02","2015-07-01","2016-01-04","2016-07-01","2017-01-03","2017-07-03","2018-01-02","2018-07-02"]:
					result_v5 = np.loadtxt("data/xgboost_probability/"+ground_truth+"_horizon_"+str(horizon)+"_"+date+"_"+"v5"+"_weight_4"+".txt")
					result_v7 = np.loadtxt("data/xgboost_probability/"+ground_truth+"_horizon_"+str(horizon)+"_"+date+"_"+"v7"+"_weight"+".txt")
					
					if ground_truth.split("_")[1]=="Co":
						result_v10 = np.loadtxt("data/xgboost_probability/"+"LMCADY"+"_"+"horizon_"+str(horizon)+"_"+date+"_v10"+"_striplag30_weight"+".txt")
					elif ground_truth.split("_")[1]=="Al":
						result_v10 = np.loadtxt("data/xgboost_probability/"+"LMAHDY"+"_"+"horizon_"+str(horizon)+"_"+date+"_v10"+"_striplag30_weight"+".txt")
					elif ground_truth.split("_")[1]=="Ni":
						result_v10 = np.loadtxt("data/xgboost_probability/"+"LMNIDY"+"_"+"horizon_"+str(horizon)+"_"+date+"_v10"+"_striplag30_weight"+".txt")
					elif ground_truth.split("_")[1]=="Ti":
						result_v10 = np.loadtxt("data/xgboost_probability/"+"LMSNDY"+"_"+"horizon_"+str(horizon)+"_"+date+"_v10"+"_striplag30_weight"+".txt")
					elif ground_truth.split("_")[1]=="Zi":
						result_v10 = np.loadtxt("data/xgboost_probability/"+"LMZSDY"+"_"+"horizon_"+str(horizon)+"_"+date+"_v10"+"_striplag30_weight"+".txt")
					elif ground_truth.split("_")[1]=="Le":
						result_v10 = np.loadtxt("data/xgboost_probability/"+"LMPBDY"+"_"+"horizon_"+str(horizon)+"_"+date+"_v10"+"_striplag30_weight"+".txt")
					
					if ground_truth.split("_")[1]=="Co":
						if date>='2017-01-03':
							LR_v5 = pd.read_csv("data/LR_probability/"+'LMCADY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+date+".csv")
						else:
							LR_v5 = pd.read_csv("data/LR_probability/"+'LMCADY'+"_h"+str(horizon)+"_v5probh"+str(horizon)+date+".csv")
					elif ground_truth.split("_")[1]=="Al":
						if date>='2017-01-03':
							LR_v5 = pd.read_csv("data/LR_probability/"+'LMAHDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+date+".csv")
						else:
							LR_v5 = pd.read_csv("data/LR_probability/"+'LMAHDY'+"_h"+str(horizon)+"_v5probh"+str(horizon)+date+".csv")
					elif ground_truth.split("_")[1]=="Ni":
						if date>='2017-01-03':
							LR_v5 = pd.read_csv("data/LR_probability/"+'LMNIDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+date+".csv")
						else:
							LR_v5 = pd.read_csv("data/LR_probability/"+'LMNIDY'+"_h"+str(horizon)+"_v5probh"+str(horizon)+date+".csv")
					elif ground_truth.split("_")[1]=="Ti":
						if date>='2017-01-03':
							LR_v5 = pd.read_csv("data/LR_probability/"+'LMSNDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+date+".csv")
						else:
							LR_v5 = pd.read_csv("data/LR_probability/"+'LMSNDY'+"_h"+str(horizon)+"_v5probh"+str(horizon)+date+".csv")
					elif ground_truth.split("_")[1]=="Zi":
						if date>='2017-01-03':
							LR_v5 = pd.read_csv("data/LR_probability/"+'LMZSDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+date+".csv")
						else:
							LR_v5 = pd.read_csv("data/LR_probability/"+'LMZSDY'+"_h"+str(horizon)+"_v5probh"+str(horizon)+date+".csv")
					elif ground_truth.split("_")[1]=="Le":
						if date>='2017-01-03':
							LR_v5 = pd.read_csv("data/LR_probability/"+'LMPBDY'+"_h"+str(horizon)+"_v5_probh"+str(horizon)+date+".csv")
						else:
							LR_v5 = pd.read_csv("data/LR_probability/"+'LMPBDY'+"_h"+str(horizon)+"_v5probh"+str(horizon)+date+".csv")
					result_lr = list(LR_v5['Prediction'])


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

					final_list_v5 = []
					v5_voting_prob_list=[]
					for j in range(len(result_v5)):
						count_1=0
						count_0=0
						pos_list = []
						neg_list = []
						for item in result_v5[j]:
							if item > 0.5:
								pos_list.append(item)
								count_1+=1
							else:
								neg_list.append(item)
								count_0+=1
						if count_1>count_0:
							v5_voting_prob_list.append(pos_list)
							final_list_v5.append(1)
						else:
							v5_voting_prob_list.append(neg_list)
							final_list_v5.append(0)
					
					final_list_v10 = []
					v10_voting_prob_list=[]
					for j in range(len(result_v10)):
						count_1=0
						count_0=0
						pos_list = []
						neg_list = []
						for item in result_v10[j]:
							if item > 0.5:
								pos_list.append(item)
								count_1+=1
							else:
								neg_list.append(item)
								count_0+=1
						if count_1>count_0:
							v10_voting_prob_list.append(pos_list)
							final_list_v10.append(1)
						else:
							v10_voting_prob_list.append(neg_list)
							final_list_v10.append(0)                   

					final_list_v7=[]
					v7_voting_prob_list=[]
					for j in range(len(result_v7)):
						count_1=0
						count_0=0
						pos_list = []
						neg_list = []
						for item in result_v7[j]:
							if item > 0.5:
								pos_list.append(item)
								count_1+=1
							else:
								neg_list.append(item)
								count_0+=1
						if count_1>count_0:
							v7_voting_prob_list.append(pos_list)
							final_list_v7.append(1)
						else:
							v7_voting_prob_list.append(neg_list)
							final_list_v7.append(0)
					#print("the length of the y_test is {}".format(len(y_va)))
					
					if len(result_v5_error)==0:
						for i in range(len(result_v5)):
							count_1=0
							count_0=0
							for item in result_v5[i]:
								if item>0.5:
									count_1+=1
								else:
									count_0+=1
							if count_1>count_0:
								result=1
							else:
								result=0
							if y_va[i]!=result:
								result_v5_error.append(1)
							else:
								result_v5_error.append(0)
							count_1=0
							count_0=0
							for item in result_v7[i]:
								if item > 0.5:
									count_1+=1
								else:
									count_0+=1
							if count_1>count_0:
								result=1
							else:
								result=0
							if y_va[i]!=result:
								result_v7_error.append(1)
							else:
								result_v7_error.append(0)
							count_1=0
							count_0=0
							for item in result_v10[i]:
								if item > 0.5:
									count_1+=1
								else:
									count_0+=1
							if count_1>count_0:
								result=1
							else:
								result=0
							if y_va[i]!=result:
								result_v10_error.append(1)
							else:
								result_v10_error.append(0)
							if result_lr[i] > 0.5:
								result=1
							else:
								result=0
							if y_va[i]!=result:
								result_lr_error.append(1)
							else:
								result_lr_error.append(0)
						final_list_1 = []
						final_list_2 = []

						true_result = []
						probal = []
	                    
	                    # we choose a specific window size to calculate the precision weight to ensemble the models results together
						for i in range(window_size+horizon-1,len(y_va)):
							true_result.append(y_va[i])
							error_xgb_v5 = np.sum(result_v5_error[length:length+window_size])+1e-05
							error_xgb_v7 = np.sum(result_v7_error[length:length+window_size])+1e-05
							error_xgb_v10 = np.sum(result_v10_error[length:length+window_size])+1e-05
							error_lr = np.sum(result_lr_error[length:length+window_size])+1e-05
							
							
							#weight_xgb_v5_1 = result_v5_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v5)/(1/error_xgb_v5+1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							#weight_xgb_v7_1 = result_v7_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v7)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							#weight_xgb_v10_1 = result_v10_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v10)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							#weight_lr_1 = result_lr_previous_weight_1[length]*beta+(1-beta)*float(1/error_lr)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							weight_xgb_v7_1 = float(1/error_xgb_v7)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							weight_xgb_v10_1 = float(1/error_xgb_v10)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							weight_lr_1 = float(1/error_lr)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)							
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
							result+=weight_xgb_v7_1*final_list_v7[i]
							result+=weight_xgb_v10_1*final_list_v10[i]
							result+=weight_lr_1*result_lr[i]
							
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
					else:
						for i in range(len(result_v5)):
							count_1=0
							count_0=0
							for item in result_v5[i]:
								if item>0.5:
									count_1+=1
								else:
									count_0+=1
							if count_1>count_0:
								result=1
							else:
								result=0
							if y_va[i]!=result:
								result_v5_error.append(1)
							else:
								result_v5_error.append(0)
							count_1=0
							count_0=0
							for item in result_v7[i]:
								if item > 0.5:
									count_1+=1
								else:
									count_0+=1
							if count_1>count_0:
								result=1
							else:
								result=0
							if y_va[i]!=result:
								result_v7_error.append(1)
							else:
								result_v7_error.append(0)
							count_1=0
							count_0=0
							for item in result_v10[i]:
								if item > 0.5:
									count_1+=1
								else:
									count_0+=1
							if count_1>count_0:
								result=1
							else:
								result=0
							if y_va[i]!=result:
								result_v10_error.append(1)
							else:
								result_v10_error.append(0)
							if result_lr[i] > 0.5:
								result=1
							else:
								result=0
							if y_va[i]!=result:
								result_lr_error.append(1)
							else:
								result_lr_error.append(0)
						final_list_1 = []
						probal = []
	                    # the same as above
						for i in range(len(y_va)):
							error_xgb_v5 = np.sum(result_v5_error[length:length+window_size])+1e-05
							error_xgb_v7 = np.sum(result_v7_error[length:length+window_size])+1e-05
							error_xgb_v10 = np.sum(result_v10_error[length:length+window_size])+1e-05
							error_lr = np.sum(result_lr_error[length:length+window_size])+1e-05
							
							
							#weight_xgb_v5_1 = result_v5_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v5)/(1/error_xgb_v5+1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							#weight_xgb_v7_1 = result_v7_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v7)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							#weight_xgb_v10_1 = result_v10_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v10)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							#weight_lr_1 = result_lr_previous_weight_1[length]*beta+(1-beta)*float(1/error_lr)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)

							#weight_xgb_v5_1 = result_v5_previous_weight_1[length]*beta+(1-beta)*float(1/error_xgb_v5)/(1/error_xgb_v5+1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							weight_xgb_v7_1 = float(1/error_xgb_v7)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							weight_xgb_v10_1 = float(1/error_xgb_v10)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							weight_lr_1 = float(1/error_lr)/(1/error_xgb_v7+1/error_xgb_v10+1/error_lr)
							
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
							result+=weight_xgb_v7_1*final_list_v7[i]
							result+=weight_xgb_v10_1*final_list_v10[i]
							result+=weight_lr_1*result_lr[i]
							
							probal.append(result)
							#probal.append(result)
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
						print("the test date is {}".format(date))