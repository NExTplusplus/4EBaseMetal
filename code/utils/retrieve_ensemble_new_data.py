import pandas as pd
import numpy as np
path_list = ['ensemble_Co_h1_renewv5_weight_window_5.txt','ensemble_Al_h1_renewv5_weight_window_5.txt','ensemble_Zi_h1_renewv5_weight_window_5.txt','ensemble_Ti_h1_renewv5_weight_window_5.txt','ensemble_Ni_h1_renewv5_weight_window_5.txt',
'ensemble_Le_h1_renewv5_weight_window_5.txt','ensemble_Co_h1_renewv5_weight_window_10.txt','ensemble_Al_h1_renewv5_weight_window_10.txt','ensemble_Zi_h1_renewv5_weight_window_10.txt','ensemble_Ti_h1_renewv5_weight_window_10.txt','ensemble_Ni_h1_renewv5_weight_window_10.txt',
'ensemble_Le_h1_renewv5_weight_window_10.txt','ensemble_Co_h1_renewv5_weight_window_15.txt','ensemble_Al_h1_renewv5_weight_window_15.txt','ensemble_Zi_h1_renewv5_weight_window_15.txt','ensemble_Ti_h1_renewv5_weight_window_15.txt','ensemble_Ni_h1_renewv5_weight_window_15.txt',
'ensemble_Le_h1_renewv5_weight_window_15.txt']
#all_voting_list = []
#average_list = []
all_ensemble_result = []
result_list = []
for new_path in path_list:
	new_path_list = new_path.split("_")
	metal = new_path_list[1]
	window_size = new_path_list[-1]
	ensemble_str = 'the y_va length is'
	window
	with open(new_path,"r") as f:
		lines = f.readlines()
		j=1
		ensemble_result_list = [0]*11
		average_sub_list[0]=metal
		average_sub_list[10]=window_size
		#rank_sub_list = [0]*22
		#average_sub_list[21]=horizon
		#average_sub_list[0]=metal
		#rank_sub_list[0]=metal
		#rank_sub_list[21]=horizon
		#sub_voting_list=[0]*7
		#sub_voting_list[0]=metal
		#sub_voting_list[6]=horizon
		for i,line in enumerate(lines):
			if ensemble_str.lower() in line.lower():
				ensemble_result_list[j]=float(line.strip("\n").split(" ")[-1])
				#average_sub_list[j]=float(lines[i+1].strip("\n").split(" ")[-1])
				#average_sub_list[j+5]=float(lines[i+3].strip("\n").split(" ")[-1])
				#average_sub_list[j+10]=float(lines[i+4].strip("\n").split(" ")[-1])
				#average_sub_list[j+15]=float(lines[i+5].strip("\n").split(" ")[-1])
				#rank_sub_list[j]=float(lines[i+6].strip("\n").split(" ")[-1])
				#rank_sub_list[j+5]=float(lines[i+7].strip("\n").split(" ")[-1])
				#rank_sub_list[j+10]=float(lines[i+8].strip("\n").split(" ")[-1])
				#rank_sub_list[j+15]=float(lines[i+9].strip("\n").split(" ")[-1])
				j+=1
		#sub_voting_list[5]=np.mean(sub_voting_list[1:5])
		#all_voting_list.append(sub_voting_list)
		#average_sub_list[5]=np.mean(average_sub_list[1:5])
		#average_sub_list[10]=np.mean(average_sub_list[6:10])
		#average_sub_list[15]=np.mean(average_sub_list[11:15])
		#average_sub_list[20]=np.mean(average_sub_list[16:20])
		#average_list.append(average_sub_list)
		#rank_sub_list[5]=np.mean(rank_sub_list[1:5])
		#rank_sub_list[10]=np.mean(rank_sub_list[6:10])
		#rank_sub_list[15]=np.mean(rank_sub_list[11:15])
		#rank_sub_list[20]=np.mean(rank_sub_list[16:20])
		#rank_list.append(rank_sub_list)
		all_ensemble_result.append(ensemble_result_list)
all_ensemble_frame = pd.DataFrame(data=all_ensemble_result,columns=['metal','2014-07-01',
	'2015-01-01','2015-07-01','2016-01-01','2016-07-01','2017-01-01','2017-07-01','2018-01-01','2018-07-01','horizon'])
all_ensemble_frame.to_csv("weight_ensemble_v2_renew.csv",index=False)
#average_frame = pd.DataFrame(data=average_list,columns=['metal','2017-01-01',
#	'2017-07-01','2018-01-01','2018-07-01','V5-V7-V10-LR','2017-01-01','2017-07-01',
#	'2018-01-01','2018-07-01','V5-V7-LR','2017-01-01','2017-07-01','2018-01-01','2018-07-01','V5-V10-LR',
#	'2017-01-01','2017-07-01','2018-01-01','2018-07-01','V7-V10-LR','horizon'])
#average_frame.to_csv("ensemble_average_v2_renew.csv",index=False)
#rank_frame = pd.DataFrame(data=rank_list,columns=['metal','2017-01-01',
#	'2017-07-01','2018-01-01','2018-07-01','V5-V7-V10-LR','2017-01-01','2017-07-01',
#	'2018-01-01','2018-07-01','V5-V7-LR','2017-01-01','2017-07-01','2018-01-01','2018-07-01','V7-V10-LR',
#	'2017-01-01','2017-07-01','2018-01-01','2018-07-01','V5-V10-LR','horizon'])
#rank_frame.to_csv("ensemble_rank_v2_renew.csv",index=False)