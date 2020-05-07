import pandas as pd
import numpy as np
from copy import copy
path_list = ['ensemble.txt']
#all_voting_list = []
#average_list = []
all_ensemble_result = []
all_ensemble_voting = []
all_ensemble_voting_weight = []
#all_lr_result = []
#all_XGB_V5_result = []
#all_XGB_V7_result = []
#all_XGB_V10_result = []
#all_voting_result = []
#all_rank_result = []
#all_rank_weight_result = []
result_list = []
for new_path in path_list:
	ensemble_str = 'the voting result is'
	with open(new_path,"r") as f:
		lines = f.readlines()
		j=1
		#lr_result = [0]*23
		#XGB_V5_result = [0]*23
		#XGB_V7_result = [0]*23
		#XGB_V10_result = [0]*23
		#voting_result = [0]*23
		#rank_result = [0]*23
		#rank_weight_result = [0]*23
		voting_result_list = [0]*23
		voting_weight_result_list = [0]*23
		#ensemble_result_list[0]=metal
		#ensemble_result_list[22]=horizon
		for i,line in enumerate(lines):
			if ensemble_str.lower() in line.lower():
				if j <= 18:
					voting_result_list[j]=float(lines[i].strip("\n").split(" ")[-1])
					voting_weight_result_list[j]=float(lines[i+2].strip("\n").split(" ")[-1])
					j+=1
					voting_result_list[j]=float(lines[i+1].strip("\n").split(" ")[-1])
					voting_weight_result_list[j]=float(lines[i+1].strip("\n").split(" ")[-1])
					voting_result_list[22]=float(lines[i+4].strip("\n").split(" ")[-1])
					voting_weight_result_list[22]=float(lines[i+4].strip("\n").split(" ")[-1])
					voting_result_list[21]=float(lines[i+3].strip("\n").split(" ")[-1])
					voting_weight_result_list[21]=float(lines[i+3].strip("\n").split(" ")[-1])
					voting_result_list[0]=lines[i+5].strip("\n").split(" ")[-1]
					voting_weight_result_list[0]=lines[i+5].strip("\n").split(" ")[-1]
					j+=1
					#ensemble_result_list[j]=float(lines[i+1].strip("\n").split(" ")[-1])
					#lr_result[j]=float(lines[i+1].strip("\n").split(" ")[-1])
					#XGB_V5_result[j]=float(lines[i+2].strip("\n").split(" ")[-1])
					#XGB_V7_result[j]=float(lines[i+3].strip("\n").split(" ")[-1])
					#XGB_V10_result[j]=float(lines[i+4].strip("\n").split(" ")[-1])
					#voting_result[j]=float(lines[i+5].strip("\n").split(" ")[-1])
					#rank_result[j]=float(lines[i+6].strip("\n").split(" ")[-1])
					#rank_weight_result[j]=float(lines[i+7].strip("\n").split(" ")[-1])
					#j+=1
					#ensemble_result_list[j]=float(lines[i].strip("\n").split(" ")[-1])
					#lr_result[j]=float(lines[i].strip("\n").split(" ")[-1])
					#XGB_V5_result[j]=float(lines[i].strip("\n").split(" ")[-1])
					#XGB_V7_result[j]=float(lines[i].strip("\n").split(" ")[-1])
					#XGB_V10_result[j]=float(lines[i].strip("\n").split(" ")[-1])
					#voting_result[j]=float(lines[i].strip("\n").split(" ")[-1])
					#rank_result[j]=float(lines[i].strip("\n").split(" ")[-1])
					#rank_weight_result[j]=float(lines[i].strip("\n").split(" ")[-1])
					#j+=1
					#print(j)
					#lr_result[22]=lines[i+8].strip("\n").split(" ")[-1]
					#XGB_V5_result[22]=lines[i+8].strip("\n").split(" ")[-1]
					#XGB_V7_result[22]=lines[i+8].strip("\n").split(" ")[-1]
					#XGB_V10_result[22]=lines[i+8].strip("\n").split(" ")[-1]
					#voting_result[22]=lines[i+8].strip("\n").split(" ")[-1]
					#rank_result[22]=lines[i+8].strip("\n").split(" ")[-1]
					#rank_weight_result[22]=lines[i+8].strip("\n").split(" ")[-1]
					#ensemble_result_list[22]=lines[i+2].strip("\n").split(" ")[-1]
					#ensemble_result_list[23]=lines[i+3].strip("\n").split(" ")[-1]
					#ensemble_result_list[24]=lines[i+4].strip("\n").split(" ")[-1]
					#ensemble_result_list[0]=lines[i+4].strip("\n").split(" ")[-1]
					#lr_result[0]=lines[i+9].strip("\n").split(" ")[-1]
					#XGB_V5_result[0]=lines[i+9].strip("\n").split(" ")[-1]
					#XGB_V7_result[0]=lines[i+9].strip("\n").split(" ")[-1]
					#XGB_V10_result[0]=lines[i+9].strip("\n").split(" ")[-1]
					#voting_result[0]=lines[i+9].strip("\n").split(" ")[-1]
					#rank_result[0]=lines[i+9].strip("\n").split(" ")[-1]
					#rank_weight_result[0]=lines[i+9].strip("\n").split(" ")[-1]
				else:
					all_ensemble_voting.append(voting_result_list)
					all_ensemble_voting_weight.append(voting_weight_result_list)
					voting_result_list=[0]*23
					voting_weight_result_list=[0]*23
					j=1
					#print(lines[i].strip("\n").split(" ")[-1])
					voting_result_list[j]=float(lines[i].strip("\n").split(" ")[-1])
					voting_weight_result_list[j]=float(lines[i+2].strip("\n").split(" ")[-1])
					j+=1
					voting_result_list[j]=float(lines[i+1].strip("\n").split(" ")[-1])
					voting_weight_result_list[j]=float(lines[i+1].strip("\n").split(" ")[-1])
					voting_result_list[22]=float(lines[i+4].strip("\n").split(" ")[-1])
					voting_weight_result_list[22]=float(lines[i+4].strip("\n").split(" ")[-1])
					voting_result_list[21]=float(lines[i+3].strip("\n").split(" ")[-1])
					voting_weight_result_list[21]=float(lines[i+3].strip("\n").split(" ")[-1])
					voting_result_list[0]=lines[i+5].strip("\n").split(" ")[-1]
					voting_weight_result_list[0]=lines[i+5].strip("\n").split(" ")[-1]
					j+=1					#all_lr_result.append(lr_result)
					#all_XGB_V5_result.append(XGB_V5_result)
					#all_XGB_V7_result.append(XGB_V7_result)
					#all_XGB_V10_result.append(XGB_V10_result)
					#all_voting_result.append(voting_result)
					#all_rank_result.append(rank_result)
					#all_rank_weight_result.append(rank_weight_result)
					#all_ensemble_result.append(ensemble_result_list)
					#j=1
					#ensemble_result_list = [0]*24
					#ensemble_result_list[j]=float(lines[i+1].strip("\n").split(" ")[-1])
					#lr_result[j]=float(lines[i+1].strip("\n").split(" ")[-1])
					#XGB_V5_result[j]=float(lines[i+2].strip("\n").split(" ")[-1])
					#XGB_V7_result[j]=float(lines[i+3].strip("\n").split(" ")[-1])
					#XGB_V10_result[j]=float(lines[i+4].strip("\n").split(" ")[-1])
					#voting_result[j]=float(lines[i+5].strip("\n").split(" ")[-1])
					#rank_result[j]=float(lines[i+6].strip("\n").split(" ")[-1])
					#rank_weight_result[j]=float(lines[i+7].strip("\n").split(" ")[-1])
					#j+=1
					#ensemble_result_list[j]=float(lines[i].strip("\n").split(" ")[-1])
					#lr_result[j]=float(lines[i].strip("\n").split(" ")[-1])
					#XGB_V5_result[j]=float(lines[i].strip("\n").split(" ")[-1])
					#XGB_V7_result[j]=float(lines[i].strip("\n").split(" ")[-1])
					#XGB_V10_result[j]=float(lines[i].strip("\n").split(" ")[-1])
					#voting_result[j]=float(lines[i].strip("\n").split(" ")[-1])
					#rank_result[j]=float(lines[i].strip("\n").split(" ")[-1])
					#rank_weight_result[j]=float(lines[i].strip("\n").split(" ")[-1])
					#j+=1
					#print(j)
					#lr_result[22]=lines[i+8].strip("\n").split(" ")[-1]
					#XGB_V5_result[22]=lines[i+8].strip("\n").split(" ")[-1]
					#XGB_V7_result[22]=lines[i+8].strip("\n").split(" ")[-1]
					#XGB_V10_result[22]=lines[i+8].strip("\n").split(" ")[-1]
					#voting_result[22]=lines[i+8].strip("\n").split(" ")[-1]
					#rank_result[22]=lines[i+8].strip("\n").split(" ")[-1]
					#rank_weight_result[22]=lines[i+8].strip("\n").split(" ")[-1]
					#ensemble_result_list[22]=lines[i+2].strip("\n").split(" ")[-1]
					#ensemble_result_list[23]=lines[i+3].strip("\n").split(" ")[-1]
					#ensemble_result_list[24]=lines[i+4].strip("\n").split(" ")[-1]
					#ensemble_result_list[0]=lines[i+4].strip("\n").split(" ")[-1]
					#lr_result[0]=lines[i+9].strip("\n").split(" ")[-1]
					#XGB_V5_result[0]=lines[i+9].strip("\n").split(" ")[-1]
					#XGB_V7_result[0]=lines[i+9].strip("\n").split(" ")[-1]
					#XGB_V10_result[0]=lines[i+9].strip("\n").split(" ")[-1]
					#voting_result[0]=lines[i+9].strip("\n").split(" ")[-1]
					#rank_result[0]=lines[i+9].strip("\n").split(" ")[-1]
					#rank_weight_result[0]=lines[i+9].strip("\n").split(" ")[-1]
		#all_lr_result.append(lr_result)
		#all_XGB_V5_result.append(XGB_V5_result)
		#all_XGB_V7_result.append(XGB_V7_result)
		#all_XGB_V10_result.append(XGB_V10_result)
		#all_voting_result.append(voting_result)
		#all_rank_result.append(rank_result)
		#all_rank_weight_result.append(rank_weight_result)
		#all_ensemble_result.append(ensemble_result_list)
		all_ensemble_voting.append(voting_result_list)
		all_ensemble_voting_weight.append(voting_weight_result_list)		
	for item in all_ensemble_voting:
		item[19]=float(item[1]*(item[2])+item[3]*item[4]+item[5]*item[6]+item[7]*item[8]+item[9]*item[10])/float(item[2]+item[4]+item[6]+item[8]+item[10])
		#item[20]=float(item[7]*item[8]+item[9]*item[10])/float(item[8]+item[10])
		item[20]=float(item[11]*item[12]+item[13]*item[14]+item[15]*item[16]+item[17]*item[18])/float(item[12]+item[14]+item[16]+item[18])
	for item in all_ensemble_voting_weight:
		item[19]=float(item[1]*item[2]+item[3]*item[4]+item[5]*item[6]+item[7]*item[8]+item[9]*item[10])/float(item[2]+item[4]+item[6]+item[8]+item[10])
		#item[20]=float(item[7]*item[8]+item[9]*item[10])/float(item[8]+item[10])
		item[20]=float(item[11]*item[12]+item[13]*item[14]+item[15]*item[16]+item[17]*item[18])/float(item[12]+item[14]+item[16]+item[18])
	#for item in all_XGB_V5_result:
	#	item[19]=float(item[1]*item[2]+item[3]*item[4]+item[5]*item[6]+item[7]*item[8]+item[9]*item[10])/float(item[2]+item[4]+item[6]+item[8]+item[10])
	#	item[20]=float(item[7]*item[8]+item[9]*item[10])/float(item[8]+item[10])
	#	item[21]=float(item[11]*item[12]+item[13]*item[14]+item[15]*item[16]+item[17]*item[18])/float(item[12]+item[14]+item[16]+item[18])
	#for item in all_XGB_V7_result:
	#	item[19]=float(item[1]*item[2]+item[3]*item[4]+item[5]*item[6]+item[7]*item[8]+item[9]*item[10])/float(item[2]+item[4]+item[6]+item[8]+item[10])
	#	item[20]=float(item[7]*item[8]+item[9]*item[10])/float(item[8]+item[10])
	#	item[21]=float(item[11]*item[12]+item[13]*item[14]+item[15]*item[16]+item[17]*item[18])/float(item[12]+item[14]+item[16]+item[18])
	#for item in all_XGB_V10_result:
	#	item[19]=float(item[1]*item[2]+item[3]*item[4]+item[5]*item[6]+item[7]*item[8]+item[9]*item[10])/float(item[2]+item[4]+item[6]+item[8]+item[10])
	#	item[20]=float(item[7]*item[8]+item[9]*item[10])/float(item[8]+item[10])
	#	item[21]=float(item[11]*item[12]+item[13]*item[14]+item[15]*item[16]+item[17]*item[18])/float(item[12]+item[14]+item[16]+item[18])
	#for item in all_voting_result:
	#	item[19]=float(item[1]*item[2]+item[3]*item[4]+item[5]*item[6]+item[7]*item[8]+item[9]*item[10])/float(item[2]+item[4]+item[6]+item[8]+item[10])
	#	item[20]=float(item[7]*item[8]+item[9]*item[10])/float(item[8]+item[10])
	#	item[21]=float(item[11]*item[12]+item[13]*item[14]+item[15]*item[16]+item[17]*item[18])/float(item[12]+item[14]+item[16]+item[18])
	#for item in all_rank_result:
	#	item[19]=float(item[1]*item[2]+item[3]*item[4]+item[5]*item[6]+item[7]*item[8]+item[9]*item[10])/float(item[2]+item[4]+item[6]+item[8]+item[10])
	#	item[20]=float(item[7]*item[8]+item[9]*item[10])/float(item[8]+item[10])
	#	item[21]=float(item[11]*item[12]+item[13]*item[14]+item[15]*item[16]+item[17]*item[18])/float(item[12]+item[14]+item[16]+item[18])
	#for item in all_rank_weight_result:
	#	item[19]=float(item[1]*item[2]+item[3]*item[4]+item[5]*item[6]+item[7]*item[8]+item[9]*item[10])/float(item[2]+item[4]+item[6]+item[8]+item[10])
	#	item[20]=float(item[7]*item[8]+item[9]*item[10])/float(item[8]+item[10])
	#	item[21]=float(item[11]*item[12]+item[13]*item[14]+item[15]*item[16]+item[17]*item[18])/float(item[12]+item[14]+item[16]+item[18])
	voting_frame = pd.DataFrame(data=all_ensemble_voting,columns=['metal','2014-07-01',"length",
		'2015-01-01',"length",'2015-07-01',"length",'2016-01-01',"length",'2016-07-01',"length",'2017-01-01',"length",'2017-07-01',"length",'2018-01-01',"length",'2018-07-01',"length",'validation','test','horizon','window_size'])
	voting_weight_frame = pd.DataFrame(data=all_ensemble_voting_weight,columns=['metal','2014-07-01',"length",
		'2015-01-01',"length",'2015-07-01',"length",'2016-01-01',"length",'2016-07-01',"length",'2017-01-01',"length",'2017-07-01',"length",'2018-01-01',"length",'2018-07-01',"length",'validation','test','horizon','window_size'])
	voting_frame = voting_frame.sort_values(by=['horizon','window_size'],ascending=(True, True))
	voting_weight_frame = voting_weight_frame.sort_values(by=['horizon','window_size'],ascending=(True, True))
	voting = np.array(voting_frame)
	voting_weight = np.array(voting_weight_frame)
	voting = list(voting)
	voting_weight = list(voting_weight)
	for i, item in enumerate(copy(voting)):
		if (i+1)%6==0:
			new_array = [0]*23
			new_array[0]="average"
			j = 0
			#while j<=16:
			#	j+=1
			#	new_array[j]=float((ensemble[i-5][j]+ensemble[i-4][j]+ensemble[i-3][j]+ensemble[i-2][j]+ensemble[i-1][j]+ensemble[i][j]))/6
			#	j+=1
			#	new_array[j]=ensemble[i][j]
				#j+=1
			new_array[19]=float((voting[i-5][19]+voting[i-4][19]+voting[i-3][19]+voting[i-2][19]+voting[i-1][19]+voting[i][19]))/6
			new_array[20]=float((voting[i-5][20]+voting[i-4][20]+voting[i-3][20]+voting[i-2][20]+voting[i-1][20]+voting[i][20]))/6
			#new_array[21]=float((voting[i-5][21]+voting[i-4][21]+voting[i-3][21]+voting[i-2][21]+voting[i-1][21]+voting[i][21]))/6
			new_array[21]=voting[i][21]
			new_array[22]=voting[i][22]
			#new_array[24]=ensemble[i][24]
			voting.append(new_array)
	for i, item in enumerate(copy(voting_weight)):
		if (i+1)%6==0:
			new_array = [0]*23
			new_array[0]="average"
			j = 0
			#while j<=16:
			#	j+=1
			#	new_array[j]=float((ensemble[i-5][j]+ensemble[i-4][j]+ensemble[i-3][j]+ensemble[i-2][j]+ensemble[i-1][j]+ensemble[i][j]))/6
			#	j+=1
			#	new_array[j]=ensemble[i][j]
				#j+=1
			new_array[19]=float((voting_weight[i-5][19]+voting_weight[i-4][19]+voting_weight[i-3][19]+voting_weight[i-2][19]+voting_weight[i-1][19]+voting_weight[i][19]))/6
			new_array[20]=float((voting_weight[i-5][20]+voting_weight[i-4][20]+voting_weight[i-3][20]+voting_weight[i-2][20]+voting_weight[i-1][20]+voting_weight[i][20]))/6
			#new_array[21]=float((voting_weight[i-5][21]+voting_weight[i-4][21]+voting_weight[i-3][21]+voting_weight[i-2][21]+voting_weight[i-1][21]+voting_weight[i][21]))/6
			new_array[21]=voting_weight[i][21]
			new_array[22]=voting_weight[i][22]
			#new_array[24]=ensemble[i][24]
			voting_weight.append(new_array)
	voting_frame = pd.DataFrame(data=voting,columns=['metal','2014-07-01',"length",
		'2015-01-01',"length",'2015-07-01',"length",'2016-01-01',"length",'2016-07-01',"length",'2017-01-01',"length",'2017-07-01',"length",'2018-01-01',"length",'2018-07-01',"length",'validation','test','horizon','window_size'])
	voting_weight_frame = pd.DataFrame(data=voting_weight,columns=['metal','2014-07-01',"length",
		'2015-01-01',"length",'2015-07-01',"length",'2016-01-01',"length",'2016-07-01',"length",'2017-01-01',"length",'2017-07-01',"length",'2018-01-01',"length",'2018-07-01',"length",'validation','test','horizon','window_size'])
	#lr_ensemble_frame = pd.DataFrame(data=all_lr_result,columns=['metal','2014-07-01',"length",
	#	'2015-01-01',"length",'2015-07-01',"length",'2016-01-01',"length",'2016-07-01',"length",'2017-01-01',"length",'2017-07-01',"length",'2018-01-01',"length",'2018-07-01',"length",'validation','validation_2016','test','horizon'])
	#XGB_V5_ensemble_frame = pd.DataFrame(data=all_XGB_V5_result,columns=['metal','2014-07-01',"length",
	#	'2015-01-01',"length",'2015-07-01',"length",'2016-01-01',"length",'2016-07-01',"length",'2017-01-01',"length",'2017-07-01',"length",'2018-01-01',"length",'2018-07-01',"length",'validation','validation_2016','test','horizon'])
	#XGB_V7_ensemble_frame = pd.DataFrame(data=all_XGB_V7_result,columns=['metal','2014-07-01',"length",
	#	'2015-01-01',"length",'2015-07-01',"length",'2016-01-01',"length",'2016-07-01',"length",'2017-01-01',"length",'2017-07-01',"length",'2018-01-01',"length",'2018-07-01',"length",'validation','validation_2016','test','horizon'])
	#XGB_V10_ensemble_frame = pd.DataFrame(data=all_XGB_V10_result,columns=['metal','2014-07-01',"length",
	#	'2015-01-01',"length",'2015-07-01',"length",'2016-01-01',"length",'2016-07-01',"length",'2017-01-01',"length",'2017-07-01',"length",'2018-01-01',"length",'2018-07-01',"length",'validation','validation_2016','test','horizon'])
	#voting_ensemble_frame = pd.DataFrame(data=all_voting_result,columns=['metal','2014-07-01',"length",
	#	'2015-01-01',"length",'2015-07-01',"length",'2016-01-01',"length",'2016-07-01',"length",'2017-01-01',"length",'2017-07-01',"length",'2018-01-01',"length",'2018-07-01',"length",'validation','validation_2016','test','horizon'])
	#rank_ensemble_frame = pd.DataFrame(data=all_rank_result,columns=['metal','2014-07-01',"length",
	#	'2015-01-01',"length",'2015-07-01',"length",'2016-01-01',"length",'2016-07-01',"length",'2017-01-01',"length",'2017-07-01',"length",'2018-01-01',"length",'2018-07-01',"length",'validation','validation_2016','test','horizon'])
	#rank_weight_ensemble_frame = pd.DataFrame(data=all_rank_weight_result,columns=['metal','2014-07-01',"length",
	#	'2015-01-01',"length",'2015-07-01',"length",'2016-01-01',"length",'2016-07-01',"length",'2017-01-01',"length",'2017-07-01',"length",'2018-01-01',"length",'2018-07-01',"length",'validation','validation_2016','test','horizon'])
	#lr_ensemble_frame.to_csv("lr_renew.csv",index=False)
	#XGB_V5_ensemble_frame.to_csv("XGB_V5_renew.csv",index=False)
	#XGB_V7_ensemble_frame.to_csv("XGB_V7_renew.csv",index=False)
	#XGB_V10_ensemble_frame.to_csv("XGB_V10_renew.csv",index=False)
	#voting_ensemble_frame.to_csv("voting_renew.csv",index=False)
	#rank_ensemble_frame.to_csv("rank_renew.csv",index=False)
	#rank_weight_ensemble_frame.to_csv("rank_weight_renew.csv",index=False)
	voting_frame = voting_frame.sort_values(by=['horizon','window_size'],ascending=(True, True))
	voting_weight_frame = voting_weight_frame.sort_values(by=['horizon','window_size'],ascending=(True, True))
	voting_frame.to_csv("voting.csv",index=False)
	voting_weight_frame.to_csv("voting_weight"+new_path.strip(".txt")[8:]+".csv",index=False)
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