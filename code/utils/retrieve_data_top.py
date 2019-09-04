import pandas as pd
path_list = ['LMZSDY_v5_l10_h1.txt','LMZSDY_v5_l10_h3.txt','LMZSDY_v5_l10_h5.txt','LMZSDY_v5_l5_h1.txt','LMZSDY_v5_l5_h3.txt','LMZSDY_v5_l5_h5.txt','LMZSDY_v5_l20_h1.txt'
,'LMZSDY_v5_l20_h3.txt','LMZSDY_v5_l20_h5.txt','LMZSDY_v5_l30_h1.txt','LMZSDY_v5_l30_h3.txt','LMZSDY_v5_l30_h5.txt','LMSNDY_v5_l5_h1.txt','LMSNDY_v5_l5_h3.txt','LMSNDY_v5_l5_h5.txt'
,'LMSNDY_v5_l10_h1.txt','LMSNDY_v5_l10_h3.txt','LMSNDY_v5_l10_h5.txt','LMSNDY_v5_l20_h1.txt','LMSNDY_v5_l20_h3.txt','LMSNDY_v5_l20_h5.txt'
,'LMSNDY_v5_l30_h1.txt','LMSNDY_v5_l30_h3.txt','LMSNDY_v5_l30_h5.txt','LMPBDY_v5_l5_h1.txt','LMPBDY_v5_l10_h1.txt','LMPBDY_v5_l20_h1.txt'
,'LMPBDY_v5_l30_h1.txt','LMPBDY_v5_l5_h3.txt','LMPBDY_v5_l10_h3.txt','LMPBDY_v5_l20_h3.txt','LMPBDY_v5_l30_h3.txt','LMPBDY_v5_l5_h5.txt','LMPBDY_v5_l10_h5.txt'
,'LMPBDY_v5_l20_h5.txt','LMPBDY_v5_l30_h5.txt','LMNIDY_v5_l5_h1.txt','LMNIDY_v5_l5_h3.txt','LMNIDY_v5_l5_h5.txt','LMNIDY_v5_l10_h1.txt'
,'LMNIDY_v5_l10_h3.txt','LMNIDY_v5_l10_h5.txt','LMNIDY_v5_l20_h1.txt','LMNIDY_v5_l20_h3.txt','LMNIDY_v5_l20_h5.txt','LMNIDY_v5_l30_h1.txt','LMNIDY_v5_l30_h3.txt'
,'LMNIDY_v5_l30_h5.txt','LMCADY_v5_l5_h1.txt','LMCADY_v5_l5_h3.txt','LMCADY_v5_l5_h5.txt','LMCADY_v5_l10_h1.txt','LMCADY_v5_l10_h3.txt','LMCADY_v5_l10_h5.txt'
,'LMCADY_v5_l20_h1.txt','LMCADY_v5_l20_h3.txt','LMCADY_v5_l20_h5.txt','LMCADY_v5_l30_h1.txt','LMCADY_v5_l30_h3.txt','LMCADY_v5_l30_h5.txt','LMAHDY_v5_l5_h1.txt'
,'LMAHDY_v5_l5_h3.txt','LMAHDY_v5_l5_h5.txt','LMAHDY_v5_l10_h1.txt','LMAHDY_v5_l10_h3.txt','LMAHDY_v5_l10_h5.txt','LMAHDY_v5_l20_h1.txt','LMAHDY_v5_l20_h3.txt','LMAHDY_v5_l20_h5.txt'
,'LMAHDY_v5_l30_h1.txt','LMAHDY_v5_l30_h3.txt','LMAHDY_v5_l30_h5.txt']
for new_path in path_list:
    path = new_path
    all_file = []
    sub_file = []
    all_voting_Str = 'the all folder voting precision is'
    lag_Str = 'the lag is'
    with open(path,"r") as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            if all_voting_Str.lower() in line.lower():
                file = []
                file.append(float(line.strip("\n").split(" ")[-1]))
                #print("the line is {} and the result is {}".format(line,line.strip("\n").split(" ")[-1]))
                for new_line in lines[i+1:i+10]:
                    file.append(float(new_line.strip("\n").split(" ")[-1]))
                    #print("the line is {} and the result is {}".format(line,line.strip("\n").split(" ")[-1]))
                sub_file.append(file)
                if lag_Str.lower() in lines[i+10].lower():
                    for result in sub_file:
                        result.append(lines[i+10].strip("\n").split(" ")[-1])
                        result.append(lines[i+11].strip("\n").split(" ")[-1])
                        result.append(lines[i+12].strip("\n").split(" ")[-1])
                    all_file+=sub_file
                    sub_file = []
    #print(all_file)
    file_dataframe = pd.DataFrame(all_file,columns=['all_voting','near_voting','far_voting','same_voting','reverse_voting','max_depth','learning_rate','gamma','min_child_weight','subsample','lag','train_date','test_date'])
    lag_list = list(file_dataframe['lag'].unique())
    print(lag_list)
    max_depth_list = list(file_dataframe['max_depth'].unique())
    learning_rate_list = list(file_dataframe['learning_rate'].unique())
    gamma_list = list(file_dataframe['gamma'].unique())
    min_child_weight_list = list(file_dataframe['min_child_weight'].unique())
    subsample_list = list(file_dataframe['subsample'].unique())
    #file_dataframe.to_csv("LMAHDY_h5_l30_xgbv6.csv",index=False)
    all_mean=[]
    for lag in lag_list:
        for max_depth in max_depth_list:
            for learning_rate in learning_rate_list:
                for gamma in gamma_list:
                    for min_child_weight in min_child_weight_list:
                        for subsample in subsample_list:
                            mean_list = []
                            mean_list.append(lag)
                            mean_list.append(max_depth)
                            mean_list.append(learning_rate)
                            mean_list.append(gamma)
                            mean_list.append(min_child_weight)
                            mean_list.append(subsample)
                            all_mean_result = file_dataframe[(file_dataframe['max_depth']==max_depth)&(file_dataframe['learning_rate']==learning_rate)
                                        &(file_dataframe['gamma']==gamma)&(file_dataframe['min_child_weight']==min_child_weight)&(file_dataframe['subsample']==subsample)&(file_dataframe['lag']==lag)]['all_voting'].mean()
                            near_mean_result = file_dataframe[(file_dataframe['max_depth']==max_depth)&(file_dataframe['learning_rate']==learning_rate)
                                        &(file_dataframe['gamma']==gamma)&(file_dataframe['min_child_weight']==min_child_weight)&(file_dataframe['subsample']==subsample)&(file_dataframe['lag']==lag)]['near_voting'].mean()
                            far_mean_result = file_dataframe[(file_dataframe['max_depth']==max_depth)&(file_dataframe['learning_rate']==learning_rate)
                                        &(file_dataframe['gamma']==gamma)&(file_dataframe['min_child_weight']==min_child_weight)&(file_dataframe['subsample']==subsample)&(file_dataframe['lag']==lag)]['far_voting'].mean()
                            reverse_mean_result = file_dataframe[(file_dataframe['max_depth']==max_depth)&(file_dataframe['learning_rate']==learning_rate)
                                        &(file_dataframe['gamma']==gamma)&(file_dataframe['min_child_weight']==min_child_weight)&(file_dataframe['subsample']==subsample)&(file_dataframe['lag']==lag)]['reverse_voting'].mean()
                            same_mean_result = file_dataframe[(file_dataframe['max_depth']==max_depth)&(file_dataframe['learning_rate']==learning_rate)
                                        &(file_dataframe['gamma']==gamma)&(file_dataframe['min_child_weight']==min_child_weight)&(file_dataframe['subsample']==subsample)&(file_dataframe['lag']==lag)]['same_voting'].mean()
                            mean_list.append(all_mean_result)
                            mean_list.append(near_mean_result)
                            mean_list.append(far_mean_result)
                            mean_list.append(reverse_mean_result)
                            mean_list.append(same_mean_result)
                            all_mean.append(mean_list)
        new_frame = pd.DataFrame(all_mean,columns = ['lag','max_depth','learning_rate','gamma','min_child_weigh','subsample','all_mean_result','near_mean_result','far_mean_result','reverse_mean_result','same_mean_result'])
        all_frame = new_frame.sort_values(by='all_mean_result',ascending=False)[:5].rename(columns = {'all_mean_result':'result'}).loc[:,['lag','max_depth','learning_rate','gamma','min_child_weigh','subsample','result']].sort_values(by="result",ascending = False)
        near_frame = new_frame.sort_values(by='near_mean_result',ascending=False)[:5].rename(columns = {'near_mean_result':'result'}).loc[:,['lag','max_depth','learning_rate','gamma','min_child_weigh','subsample','result']].sort_values(by="result",ascending = False)
        far_frame = new_frame.sort_values(by='far_mean_result',ascending=False)[:5].rename(columns = {'far_mean_result':'result'}).loc[:,['lag','max_depth','learning_rate','gamma','min_child_weigh','subsample','result']].sort_values(by="result",ascending = False)
        reverse_frame = new_frame.sort_values(by='reverse_mean_result',ascending=False)[:5].rename(columns = {'reverse_mean_result':'result'}).loc[:,['lag','max_depth','learning_rate','gamma','min_child_weigh','subsample','result']].sort_values(by="result",ascending = False)
        same_frame = new_frame.sort_values(by='same_mean_result',ascending=False)[:5].rename(columns = {'same_mean_result':'result'}).loc[:,['lag','max_depth','learning_rate','gamma','min_child_weigh','subsample','result']].sort_values(by="result",ascending = False)

        frame = pd.concat([all_frame,near_frame,far_frame,reverse_frame,same_frame], axis = 0)
        frame.to_csv(path.split("_")[0]+"_"+"v5"+"_"+str(lag)+"_"+path.split("_")[3].split(".")[0]+".csv",index = False)
    #names = ['all','near','far','reverse','same','']
    #with open((path[:-4]+"_mean.csv"),"w") as out:
    #    with open((path[:-4]+"_mean_temp.csv")) as fl:
    #        lines = fl.readlines()
    #        for i in range(len(lines)):
    #            out.write(lines[i])
    #            if i%5 == 0:
    #                out.write(names[int(i/5)]+"\n")
    #                if i != 0 and i != 25:
    #                   out.write("max_depth,learning_rate,gamma,min_child_weigh,subsample,result\n")