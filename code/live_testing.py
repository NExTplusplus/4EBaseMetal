import pandas as pd
import os 
import sys
import argparse
from controller import run,analyze_zoom
import time
import datetime

#runs a check for files that should have been generated once the commands specified with model and method are completed
#should a check show that there are files which have not been generated, it will execute a command which can target the generation of that
#specific file
def rerun_for_file(model,method, ground_truth_list = ["Al","Co","Le","Ni","Ti","Zi"], horizon_list = [1,3,5,10,20,60], dates = ""):
    '''
    Input:
        model: string identifier of model (or in the case of alstm the version)
               values: logistic, xgboost,best_loss,best_acc,ave_acc,ave_loss,pp_filter,ensemble
        method:string identifier for whether it is train or test
               values: train, test
        dates: only used for ensemble and is used to identify which dates should be considered in the file check
    '''
    rerun = ""
    folder = "model" if method == "train" else "prediction"
    
    #models which have scripts which hold the commands to run their processes
    if model in ["logistic","xgboost","best_loss","best_acc","ave_acc","ave_loss","pp_filter"]:
        bash_script = model+"_"+method+".sh"
        with open(bash_script,"r") as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                dates_to_rerun = ""
                ls = line.split(' ')
                
                #identify key parameters in each model
                if 'sleep' in line or "/bin/bash" in line:
                    continue
                if model == "xgboost":
                    dates = ls[3]
                    ground_truth = ls[5]
                    lag = ls[7]
                    step = ls[9]
                    version = ls[11]

                elif model == "logistic":
                    version = ls[5]
                    step = ls[7]
                    lag = ls[9]
                    ground_truth = ls[13]
                    dates = ls[17]
                elif model in ["best_loss","best_acc","ave_loss","ave_acc"]:
                    drop = ls[9]
                    embed = ls[11]
                    hidden = ls[15]
                    lag = ls[19]
                    step = ls[-15]
                    version = ls[-13]
                    dates = ls[-7]
                    ground_truth = "LME_Le_Spot"
                    if "--drop_out_mc" in ls:
                        drop_mc = ls[21]
                        repeat_mc = ls[23]
                        step = ls[-17]
                        version = ls[-15]
                elif model == "pp_filter":
                    version = ls[5]
                    ct = ls[11]
                    rt = ls[13]
                    step = ls[15]
                    ground_truth = ls[17]
                    dates = ls[-5]
                               
                #generate file paths that are to be checked based on dates
                for date in dates.split(','):
                    #if method is train then seaerch for model file
                    if method == "train":
                        print("check train")
                        validation_date = date.split('-')[0]+"-01-01" if date.split('-')[1] <= "07" else date.split('-')[0]+"-07-01"
                        if model == "xgboost":
                            filepath = os.path.join("result",folder,"xgboost",\
                                                               '_'.join([validation_date,ground_truth,step,lag,"9",version,"xgb.model"]))
                        elif model == "logistic":
                            filepath = os.path.join("result",folder,"logistic",\
                                                           '_'.join([version,ground_truth,step,lag,"lr",validation_date+".pkl"]))

                        elif model in ["best_loss","best_acc","ave_loss","ave_acc"]:
                            #case if monte carlo is triggered
                            if "--drop_out_mc" in ls:
                                filepath = os.path.join("result",folder,"alstm",version+"_"+model,\
                                                       '_'.join([validation_date,step,drop,hidden,embed,lag,\
                                                                 drop_mc,repeat_mc,version,"alstm.pkl"]))
                            #case if it is regular alstm
                            else:
                                filepath = os.path.join("result",folder,"alstm",version+"_"+model,\
                                                       '_'.join([validation_date,step,drop,hidden,embed,lag,version,"alstm.pkl"]))
                                
                    #if method is test then seaerch for prediction file
                    elif method == "test":
                        print("check test")
                        filepath = os.path.join("result",folder,model,'_'.join([ground_truth,date,step,version+".csv"]))
                        
                        #the below cases have modified filepaths
                        if model == "pp_filter":
                            filepath = os.path.join("result",folder,"post_process","Filter",\
                                                   '_'.join([ground_truth,date,step,"Filter.csv"]))
                            
                        elif model in ["best_loss","best_acc","ave_loss","ave_acc"]:
                            if "--drop_out_mc" in ls:
                                filepath = os.path.join("result",folder,"alstm",version+"_"+model,\
                                                       '_'.join([ground_truth,date,step,version,"True.csv"]))
                            else:
                                filepath = os.path.join("result",folder,"alstm",version+"_"+model,\
                                                       '_'.join([ground_truth,date,step,version+".csv"]))
                                
                    #case if file does not exist print error log
                    if not os.path.exists(filepath):
                        print(filepath + " file is not generated")
                        dates_to_rerun += date+","
                        
                if dates_to_rerun == "":
                    print("all files generated for {},{},{}".format(version,ground_truth,step))
                    continue
                i += 1
                
                #substitute dates 
                if model == "xgboost":
                    ls[2] = dates_to_rerun[:-1]
                elif model == "logistic":
                    ls[16] = dates_to_rerun[:-1]
                elif model in ["best_loss","best_acc","ave_loss","ave_acc"]:
                    ls[-7] = dates_to_rerun[:-1]
                elif model == "pp_filter":
                    ls[-5] = dates_to_rerun[:-1]
                
                rerun += ' '.join(ls)+"\n"

    #models which do not have bash scripts
    elif model in ["ensemble","pp_sub"]:
        for gt in ground_truth_list:
            for step in horizon_list:
                dates_to_rerun = ""
                for date in dates.split(','):
                    #filepath for case if model is ensemble
                    if model == "ensemble":
                        filepath = os.path.join("result",folder,model,\
                                                '_'.join(["LME_"+gt+"_Spot",date,str(step),"ensemble.csv"])) 
                    #filepath for case if model is substitution 
                    else:
                        filepath = os.path.join("result",folder,"post_process","Substitution", \
                                                '_'.join(["LME_"+gt+"_Spot",date,str(step),"Substitution.csv"]))  
                    if not os.path.exists(filepath):
                        print(filepath)
                        dates_to_rerun += date+","
                dates_to_rerun = dates_to_rerun[:-1]
            rerun += "python code/controller.py -s "+str(step)+" -gt LME_"+gt+"_Spot -o test -m ensemble -sou NExT -z "+dates_to_rerun+" > /dev/null 2>&1 &\n"

    print(rerun)             
    os.system(rerun)
            
                    
def live(date, \
         method, \
         ground_truth_str = "Al,Co,Le,Ni,Ti,Zi", \
         horizon_str = "1,3,5,10,20,60" \
        ):
    
    #remove 60 from xgboost
    if "60" in horizon_str:
        temp = horizon_str.split(',')
        temp.remove('60')
        xgboost_horizon_str = ','.join(temp)
    else:
        xgboost_horizon_str = horizon_str
        
    #crawl for new data and generate all available labels
#     command = ""
    command = "python financial_data_crawl.py\n"
    command += "python code/utils/generate_labels.py -sou NExT\n"
    
    #live commands for training models
    if method == "train":
        command += "python code/controller.py -s "+horizon_str+" -v v3,v5,v7 -o train -m logistic -gt "+ground_truth_str+" -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m logistic_train -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -v v10,v12,v24 -o train -m logistic -gt All -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m logistic_train -o check -z "+date+"\n"
        command += "python code/controller.py -s "+xgboost_horizon_str+" -v v3,v5,v7,v23 -o train -m xgboost -gt "+ground_truth_str+" -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m xgboost_train -o check -z "+date+" > /dev/null \n"
        command += "python code/controller.py -s "+xgboost_horizon_str+" -v v10,v24,v28,v30 -o train -m xgboost -gt All -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m xgboost_train -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt All -v v16_best_loss -o train -m alstm -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m best_loss_train -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt All -v v16_best_acc -o train -m alstm -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m best_acc_train -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt All -v v26_best_loss -o train -m alstm -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m best_loss_train -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt All -v v26_best_acc -o train -m alstm -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m best_acc_train -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt All -v v26_ave_loss -o train -m alstm -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m ave_loss_train -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt All -v v26_ave_acc -o train -m alstm -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m ave_acc_train -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt All -v r2_best_loss -o train -m alstmr -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m best_loss_train -o check -z "+date+"\n"

    #live commands for generating predictions
    elif method == "test":
        command += "python code/controller.py -s "+horizon_str+" -v v3,v5,v7,v10,v12,v24 -o test -m logistic -gt "+ground_truth_str+" -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m logistic_test -o check -z "+date+"\n"
        command += "python code/controller.py -s "+xgboost_horizon_str+" -v v3,v5,v7,v23,v10,v24,v28,v30 -o test -m xgboost -gt "+ground_truth_str+" -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m xgboost_test -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt All -v v16_best_loss -o test -m alstm -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m best_loss_test -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt All -v v16_best_acc -o test -m alstm -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m best_acc_test -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt All -v v26_best_loss -o test -m alstm -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m best_loss_test -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt All -v v26_best_acc -o test -m alstm -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m best_acc_test -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt All -v v26_ave_loss -o test -m alstm -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m ave_loss_test -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt All -v v26_ave_acc -o test -m alstm -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m ave_acc_test -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt All -v r2_best_loss -o test -m alstmr -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m best_loss_test -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt "+ground_truth_str+" -o test -m ensemble -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m ensemble -o check -z "+date+"\n"
        command += "python code/controller.py -s "+horizon_str+" -gt "+ground_truth_str+" -v ensemble,ensemble -o test -m pp_sub -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/controller.py -s "+horizon_str+" -gt "+ground_truth_str+" -v r2_best_loss,Substitution -o test -m pp_filter -sou NExT -z "+date+" > /dev/null \n"
        command += "python code/live_testing.py -m pp_filter_test -o check -z "+date+"\n"
    print(command)
    os.system(command)
        
        
def run_live(action, dates, ground_truth, steps, recent, method):
    # get time period for recent days 
    if recent >= 1:
        curr_date = datetime.date.today()
        prev_date = curr_date - datetime.timedelta(days = recent - 1)
        dates = datetime.datetime.strftime(prev_date, '%Y-%m-%d') + "::" + datetime.datetime.strftime(curr_date, '%Y-%m-%d')
        
    #live testing 
    if action == "live":
        live(dates,method,ground_truth,steps)
        
    #live check for generation of file
    elif action == "check":
        model = '_'.join(method.split('_')[:-1])
        method = method.split('_')[-1]
        rerun_for_file(model,method,ground_truth.split(','),[int(i) for i in steps.split(',')],"" if model not in ["ensemble",'pp_sub'] else dates)
    
    #return dataframe
    elif action == "return":
        final_df = pd.DataFrame()
        for gt in ground_truth.split(','):
            for step in steps.split(','):
                df = run("predict",gt,step,"NExT","Filter","post_process/Filter","True",dates)
                print(df)
                df.columns = [gt+step]
                final_df = pd.concat([final_df,df], axis = 1)
        print(final_df)
        return final_df    
    
    

        
if __name__ ==  '__main__':        
    desc = 'controller for financial models'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-m','--method', help='action', type = str, default = "test"
    )
    parser.add_argument('-z','--zoom',type = str, help = "period of dates", default = "")
    parser.add_argument('-gt','--ground_truth',type = str, help = "ground truth string", default = "Al,Co,Le,Ni,Ti,Zi")
    parser.add_argument('-s','--steps',type = str, help = "horizon string", default = "1,3,5,10,20,60")
    parser.add_argument('-r','--recent', type = int, help = "number of recent days", default = 0)
    parser.add_argument('-o','--action',type = str, help = "action", default = "live")

    args = parser.parse_args()
    run_live(args.action, args.zoom, args.ground_truth,args.steps,args.recent, args.method)