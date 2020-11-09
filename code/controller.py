import os 
import sys
import argparse
import pandas as pd
import datetime
from datetime import date
import multiprocessing as mp
import subprocess
import logging

'''
    This file is the controller to run each individual model (logistic regression, xgboost, alstm, etc)
    It has 9 functions which can be roughly categorized by level

    run:    High level function which chooses the action that is to be taken

    run_tune:       Medium level function which generates the tuning log for a specific model
    run_train:      Medium level function which generates the model instances for a specific model
    run_test:       Medium level function which generates the predictions for a specific model 
    extract:        Medium level function which extracts the predictions of a list of specified dates, metals and horizons
    run_analyze:    Medium level function which breaks down the predictions into metrics of performance (accuracy, f1_score, mae, rmse)

    analyze_zoom:   Low level function which breaks down dates into half years
    run_bash:       Low level function that implements controlled parallel processing for bash scripts
    subprocess_run: Low level function that automates the logging of standard output and error

    There are some universally defined parameters that will be stated here:
    ground_truth_str:   string which specifies the metals. It is comma-separated.
        values:         Al/Cu/Pb/Ni/Xi/Zn
    dates:               string which specifies range of dates to deploy framework on, is of format start_date::end_date, where start_date and end_date
                        are of format YYYY-mm-dd
        values:         2014-07-01::2018-12-31
    horizon_str:        string which specifies the prediction horizon. It is comma-separated.
        values:         1/3/5/10/20/60
    sou:                string which controls the method to read data. Is a reference to the source of the data.
        values:         NExT/4E
    version_str:        string which specifies feature version. It is comma-separated.
        values:         v3/v5/v7/v9/v10.......
    model:              string which specifies the model to be deployed.
        values:         logistic/xgboost/alstm/alstmr/ensemble/pp_sub/pp_filter
                        *pp_sub refers to Substitution, whereas pp_filter refers to Filter
    mc:                 string which identifies whether monte-carlo simulation is triggered.
        values:         True/False

    ground_truth:       string which specify a single metal. Cannot allow for multiple metals
    horizon:            string which specify a single horizon. Cannot allow for multiple horizons
    version:            stirng which specifies a single feature version. Cannot allow for multiple feature versions


'''

def subprocess_run(line):
    '''
        Input
            line    (str) : string which is a line of command
    '''
    
    #get ground truth
    ind = line.index('-gt')
    gt = line[ind+1]
    
    #get horizon
    ind = line.index('-s')
    horizon = line[ind+1]
    
    #get version
    if ('-v') not in line and "code/train_data_ensemble.py" in line:
        version = "ensemble"
    else:
        ind= line.index('-v')
        version = line[ind+1]
    
    #get date
    ind = line.index('-d')
    dates = line[ind+1]
    
    #get method
    method = "train" if "train" in line else "test"

    #get model
    model = line[1].split('_')[-1][:-3]
    
    #generate filepaths for logs for standard output and error
    out = open("log/"+method+"/out/"+'_'.join([model,gt,horizon,version,dates+".txt"]),"w")
    error = open("log/"+method+"/err/"+'_'.join([model,gt,horizon,version,dates+".txt"]),"w")
    
    proc = subprocess.run(line,stdout = out, stderr = error)

    # if there is an error then return the command that has failed
    if proc.returncode != 0:
        return ' '.join(line)
    else:
        return None
    
def run_bash(filename, processes = 12):
    '''
    Input
        filename    (str) : string which is the filepath of the bash script
        processes   (int) : integer which is the number of concurrent processes we will run
    '''
    
    #read bash file
    with open(filename,"r") as f:
        lines = f.readlines()
        
    logger = mp.log_to_stderr(logging.DEBUG)
    #remove lines which include sleep
    lines = list(filter(lambda x: "sleep" not in x, lines))
    
    #remove "> /dev/null 2>&1 &" from line
    lines = [line[:-20].split(' ') for line in lines]
    #generate pool of processes 
    pl = mp.Pool(processes)
    #run commands parallelly
    results = pl.map_async(subprocess_run, lines).get()
    pl.close()
    pl.join()
    logger.info(results)
    if results is None:
        return
    results = [res for res in results if res is not None]

    # if there is an error, log the command that generates the error
    if len(results) > 0:
        with open("log/"+filename.strip('.sh').split('_')[-1]+"error_log_"+lines[0][1].split("_")[-1][:-3]+"_"+ datetime.datetime.strftime(datetime.datetime.now(),"%Y%m%d;%H:%M:%S")+".txt","w") as out:
            for res in results:
                out.write(res+"\n")
            out.close()
                
                
def analyze_zoom(zoom):
    '''
    Input
        zoom    (str)   :   string which consist of start and end date of period in format "yyyy-mm-dd::yyyy-mm-dd", 
                            with first 'yyyy-mm-dd' as start date and the other is the end date
                            *does not necessarily include month and day*
    
    Output
        datelist    (list)  :   list of dates which are split into half years, with the first element and the last element denoting
                                the first date and the last date
                                ie. 2014-02-05::2015-05-07 will output [2014-02-05,2014-06-30,2014-12-31,2015-05-07] 
    '''

    if "::" not in zoom:
        return zoom if len(zoom.split(',')) == 1 else zoom.split(',')
    output = []
    td = date.today()
    yr = str(td.year)
    # get start date and end date from zoom
    start_date = zoom.split('::')[0].split('-')
    end_date = zoom.split('::')[1].split('-') if (zoom.split('::')[1]) != "" else [yr]

    # case of not full date (ie. yyyy-mm::yyyy)
    while len(start_date) < 3:
        start_date.append('01')
    
    while len(end_date) < 3:
        if len(end_date)  == 1:
            end_date.append('12')
        else:
            end_date.append("31")
    
    curr_date = start_date[0]+"-06-30" if int(start_date[1]) < 7 else start_date[0]+"-12-31"

    start_date = '-'.join(start_date)
    end_date = '-'.join(end_date)
    
    #generate list
    output.append(start_date)
    while curr_date < end_date:
        output.append(curr_date)
        curr_date = curr_date[:5]+"12-31" if curr_date[5:7] == "06" else str(int(curr_date[:4])+1)+"-06-30"
    
    output.append(end_date)

    return output

def run_tune(ground_truth, horizon, sou, version, model, mc, date):
    train = "code/train_data_"+model+".py"
    command = ""

    #generate tune command for logistic regression
    if model in ["logistic"]:
        lag = '1,5,10,20'
        for l in lag.split(','):
            command += ' '.join(['python', train, \
                                    '-gt', ground_truth, \
                                    '-s', horizon, \
                                    '-sou', sou, \
                                    '-l', l, \
                                    '-v', version, \
                                    '-d', date, \
                                    '-o', 'tune', \
                                    '>', \
                                    '/dev/null', \
                                    '2>&1', '&',"\n"])
            if int(version[1:]) >= 23:
                break
    
    #generate tune command for xgboost
    elif model in ["xgboost"]:
        lag = "1,5,10" if version not in ["v10","v12"] else "1,5"
        for l in lag.split(','):
            command += ' '.join(['python', train, \
                                    '-gt', ground_truth, \
                                    '-s', horizon, \
                                    '-sou', sou, \
                                    '-l', l, \
                                    '-v', version, \
                                    '-d', date, \
                                    '-o', "tune", \
                                    '>', \
                                    'result/validation/'+model+'/' \
                                    +'_'.join([ground_truth.split("_")[1], 'xgboost', 'l'+lag, 'h'+horizon, version+".txt"]), \
                                    '2>&1', '&',"\n"])
            if int(version[1:]) >= 23:
                break

    #generate tune command for ALSTM(classification)
    elif model in ["alstm"]:
        train = "code/train_data_ALSTM.py"
        command += ' '.join(['python', train, \
                            '-s', horizon, \
                            '-sou', sou, \
                            '-v', version, \
                            '-d', date, \
                            '-o', "tune", \
                            '-log', \
                            './result/validation/'+model+'/' \
                            +'_'.join([version, 'h'+horizon, "tune.log"]), \
                            '>', '/dev/null', \
                            '2>&1', '&',"\n"])

    #generate tune command for alstm regression            
    elif model in ["alstmr"]:
        #generate tune command for alstm regression without monte carlo to determine base hyperparameters
        if mc == "False":
            train = "code/train_data_ALSTMR.py"
            command += ' '.join(['python', train, \
                                '-s', horizon, \
                                '-sou', sou, \
                                '-v', version, \
                                '-d', date, \
                                '-o', "tune", \
                                '-log', \
                                './result/validation/alstm/' \
                                +'_'.join([version, 'h'+horizon, "tune.log"]), \
                                '>', '/dev/null', \
                                '2>&1', '&',"\n"])
            
        #generate tune command for alstm regression with monte carlo
        elif mc == "True":
            #generate tuning results from log
            filepath = ''

            command +=        ' '.join(['python', 'code/train/analyze_alstm.py', \
                                'result/validation/alstm '+ \
                                version+"_h"+horizon+"_tune.log", \
                                '>', 'result/validation/alstm/'+ \
                                version+"_h"+horizon+"_para.txt", "para","\n"]
                            )
            filepath += 'result/validation/alstm/'+version+"_h"+horizon+"_para.txt,"

            #remove last comma character
            filepath = filepath[:-1]

            #extract results and output commands
            command += ' '.join(['python', 'code/utils/analyze_alstm_tune.py', \
                                '-gt', ground_truth, \
                                '-f', filepath, \
                                '-sou', sou, \
                                '-o', 'tune', \
                                '-m', "best_loss", \
                                '-r', '1', \
                                '-mc', "1", \
                                '-d', date,"\n"]
                            )
            command += "bash best_loss_tune.sh"

    #generate tune command for ensemble
    elif model in ["ensemble"]:
        dates = ','.join(date[1:])
        train = "code/train_data_ensemble.py"
        command += ' '.join(['python', train, \
                            '-s', horizon, \
                            '-gt', ground_truth, \
                            '-o', 'tune', \
                            '-c', 'exp/ensemble_tune_all.conf', \
                            '-d', dates+".", \
                            '-t', version+"\n" 
                            ])
    
    #generate tune command for post_process filter
    elif model in ["pp_filter"]:
        train = "code/train_data_pp.py"
        dates = ','.join(date[1:])
        for gt in ground_truth.split(','):
            for s in horizon.split(','):
                command += ' '.join(['python', train, \
                                        '-gt', ground_truth, \
                                        '-s', horizon, \
                                        '-sou', "NExT", \
                                        '-v', version, \
                                        '-d', dates, \
                                        '-o', "tune", \
                                        '-m', "Filter", \
                                        '>', \
                                        '/dev/null', \
                                        '2>&1', '&',"\n"])


    print(command)
    os.system(command)

def run_train(ground_truth_str, horizon_str, sou, version_str, model, mc, dates):

    command = ""
    date = ','.join(dates[1:])
    
    
    #
    #Run all pre-train actions (such as analyzing tuning results)
    #
    # case if model is lr
    if model in ["logistic"]:
        lag = '1,5,10,20'

        train_script = ' '.join(['python', 'code/utils/'+model+"_script.py", \
                            '-gt', ground_truth_str, \
                            '-s', horizon_str, \
                            '-sou', sou, \
                            '-l', lag, \
                            '-v', version_str, \
                            '-p', './result/validation/'+model, \
                            '-out', model+"_train.sh", \
                            '-o', 'train', \
                            '-t', 'f1', \
                            '-d', date,"\n"]
                        )

        # generate shell script for training commands
        command += train_script
    
    if model == "xgboost":
        lag = '1,5,10'
        
        #generate commands to call script
        command += ' '.join(['python', 'code/utils/'+model+"_script.py", \
                            '-gt', ground_truth_str, \
                            '-s', horizon_str, \
                            '-sou', sou, \
                            '-l', lag, \
                            '-v', version_str, \
                            '-p', './result/validation/'+model, \
                            '-o', 'tune', \
                            '>','/dev/null',
                            '2>&1','&\n'    ]
                        )
        command += ' '.join(['python', 'code/utils/'+model+"_script.py", \
                            '-gt', ground_truth_str, \
                            '-s', horizon_str, \
                            '-sou', sou, \
                            '-l', lag, \
                            '-v', version_str, \
                            '-p', './result/validation/'+model, \
                            '-out', model+"_train.sh", \
                            '-o', 'train', \
                            '-d', date,"\n"]
                        )
        
        
    #case if model is alstm
    elif model in ["alstm"]:

        #generate tuning results from log
        filepath = ''
        for version in version_str.split(','):
            v = version.split('_')[0]
            method = '_'.join(version.split('_')[1:])
            for step in horizon_str.split(','):
                command += ' '.join(['python', 'code/train/analyze_alstm.py', \
                                    'result/validation/alstm '+ \
                                    v+"_h"+step+"_tune.log", \
                                    '>', 'result/validation/alstm/'+ \
                                    v+"_h"+step+"_para.txt", "para\n"]
                                )
                filepath += 'result/validation/alstm/'+v+"_h"+step+"_para.txt,"

        #remove last comma character
        filepath = filepath[:-1]

        #extract results and output commands
        command += ' '.join(['python', 'code/utils/analyze_alstm_tune.py', \
                            '-gt', ground_truth_str, \
                            '-f', filepath, \
                            '-sou', sou, \
                            '-o', 'train', \
                            '-m', method, \
                            '-d', date,"\n"]
                        )


    #case if model is alstmr
    elif model in ["alstmr"]:

        #generate tuning results from log
        filepath = ''
        for version in version_str.split(','):
            v = version.split('_')[0]
            method = '_'.join(version.split('_')[1:])
            for step in horizon_str.split(','):
                command += ' '.join(['python', 'code/train/analyze_alstm.py', \
                                    'result/validation/alstm '+ \
                                    v+"_h"+step+"_mc_tune.log", \
                                    '>', 'result/validation/alstm/'+ \
                                    v+"_h"+step+"_mc_para.txt", "para\n"]
                                )
                filepath += 'result/validation/alstm/'+v+"_h"+step+"_mc_para.txt,"

        #remove last comma character
        filepath = filepath[:-1]

        #extract results and output commands
        command += ' '.join(['python', 'code/utils/analyze_alstm_tune.py', \
                            '-gt', ground_truth_str, \
                            '-f', filepath, \
                            '-sou', sou, \
                            '-r', '1', \
                            '-mc','1', \
                            '-o', 'train', \
                            '-m', method, \
                            '-d', date,"\n"]
                        )

    print(command)
    os.system(command)
    
    #
    #Run train commands
    #
    
    if model in ["logistic","xgboost"]:
        run_bash(model+"_train.sh")
    elif model in ["alstm","alstmr"]:
        run_bash(method+"_train.sh",3)
    

def run_test(ground_truth_str, horizon_str, sou, version_str, model, mc, dates):
    command = ""
    date = ','.join(dates[1:])

    #case if model is logistic
    if model in ["logistic"]:
        lag = '1,5,10,20'

        train_script = ' '.join(['python', 'code/utils/'+model+"_script.py", \
                            '-gt', ground_truth_str, \
                            '-s', horizon_str, \
                            '-sou', sou, \
                            '-l', lag, \
                            '-v', version_str, \
                            '-p', './result/validation/'+model, \
                            '-out', model+"_test.sh", \
                            '-o', 'test', \
                            '-t', 'f1', \
                            '-d', date,"\n"]
                        )

        # generate shell script for training commands
        command += train_script
    
    #case if model is xgboost
    if model == "xgboost":
        lag = '1,5,10'
        
        #generate commands to call script
        command += ' '.join(['python', 'code/utils/'+model+"_script.py", \
                            '-gt', ground_truth_str, \
                            '-s', horizon_str, \
                            '-sou', sou, \
                            '-l', lag, \
                            '-v', version_str, \
                            '-p', './result/validation/'+model, \
                            '-o', 'tune', \
                            '>','/dev/null',
                            '2>&1','&\n'    ]
                        )
        command += ' '.join(['python', 'code/utils/'+model+"_script.py", \
                            '-gt', ground_truth_str, \
                            '-s', horizon_str, \
                            '-sou', sou, \
                            '-l', lag, \
                            '-v', version_str, \
                            '-p', './result/validation/'+model, \
                            '-out', model+"_test.sh", \
                            '-o', 'test', \
                            '-d', date,"\n"]
                        )
        
        

    #case if model is alstm
    elif model in ["alstm"]:

        #generate tuning results from log
        filepath = ''
        for version in version_str.split(','):
            v = version.split('_')[0]
            method = '_'.join(version.split('_')[1:])
            for horizon in horizon_str.split(','):
                filepath += 'result/validation/alstm/'+v+"_h"+horizon+"_para.txt,"

        #remove last comma
        filepath = filepath[:-1]

        #extract results and output commands
        command += ' '.join(['python', 'code/utils/analyze_alstm_tune.py', \
                            '-gt', ground_truth_str, \
                            '-f', filepath, \
                            '-sou', sou, \
                            '-o', 'test', \
                            '-m', method, \
                            '-d', date,"\n"]
                        )

    elif model in ["alstmr"]:

        #generate tuning results from log
        filepath = ''
        for version in version_str.split(','):
            v = version.split('_')[0]
            method = '_'.join(version.split('_')[1:])
            for horizon in horizon_str.split(','):
                filepath += 'result/validation/alstm/'+v+"_h"+horizon+"_mc_para.txt,"

        #remove last comma
        filepath = filepath[:-1]

        #extract results and output commands
        command += ' '.join(['python', 'code/utils/analyze_alstm_tune.py', \
                            '-gt', ground_truth_str, \
                            '-f', filepath, \
                            '-sou', sou, \
                            '-r', '1', \
                            '-mc','1', \
                            '-o', 'test', \
                            '-m', method, \
                            '-d', date,"\n"]
                        )

    #case if model is ensemble
    elif model in ["ensemble"]:
        train = "code/train_data_ensemble.py"
        with open("ensemble_test.sh","w") as out:
            for gt in ground_truth_str.split(','):
                for horizon in horizon_str.split(','):
                    out.write(' '.join(['python', train, \
                                        '-s', horizon, \
                                        '-gt', gt, \
                                        '-o', 'test', \
                                        '-c', 'exp/ensemble_tune_all.conf', \
                                        '-d', '.'+date, \
                                        '-t', "window", "> /dev/null 2>&1 &\n"])
                             )
                
    #case if model is post process substitution          
    elif model in ["pp_sub"]:
        with open("pp_sub_test.sh","w") as out:
            for gt in ground_truth_str.split(','):
                for horizon in horizon_str.split(','):
                    out.write(' '.join(['python code/train_data_pp.py',\
                                        '-gt', gt,\
                                        '-s', horizon, \
                                        '-sou', sou, \
                                         '-v',"ensemble,ensemble",\
                                        '-m', "Substitution,analyst", \
                                        '-d', date, "> /dev/null 2>&1 &\n"
                                        ])
                             )
    
    #generate test command for post_process filter
    elif model in ["pp_filter"]:
        reversed_version = ','.join(version_str.split(',')[::-1])
        command += ' '.join(['python code/utils/post_process_script.py', \
                            '-gt', ground_truth_str, \
                            '-s', horizon_str, \
                            '-sou', sou, \
                            '-o', 'simple', \
                            '-m', "Filter", \
                            '-v', reversed_version, \
                             '-d', date, \
                            '-p', 'result/validation/post_process/Filter','\n'
                            ])
        command += ' '.join(['python code/utils/post_process_script.py', \
                            '-gt', ground_truth_str, \
                            '-s', horizon_str, \
                            '-sou', sou, \
                             '-d', date, \
                            '-o', 'commands', \
                            '-m', "Filter", \
                            '-v', version_str, \
                            '-p', 'result/validation/post_process/Filter',\
                            '-out', model+'_test.sh', '\n'
                            ])
    print(command)
    os.system(command)

    if model in ["logistic","xgboost","pp_filter","ensemble","pp_sub"]:
        run_bash(model+"_test.sh")
    elif model in ["alstm","alstmr"]:
        run_bash(method+"_test.sh",6)


def extract(ground_truth, horizon, sou, version, model, mc, dates):
    df = pd.DataFrame()
    for date in dates[1:]:
        if date == dates[-1] and not os.path.exists("result/prediction/"+model+"/"+ '_'.join([ground_truth,date,horizon,version+".csv"])):
            date = '-'.join([date.split('-')[0],"06-30" if date.split('-')[1] <= "06" else "12-31"]) 
        print(date)
        if not os.path.exists("result/prediction/"+model+"/"+ '_'.join([ground_truth,date,horizon,version+".csv"])):
            print("Missing prediction in "+ '_'.join([ground_truth,date,horizon,version+".csv"]))
            continue
            
        df = pd.concat([df,pd.read_csv("result/prediction/"+model+"/"+ '_'.join([ground_truth,date,horizon,version+".csv"]), index_col = 0)],axis = 0)
    return df.loc[dates[0]:dates[-1],:]
    
    

def run_analyze(ground_truth_str, horizon_str, version_str, model, mc, dates):
    
    #remove the first date
    dates = ','.join(dates[1:])

    #if monte_carlo is true then it must be a regression model, thus we can infer that we are interested in both normalized and regular metrics
    if mc == "True":
        #outputs file with extra "p" behind model, denotes regular metrics
        os.system(' '.join(["python code/utils/analyze_predictions.py","-gt",ground_truth_str,"-d", dates, "-m", model, "-s", horizon_str, "-v", version_str,"-r", "price","-mc", "True","-out",model+"p_"+version_str+".csv"]))
        #outputs file with extra "r" behind model, denotes normalized metrics
        os.system(' '.join(["python code/utils/analyze_predictions.py","-gt",ground_truth_str,"-d", dates, "-m", model, "-s", horizon_str, "-v", version_str,"-r", "ret","-mc", "True","-out",model+"r_"+version_str+".csv"]))
    
    #considered a classification problem, and thus will output classification metrics
    else:
        os.system(' '.join(["python code/utils/analyze_predictions.py","-gt", ground_truth_str,"-d", dates, "-m", model, "-s", horizon_str, "-v", version_str, "-out", model+"_"+version_str+".csv"]))
    
    
def run(action, ground_truth_str, horizon_str, sou, version_str, model, mc, dates):
    ground_truth_str = ','.join(["LME_"+gt+"_Spot" for gt in ground_truth_str.split(',')])
    dates = analyze_zoom(dates)
    print(dates)
    if action == "tune":
        run_tune(ground_truth_str, horizon_str, sou, version_str, model, mc, dates)
        return

    elif action == "train":
        run_train(ground_truth_str, horizon_str, sou, version_str, model, mc, dates)
        return

    elif action == "test":
        run_test(ground_truth_str, horizon_str, sou, version_str, model, mc, dates)
        return

    elif action == "analyze":
        run_analyze(ground_truth_str, horizon_str, version_str, model, mc, dates)

    elif action == "predict":
        res = extract(ground_truth_str, horizon_str, sou, version_str, model, mc, dates)
        return res
    
if __name__ ==  '__main__':        
    desc = 'controller for components in machine learning framework'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-gt', '--ground_truth_str',
                        type=str, default="Cu,Al,Ni,Xi,Zn,Pb"
                        )
    parser.add_argument('-s', '--horizon_str',
                        type=str, default="1,3,5,10,20,60"
                        )
    parser.add_argument(
        '-sou','--source', type = str, default = "NExT"
    )
    parser.add_argument(
        '-v','--version_str',help='version_str',type = str
    )
    parser.add_argument('-o','--action', default = 'train', type = str)
    parser.add_argument('-m','--model',type =str, default = 'logistic')
    parser.add_argument('-mc','--mc',type = str, default = "True")
    parser.add_argument('-z','--zoom',type = str, help = "period of dates", default = "2014-07-01::2018-12-31")

    args = parser.parse_args()

    run(args.action,args.ground_truth_str, args.horizon_str,args.source, args.version_str, args.model, args.mc, args.zoom)


        



