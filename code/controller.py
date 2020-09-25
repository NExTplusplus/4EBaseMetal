import os 
import sys
import argparse
import pandas as pd
from datetime import date

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
        return zoom
    output = []
    td = date.today()
    yr = td.year

    # get start date and end date from zoom
    start_date = zoom.split('::')[0].split('-')
    end_date = zoom.split('::')[1].split('-') if len(zoom.split('::')) > 1 else yr

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

#generate and process tuning commands
def run_tune(ground_truth, step, sou, version, model, mc, date):
    train = "code/train_data_"+model+".py"
    command = ""

    #generate tune command for logistic regression
    if model in ["logistic"]:
        lag = '1,5,10,20'
        for l in lag.split(','):
            command += ' '.join(['python', train, \
                                    '-gt', ground_truth, \
                                    '-s', step, \
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
        lag = "1,5,10"
        for l in lag.split(','):
            command += ' '.join(['python', train, \
                                    '-gt', ground_truth, \
                                    '-s', step, \
                                    '-sou', sou, \
                                    '-l', l, \
                                    '-v', version, \
                                    '-d', date, \
                                    '-o', "tune", \
                                    '>', \
                                    'result/validation/'+model+'/' \
                                    +'_'.join([ground_truth.split("_")[1], 'xgboost', 'l'+lag, 'h'+step, version+".txt"]), \
                                    '2>&1', '&',"\n"])
            if int(version[1:]) >= 23:
                break

    #generate tune command for ALSTM(classification)
    elif model in ["alstm"]:
        train = "code/train_data_ALSTM.py"
        command += ' '.join(['python', train, \
                            '-s', step, \
                            '-sou', sou, \
                            '-v', version, \
                            '-d', date, \
                            '-o', "tune", \
                            '-log', \
                            './result/validation/'+model+'/' \
                            +'_'.join([version, 'h'+step, "tune.log"]), \
                            '>', '/dev/null', \
                            '2>&1', '&',"\n"])

    #generate tune command for alstm regression            
    elif model in ["alstmr"]:
        #generate tune command for alstm regression without monte carlo to determine base hyperparameters
        if mc == "False":
            train = "code/train_data_ALSTMR.py"
            command += ' '.join(['python', train, \
                                '-s', step, \
                                '-sou', sou, \
                                '-v', version, \
                                '-d', date, \
                                '-o', "tune", \
                                '-log', \
                                './result/validation/alstm/' \
                                +'_'.join([version, 'h'+step, "tune.log"]), \
                                '>', '/dev/null', \
                                '2>&1', '&',"\n"])
            
        #generate tune command for alstm regression with monte carlo
        elif mc == "True":
            #generate tuning results from log
            filepath = ''

            command +=        ' '.join(['python', 'code/train/analyze_alstm.py', \
                                'result/validation/alstm '+ \
                                version+"_h"+step+"_tune.log", \
                                '>', 'result/validation/alstm/'+ \
                                version+"_h"+step+"_para.txt", "para","\n"]
                            )
            filepath += 'result/validation/alstm/'+version+"_h"+step+"_para.txt,"

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
                            '-s', step, \
                            '-gt', ground_truth, \
                            '-o', 'tune', \
                            '-c', 'exp/ensemble_tune_all.conf', \
                            '-d', dates+".", \
                            '-t', "window\n" 
                            ])
        command += ' '.join(['python', train, \
                            '-s', step, \
                            '-gt', ground_truth, \
                            '-o', 'tune', \
                            '-c', 'exp/ensemble_tune_all.conf', \
                            '-d', dates+".", \
                            '-t', "fv\n" 
                            ])
    
    #generate tune command for post_process filter
    elif model in ["pp_filter"]:
        train = "code/train_data_pp.py"
        dates = ','.join(date[1:])
        command += ' '.join(['python', train, \
                                '-gt', ground_truth, \
                                '-s', step, \
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

#extraction of tuning results and generation and processing of training commands
def run_train(ground_truths_str, steps, sou, versions, model, mc, dates):

    command = ""
    date = ','.join(dates[1:])
    # case if model is lr or xgboost
    if model in ["logistic","xgboost"]:
        lag = '1,5,10' if model in ["xgboost"] else '1,5,10,20'

        #generate commands to call script
        tune_script = ' '.join(['python', 'code/utils/'+model+"_script.py", \
                            '-gt', ground_truths_str, \
                            '-s', steps, \
                            '-sou', sou, \
                            '-l', lag, \
                            '-v', versions, \
                            '-p', './result/validation/'+model, \
                            '-o', 'tune', \
                            '>','/dev/null',
                            '2>&1','&\n'    ]
                        )
        train_script = ' '.join(['python', 'code/utils/'+model+"_script.py", \
                            '-gt', ground_truths_str, \
                            '-s', steps, \
                            '-sou', sou, \
                            '-l', lag, \
                            '-v', versions, \
                            '-p', './result/validation/'+model, \
                            '-out', model+"_train.sh", \
                            '-o', 'train', \
                            '-d', date,"\n"]
                        )
        
        print(tune_script)
        print(train_script)
        
        #xgboost requires additional to extract tuning results
        if model  == "xgboost":
            command += tune_script

        # generate shell script for training commands
        command += train_script
    
        command += 'bash '+model+"_train.sh\n"

    #case if model is alstm
    elif model in ["alstm"]:

        #generate tuning results from log
        filepath = ''
        for version in versions.split(','):
            v = version.split('_')[0]
            method = '_'.join(version.split('_')[1:])
            for step in steps.split(','):
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
                            '-gt', ground_truths_str, \
                            '-f', filepath, \
                            '-sou', sou, \
                            '-o', 'train', \
                            '-m', method, \
                            '-d', date,"\n"]
                        )

        command += 'bash '+method+'_train.sh\n'

    #case if model is alstmr
    elif model in ["alstmr"]:

        #generate tuning results from log
        filepath = ''
        for version in versions.split(','):
            v = version.split('_')[0]
            method = '_'.join(version.split('_')[1:])
            for step in steps.split(','):
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
                            '-gt', ground_truths_str, \
                            '-f', filepath, \
                            '-sou', sou, \
                            '-r', '1', \
                            '-mc','1', \
                            '-o', 'train', \
                            '-m', method, \
                            '-d', date,"\n"]
                        )

        command += 'bash '+method+'_train.sh\n'
    print(command)
    os.system(command)



#generation and processing of testing commands
def run_test(ground_truths_str, steps, sou, versions, model, mc, dates):
    command = ""
    date = ','.join(dates[1:])

    # case if model is lr or xgboost
    if model in ["logistic","xgboost"]:

        lag = '1,5,10' if model in ["xgboost"] else '1,5,10,20'
        command += ' '.join(['python', 'code/utils/'+model+"_script.py", \
                            '-gt', ground_truths_str, \
                            '-s', steps, \
                            '-sou', sou, \
                            '-l', lag, \
                            '-v', versions, \
                            '-p', './result/validation/'+model, \
                            '-out', model+"_test.sh", \
                            '-o', 'test', \
                            '-d', date,'\n']
                        )

        # generate shell script for training commands
    
        command += 'bash '+model+"_test.sh\n"

    #case if model is alstm
    elif model in ["alstm"]:

        #generate tuning results from log
        filepath = ''
        for version in versions.split(','):
            v = version.split('_')[0]
            method = '_'.join(version.split('_')[1:])
            for step in steps.split(','):
                filepath += 'result/validation/alstm/'+v+"_h"+step+"_para.txt,"

        #remove last comma
        filepath = filepath[:-1]

        #extract results and output commands
        command += ' '.join(['python', 'code/utils/analyze_alstm_tune.py', \
                            '-gt', ground_truths_str, \
                            '-f', filepath, \
                            '-sou', sou, \
                            '-o', 'test', \
                            '-m', method, \
                            '-d', date,"\n"]
                        )

        command += 'bash '+method+'_test.sh\n'

    elif model in ["alstmr"]:

        #generate tuning results from log
        filepath = ''
        for version in versions.split(','):
            v = version.split('_')[0]
            method = '_'.join(version.split('_')[1:])
            for step in steps.split(','):
                filepath += 'result/validation/alstm/'+v+"_h"+step+"_mc_para.txt,"

        #remove last comma
        filepath = filepath[:-1]

        #extract results and output commands
        command += ' '.join(['python', 'code/utils/analyze_alstm_tune.py', \
                            '-gt', ground_truths_str, \
                            '-f', filepath, \
                            '-sou', sou, \
                            '-r', '1', \
                            '-mc','1', \
                            '-o', 'test', \
                            '-m', method, \
                            '-d', date,"\n"]
                        )

        command += 'bash '+method+'_test.sh\n'

    elif model in ["ensemble"]:
        train = "code/train_data_ensemble.py"
        for gt in ground_truths_str.split(','):
            for step in steps.split(','):
                command += ' '.join(['python', train, \
                                    '-s', step, \
                                    '-gt', gt, \
                                    '-o', 'test', \
                                    '-c', 'exp/ensemble_tune.conf', \
                                    '-d', '.'+date, \
                                    '-t', "window", \
                                    '>', \
                                    '/dev/null', \
                                    '2>&1', '&',"\n"])
    
    #generate test command for post_process filter
    elif model in ["pp_filter"]:
        reversed_version = ','.join([version.split(',')[::-1]])
        for gt in ground_truths_str.split(','):
            for step in steps.split(','):
                command += ' '.join(['python code/utils/post_process_script.py', \
                                    '-gt', gt, \
                                    '-s', step, \
                                    '-o', 'simple', \
                                    '-m', "Filter", \
                                    '-v', reversed_version, \
                                    '-p', './result/validation/post_process/Filter','\n'
                                    ])
        command += ' '.join(['python code/utils/post_process_script.py', \
                            '-gt', ground_truths_str, \
                            '-s', steps, \
                            '-o', 'complex', \
                            '-m', "Filter", \
                            '-v', reversed_version, \
                            '-p', './result/validation/post_process/Filter','\n'
                            ])
        command += ' '.join(['python code/utils/post_process_script.py', \
                            '-gt', ground_truths_str, \
                            '-s', steps, \
                            '-o', 'commands', \
                            '-m', "Filter", \
                            '-v', reversed_version, \
                            '-p', './result/validation/post_process/Filter',\
                            '-out', model+'_test.sh', '\n'
                            ])
        command+= "bash "+model+"_test.sh" 
    print(command)
    os.system(command)

#run analysis of predictions
def run_analyze(ground_truths_str, steps, versions, model, mc, dates):

    date = ','.join(dates[1:])
    if mc == "False":
        os.system(' '.join(['python', 'code/utils/analyze_predictions.py', \
                            '-gt', ground_truths_str, \
                            '-v', versions, \
                            '-s', steps, \
                            '-m', model, \
                            '-d', date, \
                            '-out', model+".csv"
                            ]))
        if model in ["alstmr"]:
            os.system(' '.join(['python', 'code/utils/analyze_predictions.py', \
                                '-gt', ground_truths_str, \
                                '-v', versions, \
                                '-s', steps, \
                                '-m', "alstm", \
                                '-d', date, \
                                '-r', 'ret', \
                                '-mc', '0', \
                                '-out', "alstmr_ret_T.csv"
                                ]))
            os.system(' '.join(['python', 'code/utils/analyze_predictions.py', \
                                '-gt', ground_truths_str, \
                                '-v', versions, \
                                '-s', steps, \
                                '-m', "alstm", \
                                '-d', date, \
                                '-r', 'price', \
                                '-mc', '0', \
                                '-out', "alstmr_ret_P.csv"
                                ]))
    elif mc == "True":
        os.system(' '.join(['python', 'code/utils/analyze_predictions.py', \
                            '-gt', ground_truths_str, \
                            '-v', versions, \
                            '-s', steps, \
                            '-m', "alstm", \
                            '-d', date, \
                            '-r', 'ret', \
                            '-mc', '1', \
                            '-out', model+"_ret_T.csv"
                            ]))
        os.system(' '.join(['python', 'code/utils/analyze_predictions.py', \
                            '-gt', ground_truths_str, \
                            '-v', versions, \
                            '-s', steps, \
                            '-m', "alstm", \
                            '-d', date, \
                            '-r', 'price', \
                            '-mc', '1', \
                            '-out', model+"_ret_P.csv"
                            ]))

#extract prediction of related dates
def extract(ground_truth, step, sou, version, model, mc, dates):
    df = pd.DataFrame()
    for date in dates[1:]:
        df = pd.concat([df,pd.read_csv("result/prediction/"+model+"/"+ '_'.join([ground_truth,date,step,version+".csv"]), index_col = 0)],axis = 0)
    return df.loc[dates[0]:,:]
    
    
def run(action, ground_truth, step, sou, version, model, mc, dates):
    if action == "tune":
        run_tune(ground_truth, step, sou, version, model, mc, dates)
        return

    elif action == "train":
        run_train(ground_truth, step, sou, version, model, mc, dates)
        return

    elif action == "test":
        run_test(ground_truth, step, sou, version, model, mc, dates)
        return

    elif action == "analyze":
        run_analyze(ground_truth, step, version, model, mc, dates)

    elif action == "predict":
        run_test(ground_truth, step, sou, version, model, mc, dates)
        res = extract(ground_truth, step, sou, version, model, mc, dates)
        print(res)
        return res
    
if __name__ ==  '__main__':        
    desc = 'controller for financial models'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-gt', '--ground_truth_list', help='list of ground truths, separated by ","',
                        type=str, default="Co,Al,Ni,Ti,Zi,Le"
                        )
    parser.add_argument('-s', '--steps', help='list of horizons, separated by ","',
                        type=str, default="1,3,5,10,20,60"
                        )
    parser.add_argument(
        '-sou','--source', help='source of data to be inserted into commands', type = str, default = "4E"
    )
    parser.add_argument(
        '-v','--version',help='version',type = str
    )
    parser.add_argument('-o','--action', default = 'train', type = str)
    parser.add_argument('-m','--model',type =str, default = 'lr')
    parser.add_argument('-mc','--mc',type = str, default = "True")
    parser.add_argument('-z','--zoom',type = str, help = "period of dates", default = "2014-07-01::2018-12-31")

    args = parser.parse_args()

    args.ground_truth_list = ','.join(["LME_"+gt+"_Spot" for gt in args.ground_truth_list.split(',')])
    args.zoom = analyze_zoom(args.zoom)
    run(args.action,args.ground_truth_list, args.steps,args.source, args.version, args.model, args.mc, args.zoom)


        



