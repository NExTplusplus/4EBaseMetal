import os 
import sys
import argparse
import pandas as pd

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
    # get start date and end date from zoom
    start_date = zoom.split('::')[0].split('-')
    end_date = zoom.split('::')[1].split('-')

    # case of not full date (ie. yyyy-mm::yyyy)
    while len(start_date) < 3:
        start_date.append('01')
    
    while len(end_date) < 3:
        end_date.append('01')
    
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
def run_tune(ground_truth, step, sou, version, model, regression, date):
    train = "code/train_data_"+model+".py"
    command = ""

    #generate tune command for logistic regression
    if model == "logistic":
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
    elif model == "xgboost":
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
    elif model == "alstm":
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
                            '2>&1', '&',"\n"])

    
    elif model == "ensemble":
        return
    elif model == "linear":
        return
    elif model == "pp":
        return
    print(command)
    os.system(command)

#extraction of tuning results and generation and processing of training commands
def run_train(ground_truths_str, steps, sou, versions, model, regression, dates):

    date = ','.join(dates[1:])
    # case if model is lr or xgboost
    if model in ["logistic","xgboost"]:
        lag = '1,5,10' if model == "xgboost" else '1,5,10,20'

        #generate commands to call script
        tune_script = ' '.join(['python', 'code/utils/'+model+"_script.py", \
                            '-gt', ground_truths_str, \
                            '-s', steps, \
                            '-sou', sou, \
                            '-l', lag, \
                            '-v', versions, \
                            '-p', './result/validation/'+model, \
                            '-o', 'tune']
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
                            '-d', date]
                        )
        
        print(tune_script)
        print(train_script)
        
        #xgboost requires additional to extract tuning results
        if model  == "xgboost":
            os.system(tune_script)

        # generate shell script for training commands
        os.system(train_script)
    
        os.system('bash '+model+"_train.sh")

    #case if model is alstm
    elif model in ["alstm"]:

        #generate tuning results from log
        filepath = ''
        for version in versions.split(','):
            v = version.split('_')[0]
            method = '_'.join(version.split('_')[1:])
            for step in steps.split(','):
                os.system(
                        ' '.join(['python', 'code/train/analyze_alstm.py', \
                                    'result/validation/alstm '+ \
                                    v+"_h"+step+"_tune.log", \
                                    '>', 'result/validation/alstm/'+ \
                                    v+"_h"+step+"_para.txt", "para"]
                                )
                        )
                filepath += 'result/validation/alstm/'+v+"_h"+step+"_para.txt,"

        #remove last comma character
        filepath = filepath[:-1]

        #extract results and output commands
        train_script = ' '.join(['python', 'code/utils/analyze_alstm_tune.py', \
                            '-gt', ground_truths_str, \
                            '-f', filepath, \
                            '-sou', sou, \
                            '-o', 'train', \
                            '-m', method, \
                            '-d', date]
                        )
        print(train_script)

        os.system(train_script)

        os.system('bash '+method+'_train.sh')


#generation and processing of testing commands
def run_test(ground_truths_str, steps, sou, versions, model, regression, dates):

    date = ','.join(dates[1:])

    # case if model is lr or xgboost
    if model in ["logistic","xgboost"]:

        lag = '1,5,10' if model == "xgboost" else '1,5,10,20'
        test_script = ' '.join(['python', 'code/utils/'+model+"_script.py", \
                            '-gt', ground_truths_str, \
                            '-s', steps, \
                            '-sou', sou, \
                            '-l', lag, \
                            '-v', versions, \
                            '-p', './result/validation/'+model, \
                            '-out', model+"_test.sh", \
                            '-o', 'test', \
                            '-d', date]
                        )

        print(test_script)
        # generate shell script for training commands
        os.system(test_script)
    
        os.system('bash '+model+"_test.sh")

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
        train_script = ' '.join(['python', 'code/utils/analyze_alstm_tune.py', \
                            '-gt', ground_truths_str, \
                            '-f', filepath, \
                            '-sou', sou, \
                            '-o', 'test', \
                            '-m', method, \
                            '-d', date]
                        )

        print(train_script)
        os.system(train_script)

        os.system('bash '+method+'_test.sh')

#run analysis of predictions
def run_analyze(ground_truths_str, steps, versions, model, regression, dates):
    os.system(' '.join(['python', 'code/tuils/analyze_predictions.py', \
                        '-gt', ground_truths_str, \
                        '-s', steps, \
                        '-m', model, \
                        '-d', dates, \
                        '-out', model+".csv"
                        ]))


#extract prediction of related dates
def extract(ground_truth, step, sou, version, model, regression, dates):
    df = pd.DataFrame()
    for date in dates[1:]:
        df = pd.concat([df,pd.read_csv("result/prediction/"+model+"/"+ '_'.join([ground_truth,date,step,version+".csv"]), index_col = 0)],axis = 0)
    return df.loc[dates[0]:,:]
    
    
def run(action, ground_truth, step, sou, version, model, regression, dates):
    if action == "tune":
        run_tune(ground_truth, step, sou, version, model, regression, dates)
        return

    elif action == "train":
        run_train(ground_truth, step, sou, version, model, regression, dates)
        return

    elif action == "test":
        run_test(ground_truth, step, sou, version, model, regression, dates)
        return

    elif action == "analyze":
        run_analyze(ground_truth, step, version, model, regression, dates)

    elif action == "predict":
        run_test(ground_truth, step, sou, version, model, regression, dates)
        res = extract(ground_truth, step, sou, version, model, regression, dates)
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
    parser.add_argument('-r','--regression',type = str, default = "price")
    parser.add_argument('-z','--zoom',type = str, help = "period of dates", default = "2014-07-01::2018-12-31")

    args = parser.parse_args()

    args.ground_truth_list = ','.join(["LME_"+gt+"_Spot" for gt in args.ground_truth_list.split(',')])
    args.zoom = analyze_zoom(args.zoom)
    run(args.action,args.ground_truth_list, args.steps,args.source, args.version, args.model, args.regression, args.zoom)


        



