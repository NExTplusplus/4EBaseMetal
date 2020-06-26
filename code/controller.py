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

def get_config(version):
    '''
        get configuration file
        Input
            version (str)   :   string which hold the version
        Output
            config  (str)   :   configuration filepath
    '''
    if version in ["v5","v7"]:
        #requires global data
        return "exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf"
    elif version in ["v3","v23","v24","v28","v30","v37","v39","v41","v43"]:
        return "exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf"
    elif version in ["v9","v10","v12","v16","v26"]:
        return "exp/online_v10.conf"
    elif version in ["v31","v32"]:
        return "exp/supply and demand.conf"
    elif version in ["v33","v35"]:
        return "exp/TP_v1.conf"
    elif version in ['r2']:
        return 'exp/regression_r2.conf'
    else:
        print("Version out of bounds!")

def run_tune(ground_truth, step, sou, version, model, regression, date):
    train = "code/train_data_"+model+".py"
    config = get_config(version)
    if model == "lr":
        for l in [1,5,10,20]:
            lag = str(l)
            os.system(' '.join(['python', train, \
                                '-gt', ground_truth, \
                                '-s', step, \
                                '-c', config, \
                                '-sou', sou, \
                                '-l', lag, \
                                '-v', version, \
                                '-d', date, \
                                '-o', 'tune', \
                                '>', \
                                '1', \
                                '2>&1', '&']))

    elif model == "xgboost":
        for l in [1,5,10]:
            lag = str(l)
            os.system(' '.join(['python', train, \
                                '-gt', ground_truth, \
                                '-s', step, \
                                '-c', config, \
                                '-sou', sou, \
                                '-l', lag, \
                                '-v', version, \
                                '-d', date, \
                                '-o', "tune", \
                                '>', \
                                'result/validation/'+model+'/' \
                                +'_'.join([ground_truth, 'xgboost', 'l1', 'h'+step, version+".txt"]), \
                                '2>&1', '&']))
    
    elif model == "alstm":
        os.system(' '.join(['python', train, \
                            '-s', step, \
                            '-c', config, \
                            '-sou', sou, \
                            '-v', version, \
                            '-d', date, \
                            '-o', "tune", \
                            '-log', \
                            '.result/validation/'+model+'/' \
                            +'_'.join([version, 'h'+step, "tune.log"]), \
                            '2>&1', '&']))

    
    elif model == "ensemble":
        return
    elif model == "linear":
        return
    elif model == "pp":
        return

def run_train(ground_truths_str, steps, sou, versions, model, regression, dates):

    date = ','.join(dates[1:])
    # case if model is lr or xgboost
    if model in ["lr","xgboost"]:
        model_str = model if model != "lr" else "log_reg"

        lag = '1,5,10' if model == "xgboost" else '1,5,10,20'
        #generate commands to call script
        tune_script = ' '.join(['python', 'code/utils/'+model_str+"_script.py", \
                            '-gt', ground_truths_str, \
                            '-s', steps, \
                            '-sou', sou, \
                            '-l', lag, \
                            '-v', versions, \
                            '-p', './result/validation/'+model, \
                            '-o', '"tuning"']
                        )
        train_script = ' '.join(['python', 'code/utils/'+model_str+"_script.py", \
                            '-gt', ground_truths_str, \
                            '-s', steps, \
                            '-sou', sou, \
                            '-l', lag, \
                            '-v', versions, \
                            '-p', './result/validation/'+model, \
                            '-out', model+"_train.sh", \
                            '-o', '"train commands"', \
                            '-d', date]
                        )
        
        #xgboost requires additional to aextract tuning results
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
            for step in steps.split(','):
                os.system(
                        ' '.join(['python', 'code/train/analyze_alstm_tune.py', \
                                    'result/validation/alstm/'+ \
                                    version+"_h"+step+"_tune.log", \
                                    '>', 'result/validation/alstm/'+ \
                                    version+"_h"+step+"_para.txt", "para"]
                                )
                        )
                filepath += 'result/validation/alstm/'+version+"_h"+step+"_para.txt,"

        #remove last comma
        filepath = filepath[:-1]

        #extract results and output commands
        train_script = ' '.join(['python', 'code/utils/analze_alstm_tune.py', \
                            '-gt', ground_truths_str, \
                            '-f', filepath, \
                            '-sou', sou, \
                            '-p', './result/validation/'+model, \
                            '-o', 'train', \
                            '-d', date]
                        )

        os.system(train_script)

        os.system('bash best_acc_train.sh')
        os.system('bash best_loss_train.sh')
        os.system('bash ave_acc_train.sh')
        os.system('bash ave_loss_train.sh')

def run_test(ground_truths_str, steps, sou, versions, model, regression, dates):

    date = ','.join(dates[1:])
    # case if model is lr or xgboost
    if model in ["lr","xgboost"]:
        model_str = model if model != "lr" else "log_reg"

        lag = '1,5,10' if model == "xgboost" else '1,5,10,20'
        train_script = ' '.join(['python', 'code/utils/'+model_str+"_script.py", \
                            '-gt', ground_truths_str, \
                            '-s', steps, \
                            '-sou', sou, \
                            '-l', lag, \
                            '-v', versions, \
                            '-p', './result/validation/'+model, \
                            '-out', model+"_test.sh", \
                            '-o', '"test commands"', \
                            '-d', date]
                        )

        # generate shell script for training commands
        os.system(train_script)
    
        os.system('bash '+model+"_test.sh")

    #case if model is alstm
    elif model in ["alstm"]:

        #generate tuning results from log
        filepath = ''
        for version in versions.split(','):
            for step in steps.split(','):
                filepath += 'result/validation/alstm/'+version+"_h"+step+"_para.txt,"

        #remove last comma
        filepath = filepath[:-1]

        #extract results and output commands
        train_script = ' '.join(['python', 'code/utils/analze_alstm_tune.py', \
                            '-gt', ground_truths_str, \
                            '-f', filepath, \
                            '-sou', sou, \
                            '-p', './result/validation/'+model, \
                            '-o', 'test', \
                            '-d', date]
                        )

        os.system(train_script)

        os.system('bash best_acc_test.sh')
        os.system('bash best_loss_test.sh')
        os.system('bash ave_acc_test.sh')
        os.system('bash ave_loss_test.sh')

def extract(ground_truth, step, sou, version, model, regression, dates):
    df = pd.DataFrame()
    for date in dates[1:]:
        df = pd.concat([df,pd.read_csv("result/prediction/"+model+"/"+ '_'.join([ground_truth,date,step,version+".csv"]), index_col = 0)],axis = 0)
    return df.loc[dates[0]:,:]
    
    
def run(action, ground_truth, step, sou, version, model, regression, dates):
    if action == "tune":
        run_tune(ground_truth, step, sou, version, model, regression, dates)
    elif action == "train":
        run_train(ground_truth, step, sou, version, model, regression, dates)
    elif action == "test":
        run_test(ground_truth, step, sou, version, model, regression, dates)
    elif action == "predict":
        run_test(ground_truth, step, sou, version, model, regression, dates)
        print(extract(ground_truth, step, sou, version, model, regression, dates))

    
# def total_main_controller(data_preprocess, run_prediction, predict_period, predict_recent_days, predict_metal, run_mode, whether_return_df):
#     '''
#     :param data_preprocess: bool, whether to do the step 1 to step 3
#     :param run_prediction: bool, whether to do the step 4
#     :param predict_period: str, the date we need to predict
#     :param predict_recent_days:, str, how many recent days we need to predict
#     :param predict_metal: str, the certain metal we need to predict, when it is 'all', predict all the metal
#     :param run_mode: str, whether to compare the results with the labels
#     :param whether_return_df: bool, whether to give the output results
#     '''

#     if eval(data_preprocess) == True:
        
#         print('--------------------------data preprocessing-------------------')
#         step1_path = '/mnt/gluster/Alphien/paperTrading/Team/NEXT/4EBaseMetal/code/new_Analyst_Report_Chinese/step1_crawler/'
#         step1_code = 'python crawler.py run'
#         print('------------------running step 1-------------------------------')
#         os.system('cd {};{}'.format(step1_path, step1_code))
        
#         step2_path = '/mnt/gluster/Alphien/paperTrading/Team/NEXT/4EBaseMetal/code/new_Analyst_Report_Chinese/step2_extract_html/'
#         step2_code = 'python step2_main_contraoller.py'
#         print('------------------running step 2-------------------------------')
#         os.system('cd {};{}'.format(step2_path, step2_code))
        
#         step3_path = '/mnt/gluster/Alphien/paperTrading/Team/NEXT/4EBaseMetal/code/new_Analyst_Report_Chinese/step3_extract_recommendation/'
#         step3_code = 'python call_aip.py run'
#         print('------------------running step 3-------------------------------')
#         os.system('cd {};{}'.format(step3_path, step3_code))

#     if eval(run_prediction) == True:
        
#         if predict_period == 'None':
#             if int(predict_recent_days) >= 1:
#                 end_date = datetime.datetime.now()
#                 start_date = end_date- datetime.timedelta(days=int(predict_recent_days)-1)
#                 use_prediction_period = [[datetime.datetime.strftime(start_date, '%Y-%m-%d'), datetime.datetime.strftime(end_date, '%Y-%m-%d')]]
#             else:
#                 return 'wrong input of the parameter recent days'
#         else:
#             use_prediction_period = get_half_date(predict_period)
            
#         step4_path = '/mnt/gluster/Alphien/paperTrading/Team/NEXT/4EBaseMetal/code/new_Analyst_Report_Chinese/step4_sentiment_analysis/'
        
#         for per in use_prediction_period:
            
#             os.system('cd {};python main_function.py {} {} {} {}'.format(step4_path, per[0], per[1], predict_metal, run_mode))
    
#     if eval(whether_return_df) == True:
        
#         total_df = []
#         for per in use_prediction_period:
#             total_df += get_return_df(per[0], per[1], run_mode, predict_metal)
        
#         res = pd.concat(total_df).reset_index(drop=True)
#         res = res.sort_values(by=['date', 'metal'], ascending=[True, True]).reset_index(drop=True)
#         return res
    
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


        



