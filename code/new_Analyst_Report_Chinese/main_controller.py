# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:11:48 2020

@author: Kwoks
"""
import os
import sys
import datetime

def get_half_date(input_date_period):
    '''
    :param input_date_period: str, '2018:', ':2020', '2018:2020', "::"
    :return res:[['2017-01-01', '2017-06-30']], the date we need to predict
    '''
    if input_date_period.endswith(':') and input_date_period.startswith(':'):
        process_period = '2008:' + str(datetime.datetime.now().year)

    if input_date_period.endswith(':') and not input_date_period.startswith(':'):
        process_period = input_date_period + str(datetime.datetime.now().year)
        
    if not input_date_period.endswith(':') and input_date_period.startswith(':'):
        process_period = '2008' + input_date_period
        
    if not input_date_period.endswith(':') and not input_date_period.startswith(':'):
        process_period = input_date_period
        
    
    def get_date(period):
        
        prediction_period = []
        
        cur_year = str(datetime.datetime.now().year)
        
        start_year = period.split(':')[0]
        end_year = period.split(':')[1]
        
        if end_year != cur_year:
            
            for i in range(int(start_year), int(end_year)+1):
                tmp = [['{}-01-01'.format(i), '{}-06-30'.format(i)],
                       ['{}-07-01'.format(i), '{}-12-31'.format(i)]]
                prediction_period += tmp
        else:
            for i in range(int(start_year), int(end_year)):
                tmp = [['{}-01-01'.format(i), '{}-06-30'.format(i)],
                       ['{}-07-01'.format(i), '{}-12-31'.format(i)]]
                prediction_period += tmp
            
            if datetime.datetime.now() > datetime.datetime.strptime("2020-06-30", "%Y-%m-%d"):
                
                extra = [['{}-01-01'.format(cur_year), '{}-06-30'.format(cur_year)],
                         ['{}-07-01'.format(cur_year), datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')]]
                prediction_period += extra
                
            else:
                extra = [['{}-01-01'.format(cur_year), datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')]]
                prediction_period += extra
        return prediction_period
    
    res = get_date(process_period)
    return res

def total_main_controller(data_preprocess, run_prediction, predict_period, predict_recent_days, predict_metal, run_mode):
    '''
    :param data_preprocess: bool, whether to do the step 1 to step 3
    :param run_prediction: bool, whether to do the step 4
    :param predict_period: str, the date we need to predict
    :param predict_recent_days:, str, how many recent days we need to predict
    :param predict_metal: str, the certain metal we need to predict, when it is 'all', predict all the metal
    :param run_mode: str, whether to compare the results with the labels
    '''

    if eval(data_preprocess) == True:
        
        print('--------------------------data preprocessing-------------------')
        step1_path = '/mnt/gluster/Alphien/paperTrading/Team/NEXT/4EBaseMetal/code/new_Analyst_Report_Chinese/step1_crawler/'
        step1_code = 'python crawler.py run'
        print('------------------running step 1-------------------------------')
        os.popen('cd {};{}'.format(step1_path, step1_code))
        
        step2_path = '/mnt/gluster/Alphien/paperTrading/Team/NEXT/4EBaseMetal/code/new_Analyst_Report_Chinese/step2_extract_html/'
        step2_code = 'python step2_main_contraoller.py'
        print('------------------running step 2-------------------------------')
        os.popen('cd {};{}'.format(step2_path, step2_code))
        
        step3_path = '/mnt/gluster/Alphien/paperTrading/Team/NEXT/4EBaseMetal/code/new_Analyst_Report_Chinese/step3_extract_recommendation/'
        step3_code = 'python call_aip.py run'
        print('------------------running step 3-------------------------------')
        os.popen('cd {};{}'.format(step3_path, step3_code))

    if eval(run_prediction) == True:
        
        if predict_period == 'None':
            if int(predict_recent_days) >= 1:
                end_date = datetime.datetime.now()
                start_date = end_date- datetime.timedelta(days=int(predict_recent_days)-1)
                use_prediction_period = [[datetime.datetime.strftime(start_date, '%Y-%m-%d'), datetime.datetime.strftime(end_date, '%Y-%m-%d')]]
            else:
                return 'wrong input of the parameter recent days'
        else:
            use_prediction_period = get_half_date(predict_period)
            
        step4_path = '/mnt/gluster/Alphien/paperTrading/Team/NEXT/4EBaseMetal/code/new_Analyst_Report_Chinese/step4_sentiment_analysis/'
        
        for per in use_prediction_period:
            
            os.popen('cd {};python main_function.py {} {} {} {}'.format(step4_path, per[0], per[1], predict_metal, run_mode))
    
    
if __name__ ==  '__main__':    
    
    data_preprocess = sys.argv[1]
    run_prediction = sys.argv[2]
    predict_period = sys.argv[3]
    predict_recent_days = sys.argv[4]
    predict_metal = sys.argv[5]
    run_mode = sys.argv[6]
    
    total_main_controller(data_preprocess, run_prediction, predict_period, predict_recent_days, predict_metal, run_mode)    
        
        
        