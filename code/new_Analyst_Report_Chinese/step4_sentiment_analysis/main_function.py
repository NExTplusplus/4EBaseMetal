# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 21:39:31 2019

@author: Kwoks
"""
import sys
import datetime
import pandas as pd
import sqlalchemy as sq
import model_function as mf
import train_predict_function as tpf
from configparser import ConfigParser

sys.path.append('../other_function/')
import other_function



#this function is used to the daily prediction
def main_controller(price_4e,
                    met, 
                    metal_columns, 
                    window_list, 
                    train_period, 
                    predict_period,
                    threshold, 
                    freq_win, 
                    repo_win,
                    conn
                    ):
    '''
    :param price_4e:df, the total data we get from 4e
    :param met:str, the metal we need to predict
    :param metal_columns: str, the metal column in the LME file
    :param window_list: [1, 3, 5, 10, 20, 60]
    :param train_period:list, [datetime1, datetim2], the training period
    :param predict_period:list, [datetime1, datetim2], the predicting period
    :param threshold:int, the threshold we use
    :param freq_win:int, how many reports after selection we need to consider
    :param repo_win:int, how many days we need to consider for one day preiction
    :param conn:object, the connection between the code and the database

    :return ans:dataframe, the predict score of the predict period
    '''
#    train_period_start = datetime.datetime.strftime(train_period[0], '%Y%m%d')
#    train_period_end = datetime.datetime.strftime(train_period[1], '%Y%m%d')
#    discrete_param_path = './discrete_param/{}/{}_{}_{}_{}_{}_{}.json'.format(met, met, train_period_start, train_period_end, threshold, freq_win, repo_win)
#    if os.path.exists(discrete_param_path):
#        discrete_param = other_function.load_json(discrete_param_path)
#    else:
#        discrete_param = train_func(met, metal_columns, metal_path, 
#                                    window_list, train_period, threshold, 
#                                    freq_win, repo_win, conn)
    #get the discrete quantile and the accuracy.
    discrete_param, accur = tpf.train_func(price_4e, met, metal_columns, 
                                           window_list, train_period, predict_period, threshold, 
                                           freq_win, repo_win, conn)

    #compute the predicting period score.
    ans = tpf.predict_func(met, metal_columns, window_list, 
                           train_period,predict_period,threshold, freq_win, 
                           repo_win,discrete_param,accur, conn)
    return ans
    ###########################################################################

#this function is specified for the reproduction of the online testing
def online_reproduction(price_4e,
                        met, 
                        metal_columns, 
                        window_list, 
                        train_period, 
                        predict_period,
                        threshold, 
                        freq_win, 
                        repo_win,
                        conn
                        ):
    '''
    :param price_4e:df, the total data we get from 4e
    :param met:str, the metal we need to predict
    :param metal_columns: str, the metal column in the LME file
    :param window_list: [1, 3, 5, 10, 20, 60]
    :param train_period:list, [datetime1, datetim2], the training period
    :param predict_period:list, [datetime1, datetim2], the predicting period
    :param threshold:int, the threshold we use
    :param freq_win:int, how many reports after selection we need to consider
    :param repo_win:int, how many days we need to consider for one day preiction
    :param conn:object, the connection between the code and the database

    :return ans:dataframe, the predict score of the predict period
    '''
#    train_period_start = datetime.datetime.strftime(train_period[0], '%Y%m%d')
#    train_period_end = datetime.datetime.strftime(train_period[1], '%Y%m%d')
#    discrete_param_path = './discrete_param/{}/{}_{}_{}_{}_{}_{}.json'.format(met, met, train_period_start, train_period_end, threshold, freq_win, repo_win)
#    if os.path.exists(discrete_param_path):
#        discrete_param = other_function.load_json(discrete_param_path)
#    else:
#        discrete_param = train_func(met, metal_columns, metal_path, 
#                                    window_list, train_period, threshold, 
#                                    freq_win, repo_win, conn)
    discrete_param, accur = tpf.train_func(price_4e, met, metal_columns, 
                                           window_list, train_period, predict_period, threshold, 
                                           freq_win, repo_win, conn)
    true_price, ans = tpf.train_func_predict(price_4e, met, metal_columns, window_list, 
                                             train_period,predict_period,threshold, freq_win, 
                                             repo_win,discrete_param,accur, conn)
    if len(true_price) == 0:
        print('the label is empty, code will only output the predictions')
        true_pred_df = ans
    else:
        true_pred_df = true_price.merge(ans, left_on='date', right_on='date')
    return true_pred_df

if __name__ ==  '__main__':
    
    #here the two date is the including date, for example, if you enter '2019-08-01'
    #and '2019-08-02', then the date range includes these two boundries.Hence if you
    #want to predict one day, for example, you need to enter '2019-08-01' and '2019-08-01'
    predict_start_date = sys.argv[1]
    predict_end_date = sys.argv[2]
    predict_mode = sys.argv[3]

    config_path = './step4_data/config.ini'
    conf = ConfigParser()
    conf.read(config_path)
    
    #default_param
    metal_list = eval(conf.get('default_param', 'metal_list'))
    window_list = eval(conf.get('default_param', 'window_list'))
    
    #construct the path of the metal
    metal_dict = {}
    metal_dict['Cu'] = ['LMCADY']
    metal_dict['Al'] = ['LMAHDY']
    metal_dict['Zn'] = ['LMZSDY']
    metal_dict['Pb'] = ['LMPBDY']
    metal_dict['Ni'] = ['LMNIDY']
    metal_dict['Xi'] = ['LMSNDY']
    
    #database param
    use_account = conf.get('database_param', 'account')
    use_psw = conf.get('database_param', 'password')
    use_host = conf.get('database_param', 'host')
    use_port = conf.get('database_param', 'port')
    use_database = conf.get('database_param', 'database')
    
    engine = sq.create_engine("mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8".format(use_account, use_psw, use_host, use_port, use_database))
    conn = engine.connect()

    #get the hyper param
    hyper_path = './step4_data/hyper_param.json'
    
    whether_retrain = conf.get('predict_param', 'whether_retrain')
    for met in metal_list:
        
        metal = met
        print('current metal is : {}'.format(met))
        
        metal_columns = metal_dict[metal][0]
        print('use columns:{}'.format(metal_columns))
        
        #get the data from 4e
        price_4e = mf.get_4e_data(metal_columns)        
        
        threshold_lst = eval(conf.get('predict_param', 'default_threshold'))
        #here we define the parameter for predicting certain metal in long term
        long_term_freq_win = eval(conf.get('predict_param', 'long_term_freq_window'))
        long_term_repo_win = eval(conf.get('predict_param', 'long_term_report_day'))
        long_term_predict_half = eval(conf.get('predict_param', 'long_term_predict_half'))
        long_term_whether_use_threshold_for_horizons = eval(conf.get('predict_param', 'long_term_whether_adjust_for_horizon'))
         
        #here we define the parameter for predicting certain metal in short term
        short_term_freq_win = eval(conf.get('predict_param', 'short_term_freq_window'))
        short_term_repo_win = eval(conf.get('predict_param', 'short_term_report_day'))
        short_term_predict_half = eval(conf.get('predict_param', 'short_term_predict_half'))
        short_term_whether_use_threshold_for_horizons = eval(conf.get('predict_param', 'short_term_whether_adjust_for_horizon'))

        short_term_horizon = [1, 3, 5]
        long_term_horizon = [10, 20, 60]
        
        predict_result_name = '{}_{}_{}'.format(met, predict_start_date.replace('-', ''), predict_end_date.replace('-', ''))
        #short_term prediction
        train_period,_ = tpf.find_date_in_which_half(predict_start_date, predict_end_date, short_term_predict_half)
        predict_period = [datetime.datetime.strptime(predict_start_date, '%Y-%m-%d'), datetime.datetime.strptime(predict_end_date, '%Y-%m-%d')]
        
        hyper_param = other_function.load_json(hyper_path)
        short_train_period_str0 = datetime.datetime.strftime(train_period[0], '%Y%m%d')
        short_train_period_str1 = datetime.datetime.strftime(train_period[1], '%Y%m%d')
        short_period_key = short_train_period_str0+'_'+short_train_period_str1        

        if short_period_key in hyper_param['short_term'][met].keys() and not eval(whether_retrain):
            best_param_tmp = hyper_param['short_term'][met][short_period_key]
            best_param = {}
            for key, val in best_param_tmp.items():
                best_param[eval(key)] = eval(val)
        else:
            best_param, res = tpf.adjust_param(price_4e, met,  metal_columns,  
                                               short_term_horizon,train_period,predict_period,
                                               threshold_lst,short_term_freq_win,short_term_repo_win,
                                               short_term_predict_half, 
                                               short_term_whether_use_threshold_for_horizons,conn)        
            res.to_csv('./adjustment_intermediate/{}/{}_{}_{}_short_term_adjustment.csv'.format(met, met, predict_start_date, predict_end_date), index=False)
            best_param_out = {}
            for key, val in best_param.items():
                best_param_out[str(key)] = str(val)
            hyper_param['short_term'][met][short_period_key] = best_param_out
            other_function.dump_json(hyper_param, hyper_path)            
        
        for hor, best_threshold in best_param.items():
            if predict_mode == 'reproduction':
                ans = online_reproduction(price_4e, met, metal_columns,  
                                          [hor], train_period, predict_period,
                                          best_threshold, short_term_freq_win, short_term_repo_win,conn)  
            elif predict_mode == 'run':
                ans = main_controller(price_4e, met, metal_columns,  
                                      [hor], train_period, predict_period,
                                      best_threshold, short_term_freq_win, short_term_repo_win,conn)  
            ans.to_csv('./predict_result/{}/{}/{}_{}_{}_{}.csv'.format(met, hor, predict_result_name, best_threshold, hor, predict_mode), index=False)
        
        #long_term prediction
        train_period,_ = tpf.find_date_in_which_half(predict_start_date, predict_end_date, long_term_predict_half)
        predict_period = [datetime.datetime.strptime(predict_start_date, '%Y-%m-%d'), datetime.datetime.strptime(predict_end_date, '%Y-%m-%d')]
        
        hyper_param = other_function.load_json(hyper_path)
        long_train_period_str0 = datetime.datetime.strftime(train_period[0], '%Y%m%d')
        long_train_period_str1 = datetime.datetime.strftime(train_period[1], '%Y%m%d')
        long_period_key = long_train_period_str0+'_'+long_train_period_str1
        
        if long_period_key in hyper_param['long_term'][met].keys() and not eval(whether_retrain):
            best_param_tmp = hyper_param['long_term'][met][long_period_key]
            best_param = {}
            for key, val in best_param_tmp.items():
                best_param[eval(key)] = eval(val)
        else:        
            best_param, res = tpf.adjust_param(price_4e, met,  metal_columns,  
                                               long_term_horizon,train_period,predict_period,
                                               threshold_lst,long_term_freq_win,long_term_repo_win,
                                               long_term_predict_half, 
                                               long_term_whether_use_threshold_for_horizons,conn)        
            res.to_csv('./adjustment_intermediate/{}/{}_{}_{}_long_term_adjustment.csv'.format(met, met, predict_start_date, predict_end_date), index=False)
            best_param_out = {}
            for key, val in best_param.items():
                best_param_out[str(key)] = str(val)
            hyper_param['long_term'][met][long_period_key] = best_param_out
            other_function.dump_json(hyper_param, hyper_path)
            
        for hor, best_threshold in best_param.items():
            if predict_mode == 'reproduction':
                ans = online_reproduction(price_4e, met, metal_columns,  
                                          [hor], train_period, predict_period,
                                          best_threshold, long_term_freq_win, long_term_repo_win,conn)  
            elif predict_mode == 'run':
                ans = main_controller(price_4e, met, metal_columns,  
                                      [hor], train_period, predict_period,
                                      best_threshold, long_term_freq_win, long_term_repo_win,conn)  
            ans.to_csv('./predict_result/{}/{}/{}_{}_{}_{}.csv'.format(met, hor, predict_result_name, best_threshold, hor, predict_mode), index=False)