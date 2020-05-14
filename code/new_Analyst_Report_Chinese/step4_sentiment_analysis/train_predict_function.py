# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 21:29:23 2019

@author: Kwoks
"""

import sys
import datetime
import pandas as pd
import sqlalchemy as sq
import model_function as mf
from configparser import ConfigParser

sys.path.append('../other_function/')
import other_function

#this function will help to locate the predict period in which half.
#for safe, here we only allow the predict period in one half, that is all date 
#should be before xxxx-07-01(exclude) or after xxxx-07-01(include)
def find_date_in_which_half(start_date, end_date, use_half_num):
    '''
    :param start_date: datetime/str, predict period start date
    :param end_date:datetime/str, predict period end date
    :param use_half_num:int, how many half we use to predict 
    
    :return ans: the total period used to train, e.g. [2017-01-01, 2017-12-31]
    :return detail_half: the detail period used to train[[2017-01-01, 2017-06-30], [2017-07-01, 2017-12-31]]
    '''
    if type(start_date) ==str:
        start_date_format = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start_date_format = start_date
        
    if type(end_date) == str:
        end_date_format = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_date_format = end_date
    
    #we need to keep the date in the same year
    assert start_date_format.year == end_date_format.year, 'the span is too wide, please keep the predict date in the same year'
    
    #we need to keep the date in the same half year
    middle_date = datetime.datetime.strptime('{}-07-01'.format(start_date_format.year), '%Y-%m-%d')
    assert (start_date_format<middle_date)==(end_date_format<middle_date), 'the span is too wide, please keep the predict date in the same half year'
    
    
    belong_year = start_date_format.year
    if end_date_format < middle_date:
        if use_half_num == 1:
            ans = [datetime.datetime.strptime('{}-07-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-12-31'.format(belong_year-1), '%Y-%m-%d')]
            detail_half = [[datetime.datetime.strptime('{}-07-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-12-31'.format(belong_year-1), '%Y-%m-%d')]]
        elif use_half_num == 2:
            ans = [datetime.datetime.strptime('{}-01-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-12-31'.format(belong_year-1), '%Y-%m-%d')]     
            detail_half = [[datetime.datetime.strptime('{}-01-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-06-30'.format(belong_year-1), '%Y-%m-%d')],
                           [datetime.datetime.strptime('{}-07-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-12-31'.format(belong_year-1), '%Y-%m-%d')]]
        
        elif use_half_num == 3:
            ans = [datetime.datetime.strptime('{}-07-01'.format(belong_year-2), '%Y-%m-%d'), datetime.datetime.strptime('{}-12-31'.format(belong_year-1), '%Y-%m-%d')]            
            detail_half = [[datetime.datetime.strptime('{}-07-01'.format(belong_year-2), '%Y-%m-%d'), datetime.datetime.strptime('{}-12-31'.format(belong_year-2), '%Y-%m-%d')],
                           [datetime.datetime.strptime('{}-01-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-06-30'.format(belong_year-1), '%Y-%m-%d')],
                           [datetime.datetime.strptime('{}-07-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-12-31'.format(belong_year-1), '%Y-%m-%d')]]

        elif use_half_num == 4:
            ans = [datetime.datetime.strptime('{}-01-01'.format(belong_year-2), '%Y-%m-%d'), datetime.datetime.strptime('{}-12-31'.format(belong_year-1), '%Y-%m-%d')]            
            detail_half = [[datetime.datetime.strptime('{}-01-01'.format(belong_year-2), '%Y-%m-%d'), datetime.datetime.strptime('{}-06-30'.format(belong_year-2), '%Y-%m-%d')],
                           [datetime.datetime.strptime('{}-07-01'.format(belong_year-2), '%Y-%m-%d'), datetime.datetime.strptime('{}-12-31'.format(belong_year-2), '%Y-%m-%d')],
                           [datetime.datetime.strptime('{}-01-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-06-30'.format(belong_year-1), '%Y-%m-%d')],
                           [datetime.datetime.strptime('{}-07-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-12-31'.format(belong_year-1), '%Y-%m-%d')]]
    else:
        if use_half_num == 1:
            ans = [datetime.datetime.strptime('{}-01-01'.format(belong_year), '%Y-%m-%d'), datetime.datetime.strptime('{}-06-30'.format(belong_year), '%Y-%m-%d')]
            detail_half = [[datetime.datetime.strptime('{}-01-01'.format(belong_year), '%Y-%m-%d'), datetime.datetime.strptime('{}-06-30'.format(belong_year), '%Y-%m-%d')]]
        elif use_half_num == 2:
            ans = [datetime.datetime.strptime('{}-07-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-06-30'.format(belong_year), '%Y-%m-%d')] 
            detail_half = [[datetime.datetime.strptime('{}-07-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-12-31'.format(belong_year-1), '%Y-%m-%d')],
                           [datetime.datetime.strptime('{}-01-01'.format(belong_year), '%Y-%m-%d'), datetime.datetime.strptime('{}-06-30'.format(belong_year), '%Y-%m-%d')]]
        elif use_half_num == 3:
            ans = [datetime.datetime.strptime('{}-01-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-06-30'.format(belong_year), '%Y-%m-%d')]            
            detail_half = [[datetime.datetime.strptime('{}-01-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-06-30'.format(belong_year-1), '%Y-%m-%d')],
                           [datetime.datetime.strptime('{}-07-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-12-31'.format(belong_year), '%Y-%m-%d')],
                           [datetime.datetime.strptime('{}-01-01'.format(belong_year), '%Y-%m-%d'), datetime.datetime.strptime('{}-06-30'.format(belong_year), '%Y-%m-%d')]]
        elif use_half_num == 4:
            ans = [datetime.datetime.strptime('{}-07-01'.format(belong_year-2), '%Y-%m-%d'), datetime.datetime.strptime('{}-06-30'.format(belong_year), '%Y-%m-%d')]  
            detail_half = [[datetime.datetime.strptime('{}-07-01'.format(belong_year-2), '%Y-%m-%d'), datetime.datetime.strptime('{}-12-31'.format(belong_year-2), '%Y-%m-%d')],
                           [datetime.datetime.strptime('{}-01-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-06-30'.format(belong_year-1), '%Y-%m-%d')],
                           [datetime.datetime.strptime('{}-07-01'.format(belong_year-1), '%Y-%m-%d'), datetime.datetime.strptime('{}-12-31'.format(belong_year-1), '%Y-%m-%d')],
                           [datetime.datetime.strptime('{}-01-01'.format(belong_year), '%Y-%m-%d'), datetime.datetime.strptime('{}-06-30'.format(belong_year), '%Y-%m-%d')]]
    return ans, detail_half
    
#here function is used to load the model param threshold, whcih can save more time
#when using the same period.
def save_model_param():
    pass

#this function is used to settle the result, which is used to select the best threshold
#in the choose param function
def process_score_true_price(true_price, score, window_list, threshold, t_p):
    '''
    :param true price: dataframe, contain the groundtruth
    :param score: dataframe, the predict dataframe by the model
    :param window_list: list, [1, 3, 5, 10, 20, 60]
    :param threshold: int, the according threshold of score
    :param t_p:str, the starting date of the period
    
    :return res:dataframe, the statistics results of the acc
    '''
    
    t_p_str = t_p
    
    date_column = []
    threshold_column = []
    horizon_column = []
    acc_num_column = []
    left_num_column = []
    total_num_column = []
    accuracy_column = []
    
    for i in window_list:
        date_column.append(t_p_str)
        threshold_column.append(threshold)
        horizon_column.append(i)
        sc_tmp = score[score['horizon'] == i].reset_index(drop=True)
        if len(sc_tmp) == 0 :
            total_number = 0
            left_number = 0
            acc_number = 0
            acc_num_column.append(acc_number)
            left_num_column.append(left_number)
            total_num_column.append(total_number)
        else:
            tmp = true_price.merge(sc_tmp, left_on='date', right_on='date')
            total_number = len(tmp['discrete_{}d'.format(i)])
            
            tmp = tmp[(tmp['discrete_{}d'.format(i)]!=0)&(tmp['discrete_score']!=0)]
            left_number = len(tmp['discrete_{}d'.format(i)])
            
            tmp = tmp[tmp['discrete_{}d'.format(i)]==tmp['discrete_score']]
            acc_number = len(tmp['discrete_{}d'.format(i)])
            
            total_num_column.append(total_number)
            left_num_column.append(left_number)
            acc_num_column.append(acc_number)
        
        accuracy_column.append(mf.division_method(acc_number, left_number))
    res = pd.DataFrame()
    res['date'] = date_column
    res['threshold'] = threshold_column
    res['horizon'] = horizon_column
    res['acc'] = accuracy_column
    res['acc_num'] = acc_num_column
    res['left_num'] = left_num_column
    res['total_num'] = total_num_column
    return res
    
#this function is used to choose the best threshold, we provide two methods,
#one is same threshold for all the horizon, other is different threshold for different horizon.
def choose_best_threshold(window_list, threshold_list, total_df, choose_for_horizon, default_cover_ratio=0.15):
    '''
    :param window_list:list, [1, 3, 5, 10, 20, 60]
    :param threshold_list:list, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    :param total_df:dataframe, 
    :param choose_for_horizon:bool, 
    :param default_cover_ratio:float, 
    
    :return best_param:dict, the best threshold dict.
    :return res:dataframe, the acc and cover ratio dataframe used to select the best threshold
    '''
    
    
    if choose_for_horizon:
        best_param = {}
        
        res = pd.DataFrame()
        horizon_column = []
        threshold_column = []
        accuracy_column = []
        cover_ratio_column = []
        for threshold in threshold_list:
            for win in window_list:
                horizon_column.append(win)
                threshold_column.append(threshold)
                select_df = total_df[(total_df['threshold']==threshold)&(total_df['horizon']==win)]
                accuracy_column.append(mf.division_method(select_df['acc_num'].sum(), select_df['left_num'].sum()))
                cover_ratio_column.append(mf.division_method(select_df['left_num'].sum(), select_df['report_num'].sum()))
        res['threshold'] = threshold_column
        res['horizon'] = horizon_column
        res['accuracy'] = accuracy_column
        res['cover_ratio'] = cover_ratio_column
        
        for win in window_list:
            tmp = res[res['horizon']==win]
            select_cover_ratio = tmp[tmp['cover_ratio']>=default_cover_ratio]
            if len(select_cover_ratio) > 0:
                max_id = select_cover_ratio['accuracy'].idxmax()
            else:
                max_id = tmp['accuracy'].idxmax()
            best_param[win] = tmp['threshold'][max_id]
    else:
        best_param  = {}
        res = pd.DataFrame()
        threshold_column = []
        accuracy_column  = []
        cover_ratio_column = []
        
        for threshold in threshold_list:
            threshold_column.append(threshold)
            tmp = total_df[total_df['threshold']==threshold]
            
            accuracy_column.append(mf.division_method(tmp['acc_num'].sum(), tmp['left_num'].sum()))
            cover_ratio_column.append(mf.division_method(tmp['left_num'].sum(), tmp['report_num'].sum()))
        res['threshold'] = threshold_column
        res['accuracy'] = accuracy_column
        res['cover_ratio'] = cover_ratio_column
        
        select_cover_ratio = res[res['cover_ratio']>=default_cover_ratio]
        if len(select_cover_ratio) > 0:
            max_id = select_cover_ratio['accuracy'].idxmax()
        else:
            max_id = res['accuracy'].idxmax()
        for win in window_list:
            best_param[win] = res['threshold'][max_id]
    return best_param, res

#this function is used to select the best threshold for the prediction
def adjust_param(price_4e,
                 met, 
                 metal_columns,  
                 window_list,
                 train_period,
                 predict_period,
                 threshold_list,
                 freq_win,
                 repo_win,
                 use_half, 
                 whether_use_threshold_for_horizons,
                 conn,
                 adjustment_half = 2):
    '''
    :param price_4e:df, the total data we get from 4e
    :param met:str, the metal we need to predict
    :param metal_columns: str, the metal column in the LME file
    :param window_list: [1, 3, 5, 10, 20, 60]
    :param train_period:list, [datetime1, datetim2], the training period
    :param predict_period:list, [datetime1, datetim2], the predicting period
    :param threshold_list:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    :param freq_win:int, how many reports after selection we need to consider
    :param repo_win:int, how many days we need to consider for one day preiction
    :param use_half:int, how many half years we use to estimate.
    :param whether_use_threshold_for_horizons:bool, whether we need to adjust the param for each horizon
    :param conn:object, the connection between the code and the database
    :param adjustment_half: int, the half year we use to adjust
    
    :return best_param:dict, the best threshold dict.
    :return res: dataframe, the acc and cover ratio dataframe used to select the best threshold
    '''
    _, train_detail_half = find_date_in_which_half(predict_period[0], predict_period[1], adjustment_half)
    total_df = []
    
    #get the cover num
    metal_sentiment = pd.read_sql('select * from {}_sentiment'.format(met), conn)
    cover_num = {}
    for t_p in train_detail_half:
        t_p_start = datetime.datetime.strftime(t_p[0], '%Y%m%d')
        t_p_end = datetime.datetime.strftime(t_p[1], '%Y%m%d')
        selected_sentiment = metal_sentiment[(metal_sentiment['date']<=t_p[1])&(metal_sentiment['date']>=t_p[0])]
        cover_num[t_p_start+'_'+t_p_end] = len(list(set(list(selected_sentiment['date']))))
        
    for threshold in threshold_list:
        for t_p in train_detail_half:
            t_p_start = datetime.datetime.strftime(t_p[0], '%Y%m%d')
            t_p_end = datetime.datetime.strftime(t_p[1], '%Y%m%d')
            tmp_train_period, _ = find_date_in_which_half(t_p[0], t_p[1], use_half)

            discrete_param, accur = train_func(price_4e, met, metal_columns, 
                                               window_list, tmp_train_period, t_p, threshold, 
                                               freq_win, repo_win, conn)
            true_price, score = train_func_predict(price_4e, met, metal_columns, window_list, 
                                                   tmp_train_period,t_p,threshold, freq_win, 
                                                   repo_win,discrete_param,accur, conn)
            tmp_process_score_true_price = process_score_true_price(true_price, score, window_list, threshold, [t_p_start, t_p_end])
            tmp_process_score_true_price['report_num'] = cover_num[t_p_start+'_'+t_p_end]
            tmp_process_score_true_price['cover_ratio'] = tmp_process_score_true_price['left_num']/tmp_process_score_true_price['report_num']
            total_df.append(tmp_process_score_true_price)
    input_df = pd.concat(total_df)
    input_df = input_df.reset_index(drop=True)
    best_param, res = choose_best_threshold(window_list, threshold_list, input_df, choose_for_horizon=whether_use_threshold_for_horizons)
    return best_param, res
            
    
#this function is used for the prediction with the groundtruth
def train_func_predict(price_4e,
                       met, 
                       metal_columns, 
                       window_list, 
                       train_period,
                       predict_period,
                       threshold, 
                       freq_win, 
                       repo_win,
                       discrete_param,
                       accur, 
                       conn):
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
    :param discrete_param:dict, the discrete quantile dict
    :param accur:dataframe, the accuracy we calculate for the recommendation/company
    :param conn:object, the connection between the code and the database
    
    :return true_price: dataframe, contain the groundtruth
    :return scor:dataframe, the predict score of the predict period
    '''
    
    price_forward = mf.get_price(price_4e, metal_columns, window_list, [predict_period[0], predict_period[1]+datetime.timedelta(95)])
    sentiment = mf.get_sentiment(met, predict_period, conn)
    
    price_sentiment = price_forward.merge(sentiment, left_on='Index', right_on='date',how='inner')
    price_sentiment = price_sentiment[(price_sentiment['date']<=predict_period[1])&(price_sentiment['date']>=predict_period[0])].reset_index(drop=True)
    
    for i in window_list:
        res = price_sentiment['return_{}d'.format(i)].apply(lambda  x : mf.discrete_method_with_point(discrete_param['return_{}d_discrete'.format(i)], x))
        price_sentiment['discrete_{}d'.format(i)] = [j-1 for j in res]
        del res
    res = price_sentiment['Sentiment_article'].apply(lambda x :mf.discrete_method_with_point(discrete_param['Sentiment_article_discrete'], x))
    price_sentiment['discrete_score'] = [r-1 for r in res]

#    true_price = price_sentiment[['date',metal_columns, 'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d', 'return_60d',
#                                  'discrete_1d', 'discrete_3d', 'discrete_5d', 'discrete_10d', 'discrete_20d', 'discrete_60d']]
    #newly added
    true_price = price_sentiment[['date', metal_columns] + ['return_{}d'.format(i) for i in window_list] + ['discrete_{}d'.format(i) for i in window_list]]
    true_price = true_price.drop_duplicates(keep='first').reset_index(drop=True)
    
    predict_period_start = datetime.datetime.strftime(predict_period[0], '%Y%m%d')
    predict_period_end = datetime.datetime.strftime(predict_period[1], '%Y%m%d')
    table_name = '{}_{}_{}_{}_{}_{}'.format(met, threshold, freq_win, repo_win, predict_period_start, predict_period_end)

    score_class = mf.Score(met, table_name, window_list, keep_intermediate=True)

    scor = score_class.cal_score(price_sentiment, accur, threshold, freq_win, 3, repo_win, discrete_param=discrete_param, predict=True)
    
    return true_price, scor

#this fucntion is used for the prediction without the groundtruth
def predict_func(met, 
                 metal_columns, 
                 window_list, 
                 train_period,
                 predict_period,
                 threshold, 
                 freq_win, 
                 repo_win,
                 discrete_param,
                 accur, 
                 conn):
    '''
    :param met:str, the metal we need to predict
    :param metal_columns: str, the metal column in the LME file
    :param window_list: [1, 3, 5, 10, 20, 60]
    :param train_period:list, [datetime1, datetim2], the training period
    :param predict_period:list, [datetime1, datetim2], the predicting period
    :param threshold:int, the threshold we use
    :param freq_win:int, how many reports after selection we need to consider
    :param repo_win:int, how many days we need to consider for one day preiction
    :param discrete_param:dict, the discrete quantile dict
    :param accur:dataframe, the accuracy we calculate for the recommendation/company
    :param conn:object, the connection between the code and the database

    :return scor:dataframe, the predict scoreof the predict period
    '''
    #with the trained ddiscrete param, we allocate the sentiment article into the certain group
    sentiment = mf.get_sentiment(met, predict_period, conn)
    res = sentiment['Sentiment_article'].apply(lambda x :mf.discrete_method_with_point(discrete_param['Sentiment_article_discrete'], x))
    sentiment['discrete_score'] = [r-1 for r in res]
    
#    predict_df = sentiment[['date']]
#    predict_df = predict_df.drop_duplicates(keep='first').reset_index(drop=True)
    
    predict_period_start = datetime.datetime.strftime(predict_period[0], '%Y%m%d')
    predict_period_end = datetime.datetime.strftime(predict_period[1], '%Y%m%d')
    table_name = '{}_{}_{}_{}_{}_{}'.format(met, threshold, freq_win, repo_win, predict_period_start, predict_period_end)

    score_class = mf.Score(met, table_name, window_list, keep_intermediate=True)
    score = score_class.cal_score(sentiment, accur, threshold, freq_win, score_point_num=3, cal_date=repo_win, discrete_param=discrete_param, predict=True)
#    del score['score', 'horizon']
    return score

#this function is to get the accuracy and the discrete quantile of the training period.
def train_func(price_4e,
               met, 
               metal_columns,  
               window_list, 
               train_period, 
               predict_period,
               threshold, 
               freq_win, 
               repo_win,
               conn):
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

    :return discrete_param:dict, the discretization quantile
    :return scor:dataframe, the predict score of the train period
    '''
    #here we get the sentiment for the train period, as the sentiment is rolling 
    #according to the train period, then we need to use the recommends during
    #the train period to define the dividing points
    sentiment = mf.get_sentiment(met, train_period, conn)
    print('getting sentiment score')
    
    #here we get the price for the train period, here we may miss some data at the 
    #bottom because the bottom data we can not get the future result, hence the return
    #price would be NaN
    #window_list could be [1, 3, 5, 10, 20, 60]
    price_forward = mf.get_price(price_4e, metal_columns, [1, 3, 5, 10, 20, 60], train_period)
    print('the length before price_forward merge is : {}'.format(len(price_forward)))
    
    #here we define the discrete points for the return price and the sentiment_article
    #which are based on the train period data.
    #newly added
    discrete_columns = ['return_{}d'.format(i) for i in window_list]
    
#    discrete_columns = ['return_1d', 'return_3d', 'return_5d',
#                        'return_10d', 'return_20d', 'return_60d']
    discrete_param = {}
    for i in discrete_columns:
        tier_list = mf.tier_point(3)
        quan_list = mf.quan_point(tier_list, price_forward[~pd.isna(price_forward[i])][i])
        discrete_param[i+'_discrete'] = quan_list
    for i in ['Sentiment_article']:
        tier_list = mf.tier_point(3)
        quan_list = mf.quan_point(tier_list, sentiment[~pd.isna(sentiment[i])][i])
        discrete_param[i+'_discrete'] = quan_list
    
    #after merging the data, we need to check the length of it.
    price_sentiment = price_forward.merge(sentiment, left_on='Index', right_on='date',how='inner')
    price_sentiment.drop(['Index','title','{}_fact'.format(met),'{}_action'.format(met),'{}_new_action'.format(met),'Sentiment'],axis=1,inplace = True)
    print('the length after price_forward merge is : {}'.format(len(price_sentiment)))
    
    #here we need to drop the nan columns of the price_sentiment, and also, 
    #we need to guarantee the data doesn't have duplicates.
    price_sentiment = price_sentiment.dropna()
    print('the original length of the dataframe : {}'.format(len(price_sentiment)))
    price_sentiment.drop_duplicates(keep='first', inplace=True)
    print('the processsed length of the dataframe : {}'.format(len(price_sentiment)))
    
    #divide the price data(after dropna and drop_duplicates) into 3 groups.
    for i in window_list:
        res = price_sentiment['return_{}d'.format(i)].apply(lambda  x : mf.discrete_method_with_point(discrete_param['return_{}d_discrete'.format(i)], x))
        price_sentiment['discrete_{}d'.format(i)] = [j-1 for j in res]
        del res 
    
    #divide the sentiment data into 3 groups
    #and the last step help to change the classification into [-1, 0, 1]
    res = price_sentiment['Sentiment_article'].apply(lambda x :mf.discrete_method_with_point(discrete_param['Sentiment_article_discrete'], x))
    price_sentiment['discrete_score'] = [i-1 for i in res]
    del res
    
    #define the table name, which maybe saved in the inermediate backup
    train_period_start = datetime.datetime.strftime(train_period[0], '%Y%m%d')
    train_period_end = datetime.datetime.strftime(train_period[1], '%Y%m%d')    
    
    predict_period_start = datetime.datetime.strftime(predict_period[0], '%Y%m%d')
    predict_period_end = datetime.datetime.strftime(predict_period[1], '%Y%m%d')
    table_name = '{}_{}_{}_{}_{}_{}'.format(met, threshold, freq_win, repo_win, predict_period_start, predict_period_end)
    
    score_class = mf.Score(met, table_name, window_list, keep_intermediate=True)
    
    accur = score_class.cal_accur(price_sentiment)
    accur = accur.sort_values(['url', 'date', 'news_type', 'company', 'prec_horizon'])
    
    score = score_class.cal_score(price_sentiment, accur, threshold, freq_win, score_point_num=3, cal_date=repo_win)
    
    #we need to have the final score quantile to apply in the predicting period.
    for i in window_list:
        tmp = score[score['horizon']==i].copy()
        if len(tmp) != 0:
            tier_list = mf.tier_point(3)
            quan_list = mf.quan_point(tier_list, tmp['score'])
            discrete_param['final_score_{}d'.format(i)] = quan_list
        else:
            discrete_param['final_score_{}d'.format(i)] = [0]
    other_function.dump_json(discrete_param, './discrete_param/{}/{}_{}_{}_{}_{}_{}.json'.format(met, met, train_period_start, train_period_end, threshold, freq_win, repo_win))
    
    return discrete_param, accur