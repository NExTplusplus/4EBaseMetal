# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:05:14 2019

@author: Kwoks
"""

import sys
import json
import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import sqlalchemy as sq
from scipy.stats import norm
from configparser import ConfigParser

sys.path.append('../other_function/')
import other_function

#this function mainly to extract the price for a certain period.
def get_price(metal_path, metal_columns, window_list, time_selection):
    '''
    :param metal_path: str, the file of the certain metal
    :param metal_columns: str, the certrain col of the metal in the dataframe

    :param time_selection:list, element is datetime, containing the start date and the end date
    
    
    :return price_forward:df, containing the original price, the rolling std.
    '''
    #this part used to select the data with certain time_selection
    price = pd.read_csv(metal_path)
    price['Index'] =  pd.to_datetime(price['Index'])
    price = price[(price['Index']<=time_selection[1])&(price['Index']>=time_selection[0])]
    
    #this part used to compute the increasing or decreasing ratio of the price
    price['return_1d'] = (price[metal_columns]/price[metal_columns].shift(1)-1)*100
    price.dropna(inplace = True)
    
    for i in window_list:
        price['return_{}d'.format(i)] = ((price[metal_columns] / price[metal_columns].shift(i)) -1)*100
       
    price_forward = price.copy()
    
    for i in window_list:
        price_forward['return_{}d'.format(i)] = price_forward['return_{}d'.format(i)].shift(-i)
    return price_forward

#get the certain sentiment result from the database.
def get_sentiment(met, time_selection, conn):
    '''
    :param metal_sentiment_path:str, the certain file path of the sentiment of a metal
    :param time_selection:list, element is datetime, containing the start date and the end date
    
    :return sentiment:df, the certrain sentiment dataframe without NaN and ordered by date
    '''
    sentiment = pd.read_sql('select * from {}_sentiment'.format(met), con=conn)
    #sentiment = pd.read_csv(metal_sentiment_path)
    sentiment['Sentiment_article'] = sentiment['Sentiment_article']*100
    sentiment['date'] = sentiment['date'].apply(lambda x: pd.to_datetime(x).floor('D'))
    
    sentiment.dropna(inplace=True)
    sentiment = sentiment[(sentiment['date']<=time_selection[1])&(sentiment['date']>=time_selection[0])]
    sentiment.sort_values('date',axis=0,inplace = True)
    return sentiment

#with the data and the boundary points, this function will divide the data accordingly.
def calculate_the_bin(x, quantile):
    '''
    :param x:int/float, data is in need of dividing.
    :param quantile:list, the quantile point
    
    :return bin:,int, the certain group of x
    '''
    assert type(x) != str, 'x is str'
    assert str not in [type(i) for i in quantile], "quantile contains str"
    
    bin = len(quantile)
    for i in range(len(quantile)):
        if x < quantile[i]:
            bin = i
            break
    return bin

#get the quantile point, for example, if you want to divide the data into 3 groups,
#it will return the percentile, such as [0.33, 0.66]
def tier_point(num_tier):
    '''
    :param num_tier:int, the certain number of the group

    :return tier:, list, the boundary points
    '''
    assert type(num_tier) == int, "the type of num_tier is not int"
    
    unit_tier = round(1/num_tier, 2)
    tier = [i * unit_tier for i in range(1, num_tier)]
    return tier

#get the quantile points, for example, if you want to divide the data into 3 groups
#it will return the quantile value, such as we have target [1, 2, 3, 4, 5], then will
#return [2.32, 3.64], where 2.32 is the 33% quantile point and 3.64 is the 66%
#quantile point.
def quan_point(tier, target):
    '''
    :param tier:list, the boundary points
    :param target:list/pd.series, object which is in need of dividing 
    
    :return:, list, the quantile points
    '''
    assert str not in [type(i) for i in tier], "tier contains str"
    assert str not in [type(i) for i in target], "target contains str"
    
    return [np.percentile(target, i*100) for i in tier]

#get the group of the certain target, here we construct this function is used to 
#use the apply function in pandas.
def discrete_method_with_point(tier_lst, target):
    '''
    :param tier_lst:list, the quantile point of a certain period
    :param target: int/float, the number need to be grouped.
    
    :return res:int, the certain group of the number, e.g. 1, 2, 4...
    '''
    #group_num = len(tier_lst) + 1
    
    res = 'not found'
    
    for i in range(len(tier_lst)):
        if target <= tier_lst[i]:
            del res
            res = i
            break
    
    if res == 'not found':
        del res
        res = len(tier_lst)
    return res

#this function will help you to turn a buch of data into the certain group
#for example, target:[1, 3, 4, 5, 6, 7], num_tier:[3], then will return 
#[0, 0, 0, 1, 1, 1]
def discrete_method(num_tier, target):
    '''
    :param num_tier:int, the certain number of the group
    :param target:list/pd.series, object which is in need of dividing 
    
    :return list(target_tier):, list, the result of dividing the data into several groups
    '''
    assert type(num_tier) == int, "the type of num_tier is not int"
    
    assert str not in [type(i) for i in target], "target contains str"
    
    #get the tier point
    tier = tier_point(num_tier)
    
    #get the quantile point of the target
    quantile_point = quan_point(tier, target)
    
    #return the group
    target_tier = map(lambda x : calculate_the_bin(x, quantile_point), target)
    return list(target_tier)

#this function will exclude the exception when the denominator is zero
def division_method(x, y):
    '''
    :param x: int/float, numerator
    :param y: int/float, denominator
    
    :return: the answer, need to judge whether the denominator is zero
    '''
    if y == 0:
        return 0
    else:
        return x/y

class Score(object):
    
    def __init__(self, 
                 metal, 
                 table_name, 
                 window_list, 
                 keep_intermediate=False):
        
        self.metal = metal
        self.window_list = window_list
        self.table_name = table_name
        self.keep_intermediate = keep_intermediate
        self.local_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
        
    #here we calculate the accuracy for each company in each day
    def cal_accur(self, price_sentiment):
        '''
        :param price_sentiment:df, the price dataframe merged with the sentiment dataframe
        
        :return no_dup_df: the final accur score
        
        '''
        #this function is comparing the discrete_score with the groundtruth
        def compare_score_target(dis_score, cur_target):
            '''
            :param dis_score:int/float, the discrete score with equally divided method,
            :param cur_target:int/float, the up/down ratio of the certain horizon.
            
            :return accur_value:list, contain different conditions of the pred and the ground truth
            '''
            accur_value = [None]*5
            
            if dis_score == 0:
                if dis_score == cur_target:
                    accur_value[2] = True
                else:
                    accur_value[2] = False
                    
            elif dis_score<0:
                
                if dis_score == cur_target:
                    accur_value[1] = True
                    accur_value[4] = False
                elif dis_score == -cur_target:
                    accur_value[4] = True
                    accur_value[1] = False
                else:
                    accur_value[1] = False
                    accur_value[4] = False
                    
            elif dis_score>0:
                
                if dis_score == cur_target:
                    accur_value[0] = True
                    accur_value[3] = False
                elif dis_score == -cur_target:
                    accur_value[3] = True
                    accur_value[0] = False
                else:
                    accur_value[0] = False
                    accur_value[3] = False
            return accur_value
        
        columns_name = ['url', 'date', 'company', 'news_type', 'score', 'discrete_score',
                        'accur_same_pos', 'accur_same_neg', 'accur_neu', 'accur_rev_pos', 'accur_rev_neg', 'prec_horizon']
        
        
        #copy the certain columns, please use copy() to make less error.
        score_df = price_sentiment[['url', 'date', 'company', 'discrete_score', 'news_type']].copy()
        score_df['date'] = score_df['date'].apply(lambda x : pd.to_datetime(x).floor('D'))
        score_df['score'] = price_sentiment['Sentiment_article'].copy()

        #do the discrete score for different window, to get discrete_1d, discrete_3d sth like that.
        for w in self.window_list:
            score_df['discrete_{}d'.format(w)] = price_sentiment['discrete_{}d'.format(w)].copy()
        score_df = score_df.reset_index(drop=True)
        
        #copy the dataframe, to get the same pieces as the length of the window_list
        copy_score_df = pd.DataFrame()
        tmp = score_df.copy()
        copy_score_df = copy_score_df.append([tmp]*len(self.window_list))
        
        #to allocate the horizon to the copied dataframe
        horizon = []
        for i in self.window_list:
            horizon = horizon + [i] * len(score_df)
        copy_score_df['prec_horizon'] = horizon
        copy_score_df = copy_score_df.reset_index(drop=True)

        accur_same_pos = []
        accur_same_neg = [] 
        accur_neu = []
        accur_rev_pos = [] 
        accur_rev_neg = []
        
        #need to update, better to use the apply to finish
        for i in tqdm(range(len(copy_score_df)), desc='cal the acc'):
            res = compare_score_target(copy_score_df['discrete_score'][i], copy_score_df['discrete_{}d'.format(copy_score_df['prec_horizon'][i])][i])
            accur_same_pos.append(res[0])
            accur_same_neg.append(res[1])
            accur_neu.append(res[2])
            accur_rev_pos.append(res[3])
            accur_rev_neg.append(res[4])
        
        #delete the discrete_1d, discrete_3d,etc. as these columns are used to compared  with the dis_score, no need to insert. 
        for w in self.window_list:
            del copy_score_df['discrete_{}d'.format(w)]

        #to assign the accur_value and adjust the columns of the dataframe.
        copy_score_df['accur_same_pos'] = accur_same_pos
        copy_score_df['accur_same_neg'] = accur_same_neg
        copy_score_df['accur_neu'] = accur_neu
        copy_score_df['accur_rev_pos'] = accur_rev_pos
        copy_score_df['accur_rev_neg'] = accur_rev_neg
        copy_score_df = copy_score_df[columns_name]
        
        #here we do a drop_duplicates, as the crawler produce some replicate title with same date, same company, same news_type,
        #when running the code, please notice the number of the replicates.
        #here the number usually will be the same.
        no_dup_df = copy_score_df.drop_duplicates(['url', 'date', 'company', 'news_type', 'prec_horizon'], keep='first')
        print('orig cop_score:{}'.format(len(copy_score_df)))
        print('no_dup cop_score:{}'.format(len(no_dup_df)))
        
        if self.keep_intermediate:
            no_dup_df.to_csv('./accur_score_intermediate/accur/' + self.table_name+'_train_accur.csv', index=False)
        
        #insert into the mysql.
#        no_dup_df.to_sql(self.table_name+'_train_accur', con=self.conn, if_exists='append', index=False, chunksize=1000)
        return no_dup_df

    def cal_score(self, 
                  price_sentiment, 
                  query_history,
                  threshold, 
                  freq_window, 
                  score_point_num, 
                  cal_date=None, 
                  discrete_param=False, 
                  predict=False):
        '''
        :param price_sentiment:dataframe, the price dataframe merged with the sentiment dataframe
        :param query_history:dataframe, the accur datafram we calculate in cal_accur function, which is used to aggregate the score.
        :param threshold:int, to decide whether we need to take the records of a company into consideration
        :param freq_window:int, to decide how many recently reports we will consider
        :param score_point_num:int, to decide how many groups we need to discretize the data
        :param cal_date:default is None, input is int, to decide how many days we need to consider excluding the predicting day.
        :param discrete_param:default is False, input is dict, to save the quantile number
        :param predict:bool, decide preidct or train.
        
        :return final_df:dataframe, return the score dataframe with confidence.
        '''
        
        #as here we need to use the accur, so you need to run the cal_accur first
#        query_history = pd.read_sql('Select * from {}_train_accur'.format(self.table_name), self.conn)
  
        #get the total date of the price sentiment
        date_list = list(price_sentiment['date'].unique())
        
        #here for a certain day, we calculate the accuracy for the company and the score accordingly
        #the way is to take the record where the date is before da-hor, and here we have a parameter
        #called freq_win, which control the maximum records we can have.here we choose 15.
        def cal_rea(qur_his, da, com, hor, dis, sen, freq_window):
            '''
            :param que_his:dataframe,  the accur we calculate in cal_acc function
            :param da:datetime, one of the date from the input price_sentiment
            :param com:str, the company name
            :param hor:int, the horzion we use
            :param dis:int, the discrete score of a certain recommendation
            :param sen:float, the sentiment article of of a certain recommendation
            :param freq_window:int, how many reports we need to consider, used in the pd.head() function.
            '''
            
            #we need to keep that we just know the time before the 'predict day'-hor, in case of data leak
            check_da = da - datetime.timedelta(hor)
            
            #here the head function will cause some random, so it is better to sort the accur before using
            tmp = qur_his[(qur_his['date']<check_da)&(qur_his['company']==com)&(qur_his['prec_horizon']==hor)&(qur_his['discrete_score']==dis)].head(freq_window)
            realibility_same = 0
            realibility_rev = 0
            if dis == 0:
                realibility_same = division_method(tmp['accur_neu'].sum(), tmp['accur_neu'].count())
            elif dis>0:
                realibility_same = division_method(tmp['accur_same_pos'].sum(), tmp['accur_same_pos'].count())
                realibility_rev = division_method(tmp['accur_rev_pos'].sum(), tmp['accur_rev_pos'].count())
            else:
                realibility_same = division_method(tmp['accur_same_neg'].sum(), tmp['accur_same_neg'].count())
                realibility_rev = division_method(tmp['accur_rev_neg'].sum(), tmp['accur_rev_neg'].count())

            if realibility_same>realibility_rev:
                realibility = np.exp(realibility_same)
                sc = sen
            elif realibility_same==realibility_rev:
                realibility = np.exp(realibility_same)
                sc=0
            else:
                realibility = np.exp(realibility_rev)
                sc = -sen
            if pd.isna(realibility):
                realibility = 0
            return [realibility, sc]
        
        #here we calculate the score and realibility for each date, and calculate the certain score
        insert_date = []
        insert_score = []
        insert_horizon = []
        
        #here we calculate the score according to the date of the price sentiment
        for dat in tqdm(date_list,desc='cal score'):
            
            #here the cal_date is how many days we will consider for one certain day, if without input, then we will just consider one certain day.
            if cal_date == None:
                score_date = price_sentiment[(price_sentiment['date']==dat)]
            else:
#                assert type(cal_date) == int, 'the param cal_date should be integer'
#                assert cal_date<0, 'the param cal_date should be negative'
                
                last_date = pd.to_datetime(dat) - datetime.timedelta(days=cal_date)
                score_date = price_sentiment[(price_sentiment['date']<=dat)&(price_sentiment['date']>=last_date)]
        
            if len(score_date)>threshold:
                for w in self.window_list:
                    insert_date.append(dat)
                    insert_horizon.append(w)
                    
                    ans = score_date.apply(lambda x: cal_rea(query_history, x['date'], x['company'], w, x['discrete_score'], x['Sentiment_article'], freq_window), axis=1)
                    
                    realibility_lst = [i[0] for i in ans]
                    score_lst = [i[1] for i in ans]
                    total_real = np.sum(realibility_lst)
                    
                    final_score = 0
                    for cur_real,cur_score in zip(realibility_lst,score_lst):
                        final_score += cur_score*cur_real
                    
                    final_score = final_score/total_real
                    insert_score.append(final_score)
        
        new_data_frame = pd.DataFrame()
        new_data_frame['date'] = insert_date
        new_data_frame['score'] = insert_score
        new_data_frame['horizon'] = insert_horizon
        
        
        #here we use the discrete method to divide the score into three groups,
        #if no discrete param is given, then it is the training mode, we will have 
        #the discrete points for the current score.
        if len(new_data_frame) >0:
            final_df = pd.DataFrame()
            for i in self.window_list:
                tmp = new_data_frame[new_data_frame['horizon']==i].copy()
                if discrete_param:
                    tmp_dis = tmp['score'].apply(lambda x : discrete_method_with_point(discrete_param['final_score_{}d'.format(i)], x))
                    tmp['discrete_score'] = [i-1 for i in list(tmp_dis)]
                else:
                    #
                    if score_point_num == 3:
                        tmp_dis = discrete_method(3, tmp['score'])
                        tmp['discrete_score'] = [i-1 for i in tmp_dis]
                    elif score_point_num == 2:
                        tmp_dis = discrete_method_with_point([0], tmp['score'])
                        tmp['discrete_score'] = [i-1 for i in tmp_dis]
                        
                final_df = final_df.append([tmp])

            final_df = final_df.reset_index(drop=True)
        else:
            final_df = pd.DataFrame(columns=['date', 'score', 'horizon', 'discrete_score'])
        
        
        if self.keep_intermediate and predict:
            final_df.to_csv('./accur_score_intermediate/score/'+self.table_name+'_predict_score.csv', index=False)
        else:
            final_df.to_csv('./accur_score_intermediate/score/'+self.table_name+'_train_score.csv', index=False)  
#        if predict:
#            final_df.to_sql(self.table_name+'_predict_score', con=self.conn, if_exists='append', index=False, chunksize=1000)
#        else:
#            final_df.to_sql(self.table_name+'_train_score', con=self.conn, if_exists='append', index=False, chunksize=1000)
        
        #if predict, then will output the csv in the predict_result, else it will keep the train score in the intermediate backup folder.
#        if predict:
#            final_df.to_csv('./predict_result/{}_{}.csv'.format(self.table_name+'_predict', self.local_time), index=False)
#        else:
#            if self.keep_intermediate:
#                final_df.to_csv('./intermediate_backup/score/{}_{}.csv'.format(self.table_name+'_predict', self.local_time), index=False)
        return final_df