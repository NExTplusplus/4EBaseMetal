# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:51:43 2019

@author: Kwoks
"""
import re
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from aip import AipNlp
import sqlalchemy as sq
from configparser import ConfigParser
sys.path.append('../other_function/')
import other_function

#mainly deal with the data table containg the certain metal and exclude sentences without certain metal
def preprocess(met, met_cn, df):
    '''
    :param met:str, the certain metal, like 'Cu'
    :param df:dataframe, the original dataframe extracted from step2
    
    :return df:dataframe, the dataframe after processing
    '''
    
    tmp_data = df['{}_action'.format(met)]
    new_col = []
    
    for i in tmp_data:
        only_met = [ sen for sen in re.split('\.|。', i) if met_cn in sen and ' {} '.format(met_cn) not in sen]
        new_col.append(only_met)
    df['{}_new_action'.format(met)] = new_col 
    return df

#mainly extract the data we need
def get_metal_df(met, recommend):
    '''
    :param met: str, e.g. Cu, Pb
    :param recommend: df, the total recommend df in the database, created by step 2
    
    :return df:, df, datafram with the columns we need
    '''    
    tmp = recommend[['url', 'company', 'news_type', 'published_date',
                    'date', 'title', met + '_fact', met + '_action']]
    
    df = tmp.copy()
    
    df.dropna(inplace = True)
    
    df[met + '_action'] = df[met + '_action'].apply(lambda x : '。'.join(x.split('\n')[1:]))
    
    df.sort_values(by = ['published_date'], inplace = True)
    
    df.reset_index(inplace =  True, drop = True)
    
    return df

#build the new table, (need to be update)
def build_sentiment_article(con, met):
    '''
    :param con:object, sth used to link the database
    :param met:str, the metal names, like 'Cu', 'Zn',
    :param database:str, the database we use, it is not advised to use different for different steps
    '''
    con.execute("CREATE TABLE `{}`(`url` varchar(700) NOT NULL,`id` int(11) NOT NULL AUTO_INCREMENT,`company` varchar(20) DEFAULT NULL,`news_type` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,`published_date` datetime DEFAULT NULL,`date` datetime DEFAULT NULL,`title` varchar(100) DEFAULT NULL,`{}_fact` mediumtext COMMENT '\n',`{}_action` mediumtext CHARACTER SET utf8 COLLATE utf8_general_ci,`{}_new_action` mediumtext CHARACTER SET utf8 COLLATE utf8_general_ci, `Sentiment`  mediumtext CHARACTER SET utf8 COLLATE utf8_general_ci, `Sentiment_article` FLOAT NULL,PRIMARY KEY (`url`),KEY(`id`));".format('{}_sentiment'.format(met), met, met, met))

#change the type of some columns, please use this function carefully
def change_type(df):
    '''
    :param df:df, the dataframe we need to change the column type
    
    :return df:df, the dataframe we have changed the column type
    '''
    
    stable_columns = ['published_date', 'date', 'Sentiment_article']
    
    for col in df.columns:
        if col not in stable_columns:
            df[col] = df[col].apply(lambda x: str(x))
            
    return df


#using the aip to get the sentiment score of the text
def calculate_sentiment(inputs, client, clean_words = None):
    '''
    :param inputs:list and elements are str, after preprocessing, the inputs should contain several sentences
    :param client:object, the aip client
    :param clean_words:list, if you want to clean some words we do not need.
    
    :return result:list, the result of each sentence of the inputs
    :return error:list, the error appear when we use the aip
    
    '''
    result =[]
    error = []
    existing_clean_words = [u'\xc1',u'\u2022',u'\ufeff']
    if clean_words:
        existing_clean_words += clean_words
    try:
        for sentence in inputs:
            for char in existing_clean_words:
                sentence = sentence.replace(char,"")
            output = client.sentimentClassify(sentence)
            result.append(output['items'][0])
    except Exception as e: 
        error.append(str(e))
    return result, error

#combine the result of an analyst report, as we each report we will divide it 
#into several sentences, then we need to combine the result of each sentence.
def sen_art(inputs):
    '''
    :param inputs:list, the results we get from the aip
    
    :return:float, the combined result
    '''
    result = 0
    for i in inputs:
        result+=(i['positive_prob']-i['negative_prob'])*i['confidence']
    if len(inputs)==0:
        return np.nan
    else:
        return result/len(inputs)


def main_function(metal_lst, metal_dict, recommend, conn, client):
    '''
    :param metal_lst: list and elements are str, the metal list, like ['Cu', 'Pb'....]
    :param metal_dict: dict, the dictinary contains the content relevant to the certain metal, like {'Cu':'铜'}...
    :param recommend: df, the total dataframe we have extracted from the html, finished in step2
    :param conn: object, the object link the database
    :param client: object, the client of the aip
    
    :return raise_error: dict, this dictionary will contain all the error of the certain url, which we can check after running the code.
    '''
    
    
    raise_error = {}
    for met in metal_lst:
        
        #this part is to check whether we can find the table, if not then we will create one
        result = conn.execute("SHOW TABLES LIKE '{}';".format(met+'_sentiment'))
        if not result.first():
            print('can not find {}_sentiment, will create it'.format(met))
            build_sentiment_article(conn, met)
        
        #these two will load the sentiment we have in the database, and all the recommendation of certain metal we have
        local_met_sentiment = pd.read_sql('select * from {}_sentiment'.format(met), con=conn)
        current_met_sentiment = get_metal_df(met, recommend)
        
        #this part will exclude the url which we have used the aip to get the sentiment score.
        wait_met_sentiment = current_met_sentiment[~current_met_sentiment['url'].isin(list(local_met_sentiment['url']))].copy()
        wait_met_sentiment = wait_met_sentiment.reset_index(drop=True)
        
        #this part we will do the post-process to clean some table of single word, then we will
        #exclude the empty columns so as to save the time, (exclude such as [] after we do the post-process)
        post_pre_sentiment = preprocess(met, metal_dict[met], wait_met_sentiment)
        post_pre_sentiment = post_pre_sentiment[post_pre_sentiment['{}_new_action'.format(met)].str.len()!=0].reset_index(drop=True)
        
        #in this part we will insert the result into the mysql
        ###need to be updated, insert partially into the mysql if the code crashes.
        metal_sentiment = []
        for i in tqdm(range(len(post_pre_sentiment)), desc = 'using aip for {}'.format(met)):
            res, err = calculate_sentiment(post_pre_sentiment[met + '_new_action'][i], client)
            time.sleep(1)
            metal_sentiment.append(res)
            raise_error[post_pre_sentiment['url'][i]] = err
            
            tmp_to_sql_df = pd.DataFrame(post_pre_sentiment.loc[i, list(post_pre_sentiment.columns)]).T.reset_index(drop=True)
            tmp_to_sql_df['published_date'] = tmp_to_sql_df['published_date'].apply(lambda x: pd.to_datetime(x).floor('D'))
            tmp_to_sql_df['date'] = tmp_to_sql_df['date'].apply(lambda x: pd.to_datetime(x).floor('D'))
            tmp_to_sql_df['Sentiment'] = [res]
            tmp_to_sql_df['Sentiment_article'] = tmp_to_sql_df['Sentiment'].apply(lambda x: sen_art(x))
            
            tmp_filter_df = tmp_to_sql_df[tmp_to_sql_df['Sentiment_article']!=np.nan].copy().reset_index(drop=True)
            if len(tmp_filter_df)>0:
                tmp_filter_df = change_type(tmp_filter_df)
                tmp_filter_df.to_sql('{}_sentiment'.format(met), con=conn, if_exists='append', index=False, chunksize=1000)
        #post_pre_sentiment['Sentiment'] = metal_sentiment
        #post_pre_sentiment['Sentiment_article'] = post_pre_sentiment['Sentiment'].apply(lambda x: sen_art(x))
        
        #filter_pre_sentiment = post_pre_sentiment[post_pre_sentiment['Sentiment_article']!=np.nan].copy().reset_index(drop=True)
  
        #filter_pre_sentiment = change_type(filter_pre_sentiment)
        
        #filter_pre_sentiment.to_sql('{}_sentiment'.format(met), con=conn, if_exists='append', index=False, chunksize=1000)
        
    print('completed')
    return raise_error

#run is to do the daily task
def run_function(metal_lst, metal_dict, recommend, conn, client, error_path):
    '''
    :param metal_lst: list and elements are str, the metal list, like ['Cu', 'Pb'....]
    :param metal_dict: dict, the dictinary contains the content relevant to the certain metal, like {'Cu':'铜'}...
    :param recommend: df, the total dataframe we have extracted from the html, finished in step2
    :param conn: object, the object link the database
    :param client: object, the client of the aip
    :param error_path:str, the path we need to record the error
    '''
    
    
    already_error = other_function.load_json(error_path)

    raise_error = main_function(metal_lst, metal_dict, recommend, conn, client)
    
    if already_error != {}:
        
        for k, v in already_error.items():
            
            if k in raise_error.keys():
                pass
            else:
                raise_error[k] = already_error[k]
                
    other_function.dump_json(raise_error, error_path)
    if raise_error != {}:
        print('find some error, please check it')    
    print('completed')
    
                
#check is to retry the error url
def check_function(metal_lst, metal_dict, recommend, conn, client, error_path):
    '''
    :param metal_lst: list and elements are str, the metal list, like ['Cu', 'Pb'....]
    :param metal_dict: dict, the dictinary contains the content relevant to the certain metal, like {'Cu':'铜'}...
    :param recommend: df, the total dataframe we have extracted from the html, finished in step2
    :param conn: object, the object link the database
    :param client: object, the client of the aip
    :param error_path:str, the path we need to record the error
    '''
    
    already_error = other_function.load_json(error_path)
    
    if already_error == {}:
        print('no error, no need to check')
        return 
    else:
        recommend = recommend[recommend['url'].isin(already_error)]
        
    raise_error = main_function(metal_lst, metal_dict, recommend, conn, client)
    
    other_function.dump_json(raise_error, error_path)

    print('completed')
    


if __name__ == '__main__':
    
    switch = sys.argv[1]
    
    config_path = './step3_data/config.ini'
    conf = ConfigParser()
    conf.read(config_path)
    
    #load the database param and construct the link
    use_account = conf.get('database_param', 'account')
    use_psw = conf.get('database_param', 'password')
    use_host = conf.get('database_param', 'host')
    use_port = conf.get('database_param', 'port')    
    use_database = conf.get('database_param', 'database')
    
    engine = sq.create_engine("mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8".format(use_account, use_psw, use_host, use_port, use_database))
    conn = engine.connect()
    
    #load the aip param and construct the aip client
    app_id = conf.get('aip_param', 'app_id')
    api_key = conf.get('aip_param', 'api_key')
    secret_key = conf.get('aip_param', 'secret_key')
    client = AipNlp(app_id, api_key, secret_key)
    
    #not flexible, need to be updated
    error_path = './step3_data/error_recommend.json'
    recommend = pd.read_sql('select * from recommend', con=conn)
    recommend = recommend.loc[recommend['published_date'].apply(lambda x: not isinstance(x,str))]
    
    #here we deal with the wrong date
    dat_type = [str(type(i)) for i in recommend['date']]
    recommend['dat_type'] = dat_type
    recommend = recommend[recommend['dat_type'].str.contains('time')]
    recommend = recommend.reset_index(drop=True)
    del recommend['dat_type']
    
    #here we delete the id is because the id is the auto increment column of the table
    del recommend['id']
    
    #not flexible, need to be updated
    metal_lst = ['Pb', 'Zn', 'Al', 'Ni', 'Xi', 'Cu']
    metal_dict = {'Al':'铝', 'Zn':'锌', 'Cu':'铜', 'Ni':'镍', 'Xi':'锡', 'Pb':'铅'}
    
    #two mode of running, run is to do the daily task, check  is to retry the error url
    
    #p.s. here i divide it into two function is because maybe later the logic will become complicated, 
    #which means each time we need to spend more time to know what we write before, hence I split it into
    #two function, each is independent, which will save time.
    if switch == 'run':
        run_function(metal_lst, metal_dict, recommend, conn, client, error_path)
        
    elif switch == 'check':
        check_function(metal_lst, metal_dict, recommend, conn, client, error_path)

        
    
