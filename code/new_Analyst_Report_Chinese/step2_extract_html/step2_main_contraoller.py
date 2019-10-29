# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:18:59 2019

@author: Kwoks
"""

import sys
import time
import pandas as pd
import sqlalchemy as sq
from configparser import ConfigParser

sys.path.append('../other_function/')
import other_function

from recommend_extracter import recommend_extracter
from html_extracter import html_extracter


if __name__ == '__main__':
    
    config_path = './step2_data/config.ini'
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
    
    
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    html_error_path = './step2_data/html/error_log_{}.json'.format(current_time)
    recommend_error_path = './step2_data/recommend/error_log_{}.json'.format(current_time)
    print('####################################################################')
    print('begin to extract the info from the html')
    extracter = html_extracter(conn)
    df_crawl = pd.read_sql('Select * from html', con=conn)
    
    problem = extracter.extract(df_crawl)
    
    if problem != {}:
        other_function.dump_json(problem, html_error_path)
        print('error found, please check the error file:{}'.format(html_error_path))
    else:
        print('no error found')
    print('finish extracting the info from the html')
    print('####################################################################')
          
          
    print('begin to extract the recommend from the content')
    recommend = recommend_extracter(conn)
    df_content = pd.read_sql('Select * from content',conn)
    
    keyword = ['震荡','偏强','观望','做多','轻仓','反弹','偏弱','上涨','企稳','承压','卖出','短线','短多','整理','止损',
               '多仓','突破','支持','上行','空间','回补','低位','悲观',
               '回落','弱势','抛售','回调','有望','走高','多单','上移','多头','走强','盘整','波动','上升','支撑','空单']
    first = ['认为','预计','预测','预期','建议','观点','关注','强调','交易','铜价','多头','空头']
    secondary = ['操作','短期','短线']
    split = ['。','；']
    stop_words = ['\t']
    
    problem = recommend.extract(df_content,keyword,first,secondary,split,stop_words=stop_words)
    
    if problem != {}:

        other_function.dump_json(problem, recommend_error_path)
        print('error found, please check the error file:{}'.format(recommend_error_path))
    else:
        print('no error found')
    print('####################################################################')