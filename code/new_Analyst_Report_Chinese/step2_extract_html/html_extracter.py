# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 11:10:41 2019

@author: Kwoks
"""
import re
import datetime
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from newspaper import fulltext

class html_extracter:
    def __init__(self,conn,build_db_content = False):
        
        self.conn_extracter = conn
        if build_db_content:
            self.build_content_db()
    
    def build_content_db(self):
        # Function: Set up a database to store accuracy with the following setting. 
        # Note that: All functions in this class will follow this setting, pls set up ur database accordingly to avoid error
        self.conn_extracter.execute('CREATE TABLE `content`(`url` VARCHAR(750) NOT NULL,`id` INT NOT NULL AUTO_INCREMENT,`published_date` DATETIME NOT NULL, `date` DATETIME NOT NULL, `company` VARCHAR(30) NULL,`type` VARCHAR(45) NULL,`title` TINYTEXT NULL,`content` MEDIUMTEXT NULL,PRIMARY KEY (`url`),KEY(`id`));')
    
    def extract(self,df_news):
        # Function: This function will extract content from html
        # Inputs: df_news is the crawled report dataframe gotten from live crawler, date is today date
        # function_content is the fucntion to extract content from specific html style
        
        
        # Check whether table in database has been created 
        result = self.conn_extracter.execute("SHOW TABLES LIKE 'content';")
        
        if  not result.first():
#            raise Exception('Database not exist, please use build_content_db function')
            print('can not find the content table, will create it automatically')
            self.conn_extracter.execute('CREATE TABLE `content`(`url` VARCHAR(750) NOT NULL,`id` INT NOT NULL AUTO_INCREMENT,`published_date` DATETIME NOT NULL, `date` DATETIME NOT NULL, `company` VARCHAR(30) NULL,`type` VARCHAR(45) NULL,`title` TINYTEXT NULL,`content` MEDIUMTEXT NULL,PRIMARY KEY (`url`),KEY(`id`));')

        # Record data that has problems when we try to extract
        already_url = list(pd.read_sql('select * from content', con=self.conn_extracter)['url'])
        df_news = df_news[~df_news['url'].isin(already_url)].reset_index(drop=True)
        
        problem={}
        for news_id,title,type1, name,url,html in tqdm(zip(df_news['id'],df_news['title'],df_news['type'],df_news['company'],df_news['url'],df_news['html']), desc='extracting content'):
            try:
                new_input = {}
                extract_date = self.function_date(url, html)
                new_input['published_date'] = [extract_date]
                new_input['date'] = [datetime.datetime.strftime(datetime.date.today(), '%Y-%m-%d')]
                new_input['company'] = [name]
                new_input['type'] = [type1]
                new_input['title'] = [title]
                new_input['url'] = [url]
                    
                function_content = self.choose_func(url)
                    
                #Extract Content
                text = function_content(html)
                    
                if len(text)==0:
                    #raise Exception('Empty')
                    new_input['Content'] = ['empty']
                else:
                    new_input['Content'] = [text]
                
                    
                if len(text)==0 or extract_date == 'error_page' or extract_date == 'not found date':
                    pass
                else:
                    df_input = pd.DataFrame(new_input)
                    df_input.to_sql(name='content', con=self.conn_extracter, if_exists='append',index=False)
                
        
            except Exception as e:
                # If still have problem, record it
                #print('Problem found',e)
                if str(e) in problem.keys():
                    problem[str(e)].append(news_id)
                else:
                    problem[str(e)] = [news_id]

        
#        current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
#        other_function.dump_json(problem, './step2_data/html/error_log_{}.json'.format(current_time))
        
        print ('Completed')
        return problem 
    
    def choose_func(self,url):
        # Function: this function will return a suitable function that can extract html accordingly
        fulltext_lst = ['gtaxqh','htfc','szjhqh','btqh','hlqh','fnqh','cnzsqh','ftol','dlqh','cnhtqh','mfc','gzjkqh','shqhgs','zcqh','tqfutures','shcifco','rdqh','jtqh','hhqh','bhfcc']
        keyword = url.split('.')[1]
        if keyword in fulltext_lst:
            func = lambda x: fulltext(x, 'zh')
        elif 'bocifco' in keyword:
            func = lambda x: BeautifulSoup(x, 'lxml').find("div", {"class": "contArticle_text"}).text
        elif 'xzfutures' in keyword:
            func = lambda x: BeautifulSoup(x, 'lxml').find("div", {"class": "aboutfont"}).text
        elif 'dyqh' in keyword:
            func = lambda x: BeautifulSoup(x, 'lxml').find("div", {"class": "news-list news-info"}).find("div", {"class": "cont"}).text
        elif 'sdfutures' in keyword:
            func = lambda x: BeautifulSoup(x, 'lxml').find("div", {"class": "mainContent"}).text
        elif 'guosenqh' in keyword:
            func = lambda x: BeautifulSoup(x, 'lxml').find("div", {"class": "jp_yyb_box"}).text
        elif 'gtaxqh' in keyword:
            func = lambda x: BeautifulSoup(x, 'lxml').find("div", {"class": "right_service0"}).text
        elif 'gzf2010' in keyword :
            func = lambda x: BeautifulSoup(x, 'lxml').find("div", {"class": "r right right_all"}).text
        elif 'jxgsqh' in keyword:
            func = lambda x: BeautifulSoup(x, 'lxml').find("li", {"class": "ny_li2"}).text
        elif 'hongyuanqh' in keyword:
            func = lambda x: BeautifulSoup(x, 'lxml').find("div", {"class": "about_text1"}).text
        elif 'thanf' in keyword:
            func = lambda x: BeautifulSoup(x, 'lxml').find("p", {"class": "text-justify"}).text
        elif 'scqh' in keyword:
            func = lambda x: BeautifulSoup(x, 'lxml').find('div', {'class':'jjyw-article clearfix'}).text
        elif 'xinhu' in keyword:
            func = lambda x: BeautifulSoup(x, 'lxml').find('div', {'class':'content_show'}).text
        else:
            raise Exception('Website not in choose_func please add it accordingly')
        return func
    
    def function_date(self, url, html):
        try:
            keyword = url.split('.')[1]
    
            res = 'not found date'
    
            if 'bhfcc' in keyword:
    
                res = BeautifulSoup(html, 'lxml').find('div', {'class': 'research-deta-top'}).find('span').text.strip()
    
            elif 'bocifco' in keyword:
    
                res = [i for i in BeautifulSoup(html, 'lxml').find('div', {'class': 'mainCon fr'}).findAll('span') if '更新时间：' in str(i)]
                res = re.findall(r'\d{4}年\d{1,2}月\d{1,2}日', str(res[0]))[0]
                res = res.replace('年', '-').replace('月', '-').replace('日', '')
    
            elif 'btqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class': 't-con'}).findAll('div')[0]
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'cnhtqh' in keyword:
                res = [i for i in BeautifulSoup(html, 'lxml').find('div', {'class': 'htyj_nrtit'}).findAll('span') if '时间' in str(i)]
                res = re.findall(r'\d{4}年\d{1,2}月\d{1,2}日', str(res[0]))[0]
                res = res.replace('年', '-').replace('月', '-').replace('日', '')
    
            elif 'dlqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class': 'subtitle'}).findAll('span')[0]
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'fnqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class': 'bussiness-h1'}).findAll('p')[0]
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'ftol' in keyword:
                res = BeautifulSoup(html, 'lxml').find('span', {'class': 'time ml0'})
                res = re.findall(r'\d{4}年\d{1,2}月\d{1,2}日', str(res))[0]
                res = res.replace('年', '-').replace('月', '-').replace('日', '')
    
            elif 'gtaxqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class': 'text'})
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'guosenqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class': 'jp_yyb_box'}).find('h3')
                res = re.findall(r'\d{8}', str(res))[0]
                res = res[:4]+'-'+res[4:6] + '-' +res[6:]
    
            elif 'gzf2010' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class': 'pic1'}).find('div')
                res = re.findall(r'\d{4}\/\d{1,2}\/\d{1,2}', str(res))[0]
                res = res.split('/')
                if len(res[1]) == 1:
                    res[1] = '0'+res[1]
                res = '-'.join(res)
    
            elif 'gzjkqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('h4', {'class': 'info_time h4'})
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'hhqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class': 'newsxxtitle'}).find('span')
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'hlqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('h3', {'class': 'webtit2 f5 f6'}).find('span')
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'hongyuanqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class': 'line_dashed'})
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'jtqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('td', {'valign': 'middle'})
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'jxgsqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('li', {'class': 'ny_li'}).find('span')
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'sdfutures' in keyword:
    
                if BeautifulSoup(html, 'lxml').find('div', {'class': 'mainContent'}) != None:
                    res = BeautifulSoup(html, 'lxml').find('div', {'class': 'mainContent'})
                    res = re.findall(r'\d{8}', str(res))[0]
                    res = res[:4]+'-'+res[4:6] + '-' +res[6:]                 
    
                elif BeautifulSoup(html, 'lxml').find('em', {'id': 'publish_time'}) != None:
                    res = BeautifulSoup(html, 'lxml').find('em', {'id': 'publish_time'})
                    res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
                
            elif 'shcifco' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class': 'title'})
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'shqhgs' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class': 'con_line'})
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'szjhqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'id': 'artinfo'})
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'tqfutures' in keyword:
                res = BeautifulSoup(html, 'lxml').find('h4', {'class': 'info_time h4'})
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'xzfutures' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class': 'homehm'}).find('h3')
                res = re.findall(r'\d{8}', str(res))[0]
                res = res[:4]+'-'+res[4:6] + '-' +res[6:]
    
            elif 'zcqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('td', {'class': 'zc_hei_in'})
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'cnzsqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('p', {'class': 'center fbt'})
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'dyqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class':'news-list news-info'}).find('p', {'class':'ti'})
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
    
            elif 'htfc' in keyword:
                res = BeautifulSoup(html, 'lxml').find('h3', {'class': 'wztit'})
                res = re.findall(r'\d{8}', str(res))[0]
                res = res[:4]+'-'+res[4:6] + '-' +res[6:]
    
            elif 'mfc' in keyword:
                res = BeautifulSoup(html, 'lxml').find('h1', {'id': 'titles'})
                res = re.findall(r'\d{8}', str(res))[0]
                res = res[:4]+'-'+res[4:6] + '-' +res[6:]
            elif 'rdqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class': 'work_article'}).find('span')
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]

            elif 'thanf' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class':'service-content'}).find('h2')
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
                
            elif 'xinhu' in keyword:
                res = BeautifulSoup(html, 'lxml').find('p', {'class':'tc mgt10'}).find('span', {'class':'light_gray'})
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
            elif 'scqh' in keyword:
                res = BeautifulSoup(html, 'lxml').find('div', {'class':'jjyw-top clearfix'}).find('p')
                res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', str(res))[0]
        except:
            res = 'error_page'
            
        return res

#if __name__ == '__main__':
#    
#    engine = sq.create_engine("mysql+pymysql://root:cmlpdrwan0325@localhost/Alternative_DB?host=localhost?port=3306")
#    conn = engine.connect()
#    
#    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
#    error_path = './step2_data/html/error_log_{}.json'.format(current_time)
#    
#    extracter = html_extracter(conn)
#    df_crawl = pd.read_sql('Select * from html', con=conn)
#    
#    problem = extracter.extract(df_crawl)
#    
#    if problem != {}:
#        other_function.dump_json(problem, error_path)
#        print('error found, please check the error file:{}'.format(error_path))
#    else:
#        print('no error found')