# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:18:32 2019

@author: Kwoks
"""

#the standard package 
import os
import sys
import time
import pickle
import argparse
import datetime
import pandas as pd
import sqlalchemy as sq
from random import randint
from configparser import ConfigParser
#load the utils function
sys.path.append('../other_function/')
import other_function


#package for crawler
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import *
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.support import expected_conditions as EC

class Crawler_machine:
    def __init__(self,conn,build_db_html = False):
        self.conn_html = conn
        if build_db_html:
            self.build_html_db()
    def build_html_db(self):
        # Function: Set up a database to store accuracy with the following setting. 
        # Note that: All functions in this class will follow this setting, pls set up ur database accordingly to avoid error
        self.conn_html.execute('CREATE TABLE `Alternative_DB`.`html`(`url` VARCHAR(750) NOT NULL,`id` INT NOT NULL AUTO_INCREMENT,`company` VARCHAR(30) NULL,`type` VARCHAR(45) NULL,`title` TINYTEXT NULL,`html` MEDIUMTEXT NULL,PRIMARY KEY (`url`),KEY(`id`));')
    
    def raw_crawl_func(self,driver,crawl_type,analyst_company, news_type,website_link,early_stop=True, keyword_report = 'All',
                             xpath = None,tag_element = None,keyword_next_page =None,keyword_filter = [],exclude_list=[]):
        
        # Function: This function will crawl data from the given page and it will stop crawling when we reach the url that is already in database
        # Inputs: Wait time is seconds to wait for driver to initiate, website_link is the index page link
        # keyword_report is to select all reports that contain keyword
        # If keyword_report is All, then xpath and tag_element for the all report links must be given in order to locate them
        # keyword_filter is the keyword must be contained in the title if not we will filter it out
        
        #this variable is used to tell us whether we need to put this link into the error link
        error_occurs = 0
         
        result = self.conn_html.execute("SHOW TABLES LIKE 'html';")
        
        if  not result.first():
            print('cannot find the table, we will create it')
            self.conn_html.execute('CREATE TABLE `Alternative_DB`.`html`(`url` VARCHAR(750) NOT NULL,`id` INT NOT NULL AUTO_INCREMENT,`company` VARCHAR(30) NULL,`type` VARCHAR(45) NULL,`title` TINYTEXT NULL,`html` MEDIUMTEXT NULL,PRIMARY KEY (`url`),KEY(`id`));')

#            driver.close()
#            raise Exception('Database not exist, please use build_html_db function')
        
        
        # Record we have proceed how many links we have crawled
        record_page = 0
        crawled_data =0 
        crawled_link = list(pd.read_sql('select * from html', con=conn)['url'])
        
        # Record the website firstpage
        current_page = 1
        current_website_link = website_link
        
        print('Start Crawling Report from {}'.format(analyst_company))
    
    
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[0])
        
        type_lst = ['manual_num_crawl', 'auto_click_crawl', 'auto_link_crawl','same_page_click']
        if crawl_type not in type_lst:
            driver.quit()
            raise Exception('crawl_type can only be manual_num_crawl, auto_click_crawl, auto_link_crawl,same_page_click')
        try:
            
            while True:
                SCROLL_PAUSE_TIME = 2.5

                # Get scroll height
                last_height = driver.execute_script("return document.body.scrollHeight")

                while True:
                # Scroll down to bottom
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                # Wait to load page
                    time.sleep(SCROLL_PAUSE_TIME)

                # Calculate new scroll height and compare with last scroll height
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height
                    
                if keyword_report!= 'All':
                
                # Check if we have particular report type we want to extract:
                    if type(keyword_report)!= str:
                        raise Exception('Error : keyword_report can only be one keyword and the type should be string')
                    else:
                        total_link = driver.find_elements_by_partial_link_text(keyword_report)
                else:
                    if xpath == None or tag_element == None:
                        raise Exception('Error : class_type cannot be none, have to key in class name if u choose for All link option, if not pls enter keyword_report')
                    else:
                        temp = driver.find_element_by_xpath(xpath)
                        total_link = temp.find_elements_by_tag_name(tag_element)
                        
                # Now select potentail links for report type we are interested in the current page
            
                if keyword_filter == 'All':
                    filter_link = list(set(total_link))
                else:
                    selected_link = []
                    for keyword in keyword_filter:
                        current_link = driver.find_elements_by_partial_link_text(keyword)
                        selected_link = list(set().union(selected_link,current_link))
                    filter_link = list(set(selected_link) & set(total_link))
                
                
                
                # Filter out link that contains keyword we don't want
                kickout_link = [] 
                for keyword in exclude_list:
                    current_link = driver.find_elements_by_partial_link_text(keyword)
                    kickout_link = list(set().union(kickout_link,current_link))
                        
                for kickout in kickout_link:
                    filter_link.remove(kickout)
            
                filter_link =  sorted(filter_link, key=total_link.index)
                
                repeat = 0
                for link in filter_link:
                    
                    # This is to check whether the link is crawled. Note that this is for same_page_click type
                    url = link.get_attribute("href")
                    if url in crawled_link:
                        repeat += 1
                        if early_stop:
                            break
                        else:
                            continue
                    # Crawl url,html content, title and update to database
                    else:
                        new_input = {}
                        new_input['title'] = [link.text[:50]]
                        new_input['company'] = [analyst_company]
                        new_input['type'] = [news_type]
                        new_input['url'] = [link.get_attribute("href")]
                        
                        temp = link.get_attribute("href")

                        driver.switch_to.window(driver.window_handles[1])
                        driver.get(temp)              
                        time.sleep(30)
                        
                        new_input['html'] = [str(BeautifulSoup(driver.page_source, "html.parser"))]
                        driver.switch_to.window(driver.window_handles[0])
                    
                    
                        df_input = pd.DataFrame(new_input)
                        df_input.to_sql(name='html', con=self.conn_html, if_exists='append',index=False)
                        crawled_data+=1
                
                
                print('Link crawled : ',crawled_data)
                record_page +=1
                #Check condition whehter to go to next page
            
                if repeat >= 1 and early_stop == True:
                    break
                else:
                    if keyword_next_page and crawl_type == 'auto_link_crawl':
                        
                        next_page_link = driver.find_elements_by_partial_link_text(keyword_next_page)
                    
                        # Check whether there is next page to go
                        if next_page_link:
                            page_link_check = next_page_link[0].get_attribute("href")
                            
                            if ('void' in page_link_check) or (page_link_check == None):
                                break
                            elif page_link_check == current_website_link:
                                break
                            else:
                                driver.get(page_link_check)
                                current_website_link = page_link_check
                        else:
                            break
                    elif keyword_next_page and crawl_type == 'manual_num_crawl':
                        current_page +=1
                        temp_next = driver.find_element_by_xpath(keyword_next_page)
                        next_page_link = temp_next.find_elements_by_partial_link_text(str(current_page))
                        # Check whether there is next page to go
                        if next_page_link:
                            page_link_check = next_page_link[0].click()
                        else:
                            break
                    elif keyword_next_page and (crawl_type == 'auto_click_crawl' or crawl_type == 'same_page_click'):
                        next_page_link = driver.find_elements_by_partial_link_text(keyword_next_page)
                        if crawl_type == 'same_page_click':
                            crawled_link += filter_link
                        # Check whether there is next page to go
                        if next_page_link:
                            current_website_link = driver.current_url
                            page_link_check = next_page_link[0].click()
                            time.sleep(10)
                        else:
                            break
                    else:
                        break
                    
        except Exception as e: 
            if 'Duplicate entry' in str(e):
                print('Meeting duplicate entry, completed')
                print('Link crawled : ',crawled_data)
            
            else:   
                print('Program Died at Page '+str(record_page))
                print('reason is : ')
                print(e)
                error_occurs = 1
            #driver.quit()
        
        driver.quit()
        print("Completed")
        print('Link crawled : ',crawled_data)
        return error_occurs
    
    def crawl(self,wait_time,website_link,webdriver_link, switch):
        
        
        pre_click = False
        keyword_filter = ['铜','铅','锌','镍','铝','锡','有色'] 
        exclude_list=[]
        try:
            if website_link == 'http://www.dlqh.com/page-1-4.php':
                analyst_company = '大陆'
                news_type = 'Weekly'
                keyword_report = '周报'
                keyword_next_page = None 
                xpath = None
                tag_element = None
                crawl_type = 'auto_link_crawl'
            
            elif website_link == 'http://www.cnhtqh.com.cn/list/664/1.shtml?id=jsqh':
                analyst_company = '恒泰'
                news_type = 'Weekly'
                keyword_report = 'All'
                keyword_next_page = '下一页' 
                xpath = "//div[@class='kf_jynews']"
                tag_element = 'a'
                crawl_type = 'auto_link_crawl'
            
            elif website_link == 'http://www.bocifco.com/Category_86/Index.aspx':
                analyst_company = '中银'
                news_type = 'Weekly'
                keyword_report = '周报'
                keyword_next_page = '下一页' 
                xpath = None
                tag_element = None
                crawl_type = 'auto_link_crawl'
            elif website_link == 'http://www.xzfutures.com/deeconomic.html':
                analyst_company = '兴证'
                news_type = 'Weekly'
                keyword_report = '精要'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
            elif website_link == 'http://www.xzfutures.com/deeconomic_cid_119.html':
                analyst_company = '兴证'
                news_type = 'Weekly'
                keyword_report = '周报'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
            
            elif website_link == 'https://www.mfc.com.cn/research.html':
                pre_click = True
                locate_action = "//div[@class='screen']"
                click_action=["//a[contains(text(), '常规')]","//a[contains(text(), '金属')]"]
                
                analyst_company = '美尔雅'
                news_type = 'Weekly'
                keyword_report = '周报'
                xpath = None
                tag_element = None
                keyword_next_page = '加载更多' 
                crawl_type = 'same_page_click'
            elif website_link == 'http://www.gzjkqh.com/czjy/list_36.aspx':
                analyst_company = '广金'
                news_type = 'Daily'           
                keyword_report = '种操作建议'
                xpath = None
                tag_element = None
                keyword_next_page = '>' 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
            
            elif website_link == 'http://www.shqhgs.com/lists/Dailystudy/p/1.html':
                analyst_company = '神华'
                news_type = 'Daily'
                keyword_report = '行情资信'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
            elif website_link == 'http://www.zcqh.com/zcyj_z.php?name=%E6%AF%8F%E6%97%A5%E6%97%A9%E6%8A%A5&page=1':
                analyst_company = '中财'
                news_type = 'Daily'
                keyword_report = '晨会焦点'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
                
            elif website_link == 'http://www.tqfutures.com/zpts/list_14.html':
                analyst_company = '中投'
                news_type = 'Morning'
                keyword_report = '早盘分析'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
                
            elif website_link == 'http://www.shcifco.com/html/yanjiuzhongxin/shishikuaixun/weipantishi/':
                analyst_company = '上海中期'
                news_type = 'After Market'
                keyword_report = '尾盘研发观点'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
                
            elif website_link == 'https://www.dyqh.info/research/researchreport/8/2?tag=1':
                analyst_company = '大越'
                news_type = 'Daily'
                keyword_report = '每日评论'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
            elif website_link == 'http://www.xzfutures.com/deeconomic_cid_107.html':
                analyst_company = '兴证'
                news_type = 'After Market'
                keyword_report = '研发盘末提示'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'    
            
            elif website_link == 'https://www.rdqh.com/content/index/115?page=1':
                analyst_company = '瑞达'
                news_type = 'Daily'
                keyword_report = '瑞达期货'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
                
            elif website_link == 'http://www.jtqh.cn/yfzx_1.asp?fl=1&lm=2&pz=%CD%AD':
                analyst_company = '锦泰'
                news_type = 'Morning'
                keyword_report = '铜'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_click_crawl'
                keyword_filter = 'All'
                
            elif website_link == 'http://www.sdfutures.com.cn/a/sdyanjiu/jinshu/ribao/list_151_1.html':
                analyst_company = '盛达'
                news_type = 'Daily'
                keyword_report = '盛达期货'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
            elif website_link == 'http://www.guosenqh.com.cn/main/yjzx/zp/zp_metal/index.shtml':
                analyst_company = '国信'
                news_type = 'Morning'
                keyword_report = '早评'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_click_crawl'
            elif website_link == 'http://www.gtaxqh.com/html/RESEARCH_SERVICES/pzbg/ysjgjs/':
                analyst_company = '国投'
                news_type = 'Morning'
                keyword_report = '晨报'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
            
            elif website_link == 'http://www.cnhtqh.com.cn/list/654/1.shtml?id=cjzbc':
                analyst_company = '恒泰'
                news_type = 'Morning'
                keyword_report = '早盘精要'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
            elif website_link == 'http://www.shcifco.com/html/yanjiuzhongxin/shishikuaixun/qishizaobanche/':
                analyst_company = '上海中期'
                news_type = 'Morning'
                keyword_report = '早盘研发'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
            
            elif website_link == 'https://www.dyqh.info/research/researchreport/6/0?tag=1':
                analyst_company = '大越'
                news_type = 'Daily'
                keyword_report = '交易内参'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
            
            elif website_link == 'http://www.xzfutures.com/deeconomic_page_1.html?cid=114':
                analyst_company = '兴证'
                news_type = 'Daily'
                keyword_report = '日报'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
            
            elif website_link == 'http://www.gzf2010.com.cn/ResearchCenter.aspx?c=220102&page=1':
                analyst_company = '广州期货'
                news_type = 'Daily'
                keyword_report = 'All'
                xpath = "//div[@class='r right right_all']//ul"
                tag_element = 'a'
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
            elif website_link == 'http://www.jxgsqh.com/News_list.aspx?Sort_Id=634&Mid=629':
                analyst_company = '国盛'
                news_type = 'Daily'
                keyword_report = 'All'
                xpath = "//div[@class='ny_ri_n']//ul"
                tag_element = 'a'
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
            elif website_link == 'http://www.hongyuanqh.com/hyqhnew/hyyj/more_index.jsp?1=1&oneMenuId=000200010015&twoMenuId=0002000100150001&threeMenuid=00020001001500010002&classId=000200010015000100020004&productid=100':
                analyst_company = '宏源'
                news_type = 'Morning'
                keyword_report = '有色早评'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_click_crawl'
            elif website_link == 'http://www.hhqh.com.cn/?pcyear=10-24-13':
                analyst_company = '和合'
                news_type = 'Morning'
                keyword_report = '和合期货'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
                keyword_filter = ['铜','铅','锌','镍','铝','锡','有色','早盘提示']
            
            elif website_link == 'http://www.bhfcc.com/research_page_1.html?pid=88':
                analyst_company = '渤海'
                news_type = 'Morning'
                keyword_report = '早盘提示'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_link_crawl'
            elif website_link == 'https://www.thanf.com/list-83.html':
                analyst_company = '天风'
                news_type = 'Daily'
                keyword_report = 'All'
                xpath = "//div[@class='service-index ']"
                tag_element = 'a'
                keyword_next_page = '»' 
                crawl_type = 'auto_click_crawl'
            
            elif website_link == 'http://www.bocifco.com/Category_28/Index.aspx':
                analyst_company = '中银'
                news_type = 'Daily'
                keyword_report = '日报'
                xpath = None
                tag_element = None
                keyword_next_page = '下一页' 
                crawl_type = 'auto_click_crawl'
            
            elif website_link == 'http://www.ftol.com.cn/main/yfzx/rcbg/rcbg/zaoping/index.shtml':
                analyst_company = 'Holly'
                news_type = 'Morning'
                keyword_report = '金属早评'
                xpath = None
                tag_element = None
                keyword_next_page = "//div[@class='page_footer']" 
                crawl_type = 'manual_num_crawl'
                keyword_filter = 'All'
                exclude_list = ['黑色金属早评']
            elif website_link == 'https://www.cnzsqh.com/cms/column/index?id=116':
                analyst_company = '浙商'
                news_type = 'Morning'
                keyword_report = '留单建议'
                xpath = None
                tag_element = None
                keyword_next_page = "»" 
                crawl_type = 'auto_click_crawl'
                keyword_filter = 'All'
            
            elif website_link == 'http://www.fnqh.com.cn/content/index/85':
                analyst_company = '福能'
                news_type = 'Morning'
                keyword_report = '早报'
                xpath = None
                tag_element = None
                keyword_next_page = "»" 
                crawl_type = 'auto_link_crawl'
            
            elif website_link == 'http://www.xzfutures.com/deeconomic_cid_106.html':
                analyst_company = '兴证'
                news_type = 'Morning'
                keyword_report = '中心早评'
                xpath = None
                tag_element = None
                keyword_next_page = "下一页" 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
            
            elif website_link == 'http://www.gzf2010.com.cn/ResearchCenter.aspx?c=220101':
                analyst_company = '广州期货'
                news_type = 'Morning'
                keyword_report = 'All'
                xpath = "//div[@class='r right right_all']//ul"
                tag_element = "a"
                keyword_next_page = "下一页" 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
                
            elif website_link == 'http://www.hlqh.com/article_cat.php?id=96&t_id=96':
                analyst_company = '华联'
                news_type = 'Morning'
                keyword_report = '研究所日评'
                xpath = None
                tag_element = None
                keyword_next_page = "下一页" 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
            
            elif website_link == 'https://www.btqh.com/index.php?m=content&c=index&a=lists&catid=56':
                analyst_company = '倍特'
                news_type = 'Daily'
                keyword_report = 'All'
                xpath = "//ul[@class='fxd-text']"
                tag_element = 'a'
                keyword_next_page = ">" 
                crawl_type = 'auto_link_crawl'
                
            elif website_link == 'http://www.szjhqh.com/index.php?s=news&c=category&id=10':
                analyst_company = '金汇'
                news_type = 'Daily'
                keyword_report = '有色板块'
                xpath = None
                tag_element = None
                keyword_next_page = "下一页" 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
            
            elif website_link == 'https://www.htfc.com/main/yjzx/yjbg/index.shtml':
                analyst_company = '华泰'
                news_type = 'Daily'
                keyword_report = '日报'
                xpath = None
                tag_element = None
                keyword_next_page = "下一页" 
                crawl_type = 'auto_click_crawl'
                pre_click = True
                locate_action = "//div[@class='market_box']"
                click_action=["//a[contains(text(), '日报')]","//a[contains(text(), '金属')]"]
            
            elif website_link == 'http://www.guosenqh.com.cn/main/yjzx/rp/rp_metal/index.shtml':
                analyst_company = '国信'
                news_type = 'Daily'
                keyword_report = '日评'
                xpath = None
                tag_element = None
                keyword_next_page = "下一页" 
                crawl_type = 'auto_click_crawl'
    
            elif website_link == 'http://www.shcifco.com/html/yanjiuzhongxin/shishikuaixun/panzhongshidian/':
                analyst_company = '上海中期'
                news_type = 'After Market'
                keyword_report = '午间评述'
                xpath = None
                tag_element = None
                keyword_next_page = "下一页" 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
            
            elif website_link == 'http://www.gtaxqh.com/html/RESEARCH_SERVICES/zhlcb/mrsp/':
                analyst_company = '国投安信'
                news_type = 'After Market'
                keyword_report = '货每日收评'
                xpath = None
                tag_element = None
                keyword_next_page = "下一页" 
                crawl_type = 'auto_link_crawl'
                keyword_filter = 'All'
            else:
                print('Website not in the class, please add under crawl function')
                return
        except Exception as e:
            print('sth may change in the {}'.format(website_link))
            return
        
        # Load the Chrome Driver
        chrome_options = Options()
        chrome_options.add_argument('headless')
        driver = webdriver.Chrome(webdriver_link, chrome_options=chrome_options)
        print('init driver')

        # Open the website firstpage
        driver.get(website_link)
        print('Load Page')
        time.sleep(wait_time)
        
        #Pre Click
        if pre_click:
            temp= driver.find_element_by_xpath(locate_action)
            for action in click_action: 
                temp.find_element_by_xpath(action).click()
                time.sleep(5)
                
        res = self.raw_crawl_func(driver,crawl_type,analyst_company, news_type,website_link,switch,
                                  keyword_report,xpath = xpath,tag_element = tag_element,
                                  keyword_next_page =keyword_next_page,keyword_filter=keyword_filter,exclude_list=exclude_list)
        
        return res
    
website_link_lst =[]
website_link_lst.append('http://www.dlqh.com/page-1-4.php')
website_link_lst.append('http://www.cnhtqh.com.cn/list/664/1.shtml?id=jsqh')
website_link_lst.append('http://www.bocifco.com/Category_86/Index.aspx')
website_link_lst.append('http://www.xzfutures.com/deeconomic.html')
website_link_lst.append('http://www.xzfutures.com/deeconomic_cid_119.html')
website_link_lst.append('https://www.mfc.com.cn/research.html')
website_link_lst.append('http://www.gzjkqh.com/czjy/list_36.aspx')
website_link_lst.append('http://www.shqhgs.com/lists/Dailystudy/p/1.html')
website_link_lst.append('http://www.zcqh.com/zcyj_z.php?name=%E6%AF%8F%E6%97%A5%E6%97%A9%E6%8A%A5&page=1')
website_link_lst.append('http://www.tqfutures.com/zpts/list_14.html')
website_link_lst.append('http://www.shcifco.com/html/yanjiuzhongxin/shishikuaixun/weipantishi/')
website_link_lst.append('https://www.dyqh.info/research/researchreport/8/2?tag=1')
website_link_lst.append('http://www.xzfutures.com/deeconomic_cid_107.html')
website_link_lst.append('https://www.rdqh.com/content/index/115?page=1')
website_link_lst.append('http://www.jtqh.cn/yfzx_1.asp?fl=1&lm=2&pz=%CD%AD')
website_link_lst.append('http://www.sdfutures.com.cn/a/sdyanjiu/jinshu/ribao/list_151_1.html')
website_link_lst.append('http://www.guosenqh.com.cn/main/yjzx/zp/zp_metal/index.shtml')
website_link_lst.append('http://www.gtaxqh.com/html/RESEARCH_SERVICES/pzbg/ysjgjs/')
website_link_lst.append('http://www.cnhtqh.com.cn/list/654/1.shtml?id=cjzbc')
website_link_lst.append('http://www.shcifco.com/html/yanjiuzhongxin/shishikuaixun/qishizaobanche/')
website_link_lst.append('https://www.dyqh.info/research/researchreport/6/0?tag=1')
website_link_lst.append('http://www.xzfutures.com/deeconomic_page_1.html?cid=114')
website_link_lst.append('http://www.gzf2010.com.cn/ResearchCenter.aspx?c=220102&page=1')
website_link_lst.append('http://www.jxgsqh.com/News_list.aspx?Sort_Id=634&Mid=629')
website_link_lst.append('http://www.hongyuanqh.com/hyqhnew/hyyj/more_index.jsp?1=1&oneMenuId=000200010015&twoMenuId=0002000100150001&threeMenuid=00020001001500010002&classId=000200010015000100020004&productid=100')
website_link_lst.append('http://www.hhqh.com.cn/?pcyear=10-24-13')
website_link_lst.append('http://www.bhfcc.com/research_page_1.html?pid=88')
website_link_lst.append('https://www.thanf.com/list-83.html')
website_link_lst.append('http://www.bocifco.com/Category_28/Index.aspx')
website_link_lst.append('http://www.ftol.com.cn/main/yfzx/rcbg/rcbg/zaoping/index.shtml')
website_link_lst.append('https://www.cnzsqh.com/cms/column/index?id=116')
website_link_lst.append('http://www.fnqh.com.cn/content/index/85')
website_link_lst.append('http://www.xzfutures.com/deeconomic_cid_106.html')
website_link_lst.append('http://www.gzf2010.com.cn/ResearchCenter.aspx?c=220101')
website_link_lst.append('http://www.hlqh.com/article_cat.php?id=96&t_id=96')
website_link_lst.append('https://www.btqh.com/index.php?m=content&c=index&a=lists&catid=56')
website_link_lst.append('http://www.szjhqh.com/index.php?s=news&c=category&id=10')
website_link_lst.append('https://www.htfc.com/main/yjzx/yjbg/index.shtml')
website_link_lst.append('http://www.guosenqh.com.cn/main/yjzx/rp/rp_metal/index.shtml')
website_link_lst.append('http://www.gtaxqh.com/html/RESEARCH_SERVICES/zhlcb/mrsp/')
website_link_lst.append('http://www.shcifco.com/html/yanjiuzhongxin/shishikuaixun/panzhongshidian/')

if  __name__ == '__main__':
    
    switch = sys.argv[1]
    
    wait_time = 2.5
    #this file path is the windows version, in ubuntu, need to download the ubuntu version relevant to the version of the chrome 
    webdriver_link = './step1_data/chromedriver'
    error_link_path = './step1_data/error_link.json'
    
    config_path = './step1_data/config.ini'
    conf = ConfigParser()
    conf.read(config_path)
    
    use_database = conf.get('database_param', 'database')
    use_account = conf.get('database_param', 'account')
    use_host = conf.get('database_param', 'host')
    use_psw = conf.get('database_param', 'password')
    use_port = conf.get('database_param', 'port')
    
    engine = sq.create_engine("mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8".format(use_account, use_psw, use_host, use_port, use_database))
    conn = engine.connect()
    crawler = Crawler_machine(conn)

    if switch == 'check':
        
        switch = False
        
        error_link = other_function.load_json(error_link_path)
        
        if error_link == []:
            
            print('the error file is empty, it is unnecessary to run again')
            
        else:
            print('begin to run the error link again')
            
            error = []
            
            for website in error_link:
                tmp = crawler.crawl(wait_time, website, webdriver_link, switch)
                if tmp == 1:
                    error.append(website)
                    
            other_function.dump_json(error, error_link_path)
            print('crawling the error link is finished, please check the error file to see if it is empty, if not then run again')
    elif switch == 'run':
        
        switch = True
        
        original_error = other_function.load_json(error_link_path)
        
        if original_error  !=  []:
            print('error file is not empty, remember to check please')
        
        error = []
        for website in website_link_lst:
            tmp = crawler.crawl(wait_time, website, webdriver_link, True)
            if tmp == 1:
                error.append(website)
        
        other_function.dump_json(error+original_error, error_link_path)
        
        print('crawling the link is finished, please check the error file to see if it is empty, if not then run again')
        
        
    
    
    
    
    
    
    
