# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:31:05 2019

@author: Kwoks
"""
import re
import pandas as pd
from tqdm import tqdm


class recommend_extracter:
    def __init__(self,conn,build_db_recommend = False):
        
        self.conn_extracter = conn
        if build_db_recommend:
            self.build_recommend_db()
    
    def build_recommend_db(self):
        # Function: Set up a database to store accuracy with the following setting. 
        # Note that: All functions in this class will follow this setting, pls set up ur database accordingly to avoid error
        self.conn_extracter.execute("CREATE TABLE `recommend`(`url` varchar(700) NOT NULL,`id` int(11) NOT NULL AUTO_INCREMENT,`company` varchar(20) DEFAULT NULL,`news_type` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,`date` datetime DEFAULT NULL,`title` varchar(100) DEFAULT NULL,`Cu_fact` mediumtext COMMENT '\n',`Cu_action` mediumtext CHARACTER SET utf8 COLLATE utf8_general_ci,`Zn_fact` mediumtext,`Zn_action` mediumtext,`Pb_fact` mediumtext,`Pb_action` mediumtext,`Al_fact` mediumtext,`Al_action` mediumtext,`Ni_action` mediumtext,`Ni_fact` mediumtext,`Xi_action` mediumtext,`Xi_fact` mediumtext,`Other` mediumtext CHARACTER SET utf8 COLLATE utf8_general_ci,PRIMARY KEY (`url`),KEY(`id`));")
    
    def cleaning(self,text,stop_words=[]):
        # Function: This function will split article into paragraph and replace all stop_words into empty string
        # Input: Text is the content, stop_words is list of use characters we want to replace into empty string
        
        # Split article into paragraphs by common split characther 
        command_split = '\n' + "|" +'；'
        paragraph_lst = re.split(command_split,text)
        result =[]
        
        # For each paragraphs, replace the stop_words
        for paragraph in paragraph_lst:
            for word in stop_words:
                paragraph = paragraph.replace(word,"")
            if len(paragraph)>0:
                result.append(paragraph)
        return result
    
    def classify(self,paragraph_lst):
        # Function: This function will classify each paragraph
        # Inputs: paragraph_lst is a list of paragraphs
        
        # Create a dictionary to store respective sentence
        metal = ['铜','镍','铝','锌','铅','锡']
        category ={}
        category['other']=[]
        for metal_type in metal:
            category[metal_type]=[]
        
        
        for paragraph in paragraph_lst:
            # If the paragraph mention specific metals, we record under the metal dictinoary
            for metal_type in metal:
                check = 0
                if metal_type in paragraph:
                    category[metal_type].append(paragraph)
                    check = 1
                    
            # If it does not record under any of the metals, we will record it in other
            if check == 0:
                category['other'].append(paragraph)
    
        # Combine paragraphs in other into one long paragraph
        new_other = ""
        for paragraph in category['other']:
            new_other = new_other+'\n'+paragraph
    
        category['other'] = new_other
        return category
    
    def extract_recommendation(self,paragraph_lst,split_word,key_words=[]):
        
        # Function: This function will extract recommendation and fact from given paragraph based on certain key_words
        # Inputs: paragraph_lst is list of paragraphs, split_word is list of chracters that can split paragraph into sentences,
        # keyword is list of words to identify recommendation
        
        fact = ""
        action = ""
        
        for index in range(0,len(paragraph_lst)):
            # Check each paragraph
            para = paragraph_lst[index]
            modification = []
            for key in key_words: 
                
                # Check whether the paragraph contain any keyword
                if key in para:
                    
                    # Split the para into sentences 
                    
                    command_split = split_word[0]
                    for split in split_word[1:]:
                        command_split = command_split + "|" +split
                    sentences = re.split(command_split,para)
                    for sentence in sentences:
                        # Check whether we got the action sentence already
                        if key in sentence and sentence not in modification:
                            action = action + '\n' + sentence
                            modification.append(sentence)
                            
            # Delete the action sentence from the paragraph
            for i in modification:
                para = para.replace(i,"")
            fact = fact + '\n'+para
        return (fact,action)
    
    def second_clean(self,fact_action_tuple, first_filter, second_filter,final_check):
        
        # Function: This fuction will further clean the action list generated from extract_recommendation
        # Inputs: fact_action_tuple is result gotten from extract_recommendation
        #  firs_filter, second_filter and final_check are list of keywords that action sentence must contain
        
        fact = fact_action_tuple[0]
        selected = fact_action_tuple[1].split('\n')
        action_lst = []
        result = ""
        
        # Check each potential action whehter contain keyword in filter
        for potential in selected:
            
            # status is to record whether it become action 
            status = 0
            
            # Previous is to record the number of inital recommendation sentences
            previous = len(action_lst)
            
            # Check whether each sentence contain first level keyword
            for key in first_filter:
                if key in potential:
                    action_lst.append(potential)
                    status = 1
                    break
                    
            # If nothing in first_filter can filter out, use second level
            if len(action_lst)== previous:
                for key in second_filter:
                    if key in potential:
                        action_lst.append(potential)
                        status = 1
                        break
                        
            if status==0:
                fact = fact + '\n'+ potential   
                
        for choosen in action_lst:
            status = 0
            for check in final_check:
                if check in choosen:
                    result = result+'\n'+choosen
                    status = 1
                    break
            if status==0:
                fact = fact + '\n'+ choosen
        return (fact,result)
    
    def extract(self,df_content,keyword,first,secondary,split,stop_words=[],update = True):
        # Function: This function will extract recommendation and fact from dataframe of 
        #           content (gotten from live html extracter). Moreover, it will classify
        #           the list of recommendation and facts according to metal type
        
        recommend_check = self.conn_extracter.execute("SHOW TABLES LIKE 'recommend';")
        
        if  not recommend_check.first():
#            raise Exception('Database not exist, please use build_content_db function')
            print('can not find the recommend table, will create it automatically')
            #self.conn_extracter.execute("CREATE TABLE `Alternative_DB`.`recommend`(`url` varchar(700) NOT NULL,`id` int(11) NOT NULL AUTO_INCREMENT,`company` varchar(20) DEFAULT NULL,`news type` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,`date` datetime DEFAULT NULL,`title` varchar(100) DEFAULT NULL,`Cu_fact` mediumtext COMMENT '\n',`Cu_action` mediumtext CHARACTER SET utf8 COLLATE utf8_general_ci,`Zn_fact` mediumtext,`Zn_action` mediumtext,`Pb_fact` mediumtext,`Pb_action` mediumtext,`Al_fact` mediumtext,`Al_action` mediumtext,`Ni_action` mediumtext,`Ni_fact` mediumtext,`Xi_action` mediumtext,`Xi_fact` mediumtext,`Other` mediumtext CHARACTER SET utf8 COLLATE utf8_general_ci,PRIMARY KEY (`url`),KEY(`id`));")
            self.build_recommend_db()
        
        
        metal_lst = ['铜','镍','铝','锌','铅','锡']
        fact_lst = ['Cu_fact','Ni_fact','Al_fact','Zn_fact','Pb_fact','Xi_fact']
        action_lst = ['Cu_action','Ni_action','Al_action','Zn_action','Pb_action','Xi_action']

        problem = {}
        already_url = list(pd.read_sql('select * from recommend', con=self.conn_extracter)['url'])

        df_content = df_content[~df_content['url'].isin(already_url)].reset_index(drop=True)

        for idx, url,com, news_type, date, title, content in tqdm(zip(df_content['id'], df_content['url'],df_content['company'],df_content['type'],
                                                                 df_content['date'],df_content['title'],df_content['content']), desc='extracting recommend'):
            
            try:
                result = {}
                # Create dictionary input
                for key in ['url','company', 'news_type', 'date', 'title', 'Cu_fact', 'Cu_action','Zn_fact', 
                            'Zn_action', 'Pb_fact', 'Pb_action', 'Al_fact', 'Al_action',
                            'Ni_action', 'Ni_fact', 'Xi_action', 'Xi_fact', 'Other']:
                    result[key] = []
                    
                    
                    
                result['url'].append(url)
                result['company'].append(com)
                result['news_type'].append(news_type)
                result['date'].append(date)
                result['title'].append(title)
                categories = self.classify(self.cleaning(content,stop_words=stop_words))
                    
                # extract recommendation and clean it for each metal
                for metal,fact,action in zip(metal_lst,fact_lst,action_lst):
                    # First level keyword
                    new_first = first +[metal+'价']
                        
                    # Extract recommendation
                    processing = self.extract_recommendation(categories[metal],split,keyword)  
                        
                    # Clean recommendation
                    cleaned_rec = self.second_clean(processing,new_first,secondary,keyword)
                        
                    # Record result
                    result[fact].append(cleaned_rec[0])
                    result[action].append(cleaned_rec[1])
                        
                result['Other'].append(categories['other'])
                df_input = pd.DataFrame(result)
                df_input.to_sql(name='recommend', con=self.conn_extracter, if_exists='append',index=False)
            except Exception as e:
                # If still have problem, record it
                print('Problem found',e)
                if str(e) in problem.keys():
                    problem[str(e)].append(idx)
                else:
                    problem[str(e)] = [idx]
        print('completed')
#        df_result = pd.DataFrame(result)
#        if update:
#            # Check whether table in database has been created 
#            check = self.conn_extracter.execute("SHOW TABLES LIKE 'recommend';")
#            if  not check.first():
#                print('Database not exist, please use build_recommend_db function')
#            else:
#                df_result.to_sql(name='recommend', con=self.conn_extracter, if_exists='append',index=False)
#            
        return problem
    
    
#if __name__ == '__main__':
#    
#    engine = sq.create_engine("mysql+pymysql://root:cmlpdrwan0325@localhost/Alternative_DB?host=localhost?port=3306")
#    conn = engine.connect()
#    
#    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
#    error_path = './step2_data/recommend/error_log_{}.json'.format(current_time)
#    
#    recommend = recommend_extracter(conn)
#    df_content = pd.read_sql('Select * from content',conn)
#    
#    keyword = ['震荡','偏强','观望','做多','轻仓','反弹','偏弱','上涨','企稳','承压','卖出','短线','短多','整理','止损',
#               '多仓','突破','支持','上行','空间','回补','低位','悲观',
#               '回落','弱势','抛售','回调','有望','走高','多单','上移','多头','走强','盘整','波动','上升','支撑','空单']
#    first = ['认为','预计','预测','预期','建议','观点','关注','强调','交易','铜价','多头','空头']
#    secondary = ['操作','短期','短线']
#    split = ['。','；']
#    stop_words = ['\t']
#    
#    problem = recommend.extract(df_content,keyword,first,secondary,split,stop_words=stop_words)
#    
#    if problem != {}:
#
#        other_function.dump_json(problem, error_path)
#        print('error found, please check the error file:{}'.format(error_path))
#    else:
#        print('no error found')
    
    
    