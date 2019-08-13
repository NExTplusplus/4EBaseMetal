# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:47:35 2019

@author: Kwoks
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from aip import AipNlp
from collections import Counter

#use the current recommend score.
recommend_path = '../big/recommend.csv'
recommend_total = pd.read_csv(recommend_path)

#metal list
metal_list = ['Cu', 'Zn', 'Pb', 'Al', 'Ni', 'Xi']

def get_metal_df(met, recommend):
    
    tmp = recommend[['url', 'company', 'news type', 
                    'date', 'title', met + '_fact', met + '_action']]
    
    df = tmp.copy()
    
    df.dropna(inplace = True)
    
    df[met + '_action'] = df[met + '_action'].apply(lambda x : x.split('\n')[1:])
    
    df.sort_values(by = ['date'], inplace = True)
    
    df.reset_index(inplace =  True, drop = True)
    
    return df


APP_ID = '15146529'
API_KEY = 'Dlgz79CkU4ZrbvM0dtpFnNwL'
SECRET_KEY = 'ND85LorrONMANcNeLavqF149Lcic0SB0'
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

# inputs is list of recommnedations, clean_words is the list of character we need to clean 
def calculate_sentiment(inputs,clean_words = None):
    index = 0
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
        error.append(e)
    return result, error

# inputs is list of dict which contains the sentiment of the corresponding sentence
def sen_art(inputs):
    result = 0
    for i in inputs:
        result+=(i['positive_prob']-i['negative_prob'])*i['confidence']
    if len(inputs)==0:
        return np.nan
    else:
        return result/len(inputs)

for met in metal_list:
    print('*******preprocessing metal: {}*******'.format(met))
    
    current_df = get_metal_df(met, recommend_total)
    
    metal_sentiment = []
    
    raise_error = []
    try:
        with tqdm(current_df[met + '_action'], desc= 'processing {} with aip'.format(met)) as t:
            for i in t:
        
                result, error = calculate_sentiment(i)
        
                metal_sentiment.append(result)
        
                raise_error.append(error)    
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()
        
    current_df['Sentiment'] = metal_sentiment
    
    current_df['Sentiment_article'] = current_df['Sentiment'].apply(sen_art)
    
    current_df.to_csv('./metal_score/{}_sentiment.csv'.format(met), index = False)

    error_counter = Counter([str(i) for i in raise_error if i != []])

    print('*******raise error num: {}, type:{}*******'.format(len(error_counter), list(error_counter.keys())))
    
    print('*******metal completed: {}*******'.format(met))
    
    
    












