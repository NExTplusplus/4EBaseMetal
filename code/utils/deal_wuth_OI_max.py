#-*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
file_list = ["LMCADS_data_time_split.csv","LMNIDS_data_time_split.csv","LMPBDS_data_time_split.csv","LMPBDS_data_time_split.csv",
            "LMPBDS_data_time_split.csv"]
for file in file_list:
    file_dataframe = pd.read_csv(file)
    column = list(file_dataframe.columns)
    column.remove('Unnamed: 0')
    column.remove('Index')
    column.remove('year')
    column.remove('month')
    column.remove('day')
    column_name = column[0]
    print(column_name)
    year_list=list(set(file_dataframe['year'].values))
    month_list = list(set(file_dataframe['month'].values))
    for year in year_list:
        for month in month_list:
            day_list=[]
            day_dict={}
            index_list=file_dataframe[(file_dataframe['year']==year)&(file_dataframe['month']==month)].index
            if not index_list.empty:
                for index in index_list:
                    day_list.append(file_dataframe[(file_dataframe['year']==year)&(file_dataframe['month']==month)][column_name][index])
                    day_dict[file_dataframe[(file_dataframe['year']==year)&(file_dataframe['month']==month)][column_name][index]]=index
                print(day_list)
                max_item = max(day_list)
                min_item = min(day_list)
                file_dataframe[(file_dataframe['year']==year)&(file_dataframe['month']==month)][column_name][day_dict[max_item]]=np.mean(day_list)
    new_file_name = column_name+"_max.csv"
    file_dataframe.to_csv(new_file_name)