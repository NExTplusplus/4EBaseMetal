import rpy2.robjects as robjects
import json
import pandas as pd
import numpy as np
import datetime

robjects.r(".sourceQlib(chgwd = FALSE)")

#get todays date
today = datetime.datetime.now()
today = '-'.join([str(n) for n in [today.year,today.month,today.day]])

#read configuration file for reading data 
with open("exp/financial_data_crawl.conf","r") as f:
    config = json.load(f)

#iterate across key,value pair in config where key is R function name and value is list of list with 2 elements, ticker and column
with open("log/crawl/"+today+"_crawl_log.txt","w") as out:
    for key,val in config.items():
        curr_key = key.split('|')

        for l in val:

            ticker = l[0]            #ticker that identifies required asset in 4E database
            filepath = l[1]          #temporary path 
            lag = "Lag" in filepath  #boolean value of lag requirement in column
            temp_key = curr_key[0] if not lag else "lag("+curr_key[0]  #R function name with lag if required
            lag_str= (")" if not lag else "),1)") + ("" if len(curr_key) == 1 else curr_key[1])
            option = ", asPrice = TRUE" if "getTickers" in curr_key[0] else ""
            rcode = "write.csv(as.data.frame({}('{}'{},zoom = '2003::'{}),'{}')".format(temp_key,ticker,option,lag_str,filepath)
            robjects.r(rcode)
            crawled = pd.read_csv(filepath, index_col = 0)
            original = pd.read_csv(filepath.split('temp_')[1],index_col = 0)
            crawled.columns = original.columns

            #PA has Close column which has to be independently generated
            if ticker == "PA":
                rcode1 = "write.csv(as.data.frame(lag(getGen('PA1S',zoom = '2003::'),1)), 'temp_data/Financial Data/COMEX/Generic/Lagged/PA1S.csv')"
                print(rcode1)
                robjects.r(rcode1)
                to_overwrite = pd.read_csv('temp_data/Financial Data/COMEX/Generic/Lagged/PA1S.csv', index_col = 0)
                to_overwrite.columns = ["Close"]
                crawled["Close"] = to_overwrite["Close"]

            #To cater to different precisions
            if ticker == "CNYUSD Curncy":
                crawled = crawled.applymap(lambda x : round(float(x),5) if x != "NA" else x)
            elif ticker == "SHSZ300 Index":
                crawled = crawled.applymap(lambda x : round(float(x),4) if x != "NA" else x)
            else:
                crawled = crawled.applymap(lambda x : round(float(x),3) if x != "NA" else x)

            #remove irrelevant data points from crawled
            crawled = crawled[max(crawled.index.values[0],original.index.values[0]):]

            crawled_dates = set(crawled.index.values)

            out.write(ticker+"\n")
            out.write(filepath+"\n")
            out.write("Change in the last 7 days\n")
            #check last 7 days
            dates_to_compare = original.index.values[-7:]
            comparison = crawled.loc[dates_to_compare,:] == original.loc[dates_to_compare,:]
            comparison = comparison.apply(lambda x: x.all(), axis = 1)
            for date in comparison.index.values:
                out.write(date+"\n") if not comparison[date] else None
            #replace last 7 days of data 
            original = original[:-7]
            
            #check for dates to be added
            dates_to_be_added = sorted(crawled_dates - set(original.index.values))
            larger_than_original = [ d > max(original.index.values) for d in dates_to_be_added]
            dates_to_be_added = [d for (d, b) in zip(dates_to_be_added,larger_than_original) if b]
            df_to_be_added = crawled.loc[sorted(dates_to_be_added)]
            original = pd.concat([original,df_to_be_added],axis = 0)
            
            out.write("Added days\n")
            for date in df_to_be_added.index.values:
                out.write(date+"\n")
                

            original.to_csv(filepath.split('temp_')[1])
        