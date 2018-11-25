import pandas as pd
import numpy as np

# This function will calculate Price Volume Trend as mentioned in google drive/ technical indicator for more explanations
# close is the column for closing price and volume is the column for volume.
# It is encourged to use different version of volumes to try the result

def pvt (close,volume):
    pvt = close.shift(1)
    pvt.iloc[1] = ((close.iloc[1]/close.iloc[0])-1)*volume.iloc[1]
    for i in range(2,len(close)):
        pvt.iloc[i] = ((close.iloc[i]/close.iloc[i-1])-1)*volume.iloc[i] + pvt.iloc[i-1]     
    return pvt

# This function will calculate divergence between Price Volume Trend and percentage price change 
# as mentioned in google drive/ technical indicator for more explanations
# close is the column for closing price and pvt is the column for pvt.
# It is encourged to use different version of pvts to try the result
def divergence_pvt (close,pvt):
    percentage_change = (close/close.shift(1))-1
    pvt_change = (pvt/pvt.shift(1))-1
    return percentage_change-pvt_change

# This function will calculate accumulation/distribution as mentioned in google drive/ technical indicator for more explanations
# "X" is the dataframe we want to process
# close,low,opened,high are the column for closing price, low price, open price and high price respectively
# volume is the column for volume. 
# It is encourged to use different version of volumes to try the result
def ad (close,low,opened,high,volume):
    close_p = close
    low_p =  low
    open_p =  opened
    high_p =  high
    money_flow = ((close_p - low_p)-(high_p-close_p))/(high_p-low_p)      
    return money_flow*volume

# This function will calculate divergence between accumulation/distribution and percentage price change 
# as mentioned in google drive/ technical indicator for more explanations
# close is the column for closing price and ad is the column for ad.
# It is encourged to use different version of ads to try the result
def divergence_ad (close,ad):
    percentage_change = (close/close.shift(1))-1
    ad_change = (ad/ad.shift(1))-1
    return percentage_change-ad_change
