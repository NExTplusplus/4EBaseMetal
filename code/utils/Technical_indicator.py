import pandas as pd
import numpy as np
from copy import copy

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
def divergence_pvt (close,pvt,train_end,params):
    percentage_change = (close/close.shift(1))-1
    pvt_change = (pvt/pvt.shift(1))-1
    divPT = percentage_change-pvt_change
    temp = sorted(copy(divPT[:train_end]))
    mx = temp[-1]
    mn = temp[0]
    if params['both'] == 1:
        mx = temp[int(np.floor((1-params['strength'])*len(temp)))]
    elif params['both'] == 2:
        mn = temp[int(np.ceil(params['strength']*len(temp)))]
    elif params['both'] == 3:
        mx = temp[int(np.floor((1-params['strength'])*len(temp)))]
        mn = temp[int(np.ceil(params['strength']*len(temp)))]
    for i in range(len(divPT)):
        if divPT[i] > mx:
            divPT[i] = mx
        elif divPT[i] < mn:
            divPT[i] = mn    
    return divPT

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
    for i in range(len(ad_change)):
        if np.isinf(ad_change[i]):
            ad_change[i] = np.nan
    return percentage_change-ad_change
