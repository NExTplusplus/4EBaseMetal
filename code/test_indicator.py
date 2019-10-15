# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy import spline
import pywt

def wavelet_trans(data):
    db4 = pywt.Wavelet('db4')
    coef = pywt.wavedec(data,db4)
    coef[len(coef)-1] *= 0
    coef[len(coef)-2] *= 0
    meta = pywt.waverec(coef,db4)
    return meta

if os.getcwd != 'D://Next++//4E\Wanying//4EBaseMetal//data//Financial Data//LME':
    os.chdir('D://Next++//4E\Wanying//4EBaseMetal//data//Financial Data//LME')
file_list = glob.glob("LM*DY.csv") 
start_date = "2004-01-01"
for file in file_list:
    data = pd.read_csv(file)
    data.columns = ['Date','Close']
    data.Date = pd.to_datetime(data.Date,format = '%Y-%m-%d')
#    x = np.arrary(range(len(data.Date)))
#    x_smooth = np.linspace(x.min(),x.max(),)
    data['Return'] = np.log(data.Close/data.Close.shift(5))
    data['EMA'] = data.Return.ewm(span = 50).mean()
    
    data['Accelerator'] = (data.Return-data.Return.shift(5))/data.Return.shift(5)
#    data.Accelerator = data.Accelerator.rolling(20).mean()
    data['Momentum'] = data.Close-data.Close.shift(100)
    data = data.loc[data.Date>start_date]
    data.EMA = wavelet_trans(data.EMA)
    ticklabel = data.Date
    #xticks = np.arange(0,len(ticklabel),240)
    plt.figure(figsize = (18,30))
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(data.Date, data.Close,label = 'Close',color=color)
    ax1.set_ylabel('Close', color=color)
    ax1.tick_params(axis='y',color=color)
    #ax1.set_xticks(xticks)
    #ax1.set_xticklabels(ticklabel[xticks],rotation = 45)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.plot(data.Date, data.EMA,label = 'EMA',color=color)
    ax2.set_ylabel('EMA', color=color)
    ax2.tick_params(axis='y',color=color)
    #ax2.set_xticks(xticks)
    #ax2.set_xticklabels(ticklabel[xticks],rotation = 45)
#    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(file[:-4]+'.png')
#copper_price.columns = ['Date','Open','High','Low','Close','Volume']
#copper_price['RETURN'] = np.log(copper_price.Close/copper_price.Close.shift(1))
#copper_price['RETURN'] = (copper_price.Close - copper_price.Close.shift(1))/copper_price.Close.shift(1)
#copper_price = copper_price[~copper_price.isin([np.nan,np.inf,np.inf]).any(1)]
#copper_price['RSI'] = talib.RSI(copper_price.Close,timeperiod = 14)
        
#copper_price['SAR'] = talib.SAR(copper_price.High,copper_price.Low,acceleration = 0.02, maximum = 0.2)
#copper_price['diff'] = copper_price.Close - copper_price.SAR

#start_date = '2007-01-01'
#end_date = '2014-07-01
#copper_price = copper_price[copper_price.Date>start_date]
#copper_price = copper_price[copper_price.Date<end_date]

                
    














