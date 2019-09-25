# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
import talib
import numpy as np
import matplotlib.pyplot as plt
import copy

def rsi(Close,period = 14,upper = 70,lower = 30):
    rsi = ta.RSI(Close,timeperiod = period)
    rsi_signal = pd.Series(index = Close.index,data = [0]*len(Close))
    for i in range(1,len(rsi_signal)):
        if rsi.iloc[i-1]>upper:
            rsi_signal.iloc[i] = -1
        elif rsi.iloc[i-1]<lower:
            rsi_signal.iloc[i] = 1

    return rsi_signal

if os.getcwd != 'D://Next++//4E\Wanying//4EBaseMetal//data//Financial Data//LME':
    os.chdir('D://Next++//4E\Wanying//4EBaseMetal//data//Financial Data//LME')

copper_price = pd.read_csv('LMECopper3M.csv')
copper_price.columns = ['Date','Open','High','Low','Close','Volume']
copper_price.Date = pd.to_datetime(copper_price.Date,format = '%Y-%m-%d')
copper_price['RETURN'] = np.log(copper_price.Close/copper_price.Close.shift(1))
#copper_price['RETURN'] = (copper_price.Close - copper_price.Close.shift(1))/copper_price.Close.shift(1)
copper_price = copper_price[~copper_price.isin([np.nan,np.inf,np.inf]).any(1)]
copper_price['RSI'] = talib.RSI(copper_price.Close,timeperiod = 14)
        
copper_price['SAR'] = talib.SAR(copper_price.High,copper_price.Low,acceleration = 0.02, maximum = 0.2)
copper_price['diff'] = copper_price.Close - copper_price.SAR

start_date = '2015-01-01'
end_date = '2014-07-01'
copper_price = copper_price[copper_price.Date>start_date]
copper_price = copper_price[copper_price.Date<end_date]


for upper in range(60,91,10):
    for lower in range(20,51,10):
        















ticklabel = copper_price.Date
#xticks = np.arange(0,len(ticklabel),100)
plt.figure(figsize = (15,12))
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(copper_price.Date, copper_price.Close,label = 'Close',color=color)
ax1.set_ylabel('Close', color=color)
ax1.tick_params(axis='y',color=color)
#ax1.set_xticks(xticks)
ax1.set_xticklabels(copper_price.Date,rotation = 45)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.plot(copper_price.Date, copper_price['PPO'],label = 'PPO',color=color)
ax2.set_ylabel('PPO', color=color)
ax2.tick_params(axis='y',color=color)
#ax2.set_xticks(xticks)
#ax2.set_xticklabels(ticklabel[xticks],rotation = 45)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

copper_price['Signal'] = pd.Series(index = copper_price.index,data = [0]*len(copper_price.RSI))
for i in range(1,len(copper_price.RSI)):
    if copper_price.RSI.iloc[i-1]>50:
        copper_price.Signal.iloc[i] = -1
    elif copper_price.RSI.iloc[i-1]<30:
        copper_price.Signal.iloc[i] = 1

plt.figure(figsize = (15,12))
buy = np.array(copper_price.Signal)==1
sell = np.array(copper_price.Signal)==-1
sell = np.append(0,sell[:-1])
copper_price['Backtest'] = (copper_price.Close.pct_change(1).fillna(0)*copper_price.Signal+1).cumprod()
copper_price['net_value'] = copper_price.Close/copper_price.Close.iloc[0] 

plt.plot(copper_price.Date,copper_price.Backtest,label = 'STRATEGY')
plt.plot(copper_price.Date,copper_price.net_value,label = 'NET VALUE')
plt.legend()
plt.show()






