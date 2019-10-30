import pandas as pd
import numpy as np
from itertools import accumulate
from copy import copy

# This function will calculate Price Volume Trend as mentioned in google drive/ technical indicator for more explanations
# close is the column for closing price and volume is the column for volume.
# When price adjusted volume on up days outpaces that on down day, then PVT rises, vice versa.
# When PVT goes up, the buying pressure is up as well
def pvt (idx,Close,Volume):
    ans = np.log(Close/Close.shift(1))*Volume
    ans = pd.Series(index = idx[1:],data = accumulate(ans.dropna()))
    ans = (ans>ans.shift(1))*1
    ans.loc[ans==0] = -1
    
    return ans

#def divergence_pvt_OI(idx,Close,OI,train_end,params):
#    
#    pvt = np.log(Close/Close.shift(1))*OI
#    pvt = pd.Series(index = idx[1:],data = accumulate(pvt.dropna()))
#
#    divPT = pvt/pvt.shift(1) - Close/Close.shift(1)
#    temp = sorted(copy(divPT[:train_end]))
#    mx = temp[-1]
#    mn = temp[0]
#    if params['both'] == 1:
#        mx = temp[int(np.floor((1-params['strength'])*len(temp)))]
#    elif params['both'] == 2:
#        mn = temp[int(np.ceil(params['strength']*len(temp)))]
#    elif params['both'] == 3:
#        mx = temp[int(np.floor((1-params['strength'])*len(temp)))]
#        mn = temp[int(np.ceil(params['strength']*len(temp)))]
#    divPT = divPT.apply(lambda x: min(x,mx))
#    divPT = divPT.apply(lambda x: max(x,mn))

def divergence_pvt(close,volume,train_end,params):
    
    pvt = close.shift(1)
    pvt.iloc[1] = ((close.iloc[1]/close.iloc[0])-1)*volume.iloc[1]
    for i in range(2,len(close)):
    	pvt.iloc[i] = ((close.iloc[i]/close.iloc[i-1])-1)*volume.iloc[i] + pvt.iloc[i-1]  
    
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

#This function will calculate the volatility scissor difference.
#High is a series of high price per day, so on so as. Window is an integer
#This indicator is a measure of difference between upward volatility and downward volatility within one day.
#The benchmark of this indicator is open price.
def vsd(High,Low,Open,window):
    sdiff = (High - 2*Open + Low)/Open
    sdiff_win = sdiff.rolling(window).mean()
    vsd = pd.Series(index = High.index,data = [0]*len(High))
    tmp = np.array(sdiff_win)>0
    vsd.iloc[1:] = tmp[:-1]*1
    return vsd
#This function will calculate Bollinger Bands.
#Close is a series of close price.
#When the price of the commodity considered is volatile, the bands tend to expand.
#When of close price is higher than the upper band,you get an "overbought" signal.
def bollinger(Close,window, limiting_factor = 2):
    middle = Close.rolling(window).mean()
    rrange = Close.rolling(window).std()
    upper = middle + limiting_factor*rrange
    lower = middle - limiting_factor*rrange
    
    bollinger = pd.Series(index = Close.index,data = [0]*len(Close))
    bollinger.loc[Close>upper] = 1
    bollinger.loc[Close<lower] = -1
    
    return bollinger

import talib as ta

#This function will calculate the Normalized Average True Range
#High is a series of high price per day, so on so as
#NART is a measure of volatilty normalized by close price, more comparable across securities.
def natr(High,Low,Close,window):
    
    return ta.NATR(High,Low,Close,timeperiod = window)


#This function will calculate the Exponential Moving Average.
#Close is a series of close price.
#The EMA is normalized by close price, will be more comparable accross the price on difference days.
def ema(Close,window = 12):
    tmp = ta.EMA(Close,timeperiod = window)
    ema = tmp/Close
    
    return ema

#This function will calculate the percntage price oscillator.
#Close is a series of close price.Both fast and slow are integer.
#PPO measures the difference between two moving averages as a percnetage of the larger moving average.
#If PPO is higher than 1, the more recently price is higher.
def ppo(Close,fast = 12,slow = 26):
    tmp_ppo = ta.PPO(Close,fastperiod = fast,slowperiod = slow, matype = 0)
    ppo = pd.Series(index = Close.index,data = [0]*len(Close))
    ppo.loc[tmp_ppo>0] = 1
    ppo.loc[tmp_ppo<=0] = -1
    
    return ppo

#This function will calculate the Volatility Based Momentum, a volatility-adjusted measure of momentum.
##Close is a series of close price,so on so as. Window is an integer
def vbm(High,Low,Close,window):
    atr = ta.ATR(High,Low,Close,timeperiod = window)
    vbm = (Close - Close.shift(window))/atr
    
    return vbm

#This function will calculate the "Stop and Reverse",
#Which is used to determine trend direnction and potential reversals in price.
#High is a series of high price, so on so as
#Acceleration factor starts at 0.02, and increases by 0.02, up to a maximum of 0.2.
def sar(High,Low,Close,initial=0.02,maximum = 0.2):
    tmp_sar = ta.SAR(High,Low,acceleration = initial, maximum = maximum)
    sar =  pd.Series(index = High.index,data = [0]*len(High))
    sar.loc[Close>tmp_sar] = 1
    sar.loc[Close<tmp_sar] = -1
    
    return sar

#This function will calculate the Relative Strength Index.
#RSI is a siganl of overbought and oversold, more sensitive for bearish market.
def rsi(Close,period = 14,upper = 70,lower = 30):
    rsi = ta.RSI(Close,timeperiod = period)
    rsi_signal = pd.Series(index = Close.index,data = [0]*len(Close))
    for i in range(1,len(rsi_signal)):
        if rsi.iloc[i-1]>upper:
            rsi_signal.iloc[i] = -1
        elif rsi.iloc[i-1]<lower:
            rsi_signal.iloc[i] = 1
#    rsi.loc[rsi>upper] = -1
#    rsi.loc[rsi<lower] = 1
#    rsi.loc[abs(rsi)>1] = 0
#    rsi_signal.iloc[1:] = rsi.iloc[:-1]
#    rsi_signal = rsi_signal.shift(1)
    
    return rsi_signal


#This function will generate the predictions based on the 1st strategy
#Short and Medium is window length of short-term and medium-term EMA respectively.
def strategy_1(Close,Short,Medium):
    short_ema = ta.EMA(Close,Short)
    medium_ema = ta.EMA(Close,Medium)
    ans = pd.Series(index = Close.index,data = [0]*len(Close))
    ans.loc[short_ema>medium_ema] = 1
    ans.loc[short_ema<medium_ema] = -1
    ans = ans[Medium:]

    return ans

#This function will generate the predictions based on the 2nd strategy
#N is the window length, the amount of days for price comparison
def strategy_2(Close,N):
    ans = pd.Series(index = Close.index,data = [0]*len(Close))
    ans.loc[Close>Close.shift(N)] = 1
    ans.loc[Close<Close.shift(N)] = -1
    ans = ans[N:]
    
    return ans
#This function will generate the predictions based on the 3rd strategy
#price can be either High Price or Close Price, window is the amount of days for our price difference comparison
def strategy_3(price,window):
    ans = copy(price)
    max_price = ans.rolling(window).max().shift(1).dropna()
    min_price = ans.rolling(window).min().shift(1).dropna()
    ans = ans[window:]
    temp_price = copy(ans)
    ans.loc[temp_price > max_price] = 1
    ans.loc[temp_price < min_price] = -1
    ans.loc[abs(ans) != 1] = 0

    return ans
#This function will generate the predictions based on the 4rd strategy(volatility filter)
def strategy_4(High, Low, Close,window,limiting_factor):
    ans = copy(Close)
    atr = ta.ATR(High,Low,Close,timeperiod = window)
    HHH = Close.rolling(window).max()
    LLL = Close.rolling(window).min()
    HLGL = (HHH+LLL)*limiting_factor-(HHH-LLL)
    ans.loc[HLGL>atr.shift(1)] = 1
    ans.loc[HLGL<atr.shift(1)] = -1
    ans.loc[abs(ans)!=1]=0
    
    return ans
#This function is used to calculate rolling std.
def strategy_5(Close,window):
    ans = copy(Close)
    ans = Close.rolling(window).std()
    
    return ans

#This function is used to construct long-short signal based on ATR.
#Limiting_factor and window are inputs from config file.
def strategy_6(High, Low, Close, window, limiting_factor):
    ATR = ta.ATR(High,Low,Close,timeperiod = window)
    MA = Close.rolling(window).mean()
    ans = copy(Close)
    ans.loc[Close > MA + limiting_factor*ATR] = 1
    ans.loc[Close < MA - limiting_factor*ATR] = -1
    ans.loc[abs(ans) != 1] = 0

    return ans

#This function is used to construct long-short signal based on bollinger band.
#Limiting_factor and window are inputs from config file.
def strategy_7(Close, window, limiting_factor):
    BB = bollinger(Close, window, limiting_factor)
    MA = Close.rolling(window).mean()
    ans = copy(Close)
    ans.loc[(BB==1)&(Close.shift(1) > MA.shift(1))] = 1
    ans.loc[(BB==-1)&(Close.shift(1) < MA.shift(1))] = -1
    ans.loc[abs(ans) != 1] = 0

    return ans

#This function is used to construct long-short signal based on MACD band.
#FastLength, SlowLength and MACDLength are inputs from config file.
def strategy_9(Close, FastLength, SlowLength, MACDLength):
    MACDValue = Close.rolling(FastLength).mean() - Close.rolling(SlowLength).mean()
    AvgMACD = MACDValue.rolling(MACDLength).mean()
    MACDDiff = MACDValue - AvgMACD
    ans = copy(Close)
    ans.loc[(MACDValue>0)&(MACDDiff>0)] = 1
    ans.loc[(MACDValue<0)&(MACDDiff<0)] = -1
    ans.loc[abs(ans) != 1] = 0

    return ans

#This function is used to construct features indicate the general trend of security price.
#trend1: compare the price of last day and first day of target periods
#trend2: compute the num of upward and downward movement.
#num_of_month is input from config, 1,3,6 month.

def trend_1(Date,Close,num_of_month):
    '''
    EXAMPLE:
    num_of_month = 1
    start_month:Aug 
    end_month = Sep
    target_month = Oct
    (start,end) = [1st Aug,1st Sep)
    (end,target) = [1st Sep,1st Oct)
    '''
    ans = pd.concat([Close,Close],axis = 1)
    ans.columns = ["Trend1","Trend2"]
    start_month = int(str(Date[0]).split('-')[1])    
    end_month = (start_month+num_of_month)%12
    target_month = (end_month+num_of_month)%12
    start = 0
    end = 1
    target = 30
    Return = (Close>Close.shift(1))*1
    while(target<len(Date)):
        
        if (int(str(Date[end]).split('-')[1])!=end_month and (int(str(Date[target]).split('-')[1])!=target_month)):
            end+=1
            target+=1
        elif (int(str(Date[target]).split('-')[1])==target_month) and (int(str(Date[end]).split('-')[1])!=end_month):
            end+=1
        elif (int(str(Date[target]).split('-')[1])!=target_month) and int(str(Date[end]).split('-')[1])==end_month:
            target+=1
        else:
            if start==0:
                ans.Trend1.iloc[start:end] = None
                ans.Trend2.iloc[start:end]= None            
            ans.Trend1.iloc[end:target] = 2*(Close.iloc[end-1]>Close.iloc[start])-1

            tmp = Return.iloc[start:end].sum()/(end-start)
            if tmp>=0.5:
                ans.Trend2.iloc[end:target] = 1
            else:
                ans.Trend2.iloc[end:target] = -1
            start = end
            start_month = int(str(Date[start]).split('-')[1])
            end_month = (start_month+num_of_month)%12
            if end_month == 0:
                end_month = 12
            target_month = (end_month+num_of_month)%12
            if target_month ==0:
                target_month = 12
            end = target
            
    ans.Trend1[end:] = (Close.iloc[end-1]>Close.iloc[start])*2-1
    ans.Trend2[end:] = (Return[start:end].sum()/(end-start)>0.5)*2-1
    return ans.Trend1,ans.Trend2    

def trend_ex1(Close,num_of_days):
    ans = pd.concat([Close,Close],axis = 1)
    ans.columns = ["Trend1","Trend2"]
    Return = (Close>Close.shift(1))*1
    for i in range(num_of_days,len(ans.Trend1)):
        ans.Trend1.iloc[i] = (Return.iloc[i-num_of_days:i+1].sum()/(num_of_days+1)>0.5)*2-1
        ans.Trend2.iloc[i] = (Close.iloc[i]>Close.iloc[i-num_of_days])*2-1
    ans.Trend1.iloc[:num_of_days] = None
    ans.Trend2.iloc[:num_of_days] = None
    
    return ans.Trend1,ans.Trend2

    




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
