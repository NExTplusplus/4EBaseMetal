import numpy as np
import pandas as pd
import math

class Dataset:

    def __init__(self, T, split_ratio=0.8, normalized=False, version=0):
        
        if version<4 and version >=0:
            TotalData = pd.read_csv("~/zwy/data/PriceDataAdj_Final_0827.csv")
        else:
            TotalData = pd.read_csv("~/zwy/data/PriceDataAdjReduced_Final_0827.csv")

        target = list(TotalData['ExcessReturnRatio'])
        target = [eval(ele) for ele in target]
        target = np.array(target)
        self.case_size = TotalData.shape[0]
        self.per_ts_size = target.shape[1]
        
        self.T = T
        
        if T==10:
            self.per_data_size = self.per_ts_size-T-4 # take the whole
        elif T==5:
            self.per_data_size = self.per_ts_size-T-4-5 # neglect the first five point

        target = target.reshape(-1,)
        
        self.feature_size_used = 4

        driving = []
        
        if version == -1:
            feature_list = ['ExcessReturnRatio']
            self.feature_size_used = 1
        else:
            feature_list = ['ExcessReturnRatio','ClosePrice', 'DayChangeRate','AmplitudeRate']
        
            if version%2 == 0:
                feature_list += ['MA_5', 'MA_10', 'MA_20']
                self.feature_size_used += 3
            elif version%2 == 1:
                feature_list += ['EMA_5', 'EMA_10', 'EMA_20']
                self.feature_size_used += 3
            if version%4 >= 2:
                feature_list += ['DEA','MACD','Volatility_5','Volatility_10','Volatility_20','TurnoverRateDay','TurnoverRateWeek','TurnoverRateMonth']
                self.feature_size_used += 8
            if version >= 8:
                feature_list += ['PETTM','PB','PSTTM','PCTTM']
                self.feature_size_used += 4
        
        print("Data Version is {}, with feature list of {}, feature size of {}".format(version, feature_list, self.feature_size_used))
        
        for feature in feature_list:
            cur_driving = list(TotalData[feature])
            cur_driving = [eval(ele) for ele in cur_driving]
            cur_driving = np.array(cur_driving)
            cur_driving = list(cur_driving.reshape(-1,))
            driving += cur_driving
        
        driving = np.array(driving)
        driving = driving.reshape(self.feature_size_used,-1)
        driving = driving.transpose()

        # SEEMS NO NEED TO MINUS 1
        self.per_train_size = int(split_ratio * self.per_data_size)
        self.per_test_size = self.per_data_size - self.per_train_size
        self.X, self.y = self.time_series_gen(driving, target, T)
        
        self.train_size = self.per_train_size * self.case_size
        self.test_size = self.per_test_size * self.case_size
        self.data_size = self.per_data_size * self.case_size
        
        print("per_train_size: {}, per_test_size: {}".format(self.per_train_size,self.per_test_size))
 
    def get_size(self):
        return self.train_size, self.test_size

    def get_day_size(self):
        return self.per_train_size, self.per_test_size

    def get_num_features(self):
        return self.feature_size_used

    def get_train_set(self):
        
        train_set = self.X[:self.case_size*self.per_train_size]
        train_set_y = self.y[:self.case_size*self.per_train_size]

        return train_set, train_set_y.reshape(-1,)

    def get_test_set(self):

        test_set = self.X[self.case_size*self.per_train_size:]
        test_set_y = self.y[self.case_size*self.per_train_size:]
        
        return test_set, test_set_y.reshape(-1,)

    def time_series_gen(self, X, y, T):
        ts_x, ts_y= [], []

        if T == 10:
            trunc = 0
        elif T == 5:
            trunc = 5
        
        for day in range(self.per_data_size):
            for i in range(self.case_size):
                start = trunc + i*self.per_ts_size + day
                end = start+T
                ts_x.append(X[start:end,:])
                ts_y.append(y[end+4])

        return np.array(ts_x), np.array(ts_y)
