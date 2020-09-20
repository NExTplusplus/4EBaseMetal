import numpy as np
import pandas as pd

class Ensemble():

    def __init__(self, version):
        self.version = version

    # voting mechanism
    @staticmethod
    def voting(df, uncertainty):
        '''
        df: dataframe which holds the predictions across all combinations
        '''
        threshold = np.ceil(len(df.columns)/2)
        votes = df.sum(axis = 1)
        res = (votes >= threshold) * 1.0
        if uncertainty:
            return res, abs(0.5 - votes/len(df.columns))
        else:
            return res

    # weighting mechanism
    @staticmethod
    def weight(df, label, window, horizon, uncertainty):
        '''
        df:     dataframe which holds the predictions
        label:  label is the true value of the dates
        window: window is the window size that we calculate errors with
        horizon:horizon is the amount of days we are predicting ahead of
        '''
        error = df.ne(label.to_numpy(),axis = 0)*1.0
        cum_error = error.rolling(window = window,min_periods = 1).sum()
        cum_error = cum_error.shift(horizon).fillna(0) + 1e-6
        rec = (1.0/cum_error).sum(axis = 1)
        pred = (((1.0/cum_error).div(rec.to_numpy(),axis = 0) * df).sum(axis = 1) > 0.5)*1.0
        if uncertainty:
            return pred, abs(0.5 - ((1.0/cum_error).div(rec.to_numpy(),axis = 0) * df).sum(axis = 1))
        else:
            return pred
    
    # generate prediction
    def predict(self, df, label = None, window = None, horizon = None, uncertainty = False):
        temp = df
        if self.version == "vote":
            temp = self.voting(df, uncertainty)
        else:
            assert window is not None and horizon is not None, "input error"
            temp = self.weight(df, label, window, horizon, uncertainty)
        return temp
            
    