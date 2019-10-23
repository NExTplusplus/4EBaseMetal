import numpy as np
from scipy.stats import pearsonr
from numpy.linalg import solve
import pandas as pd
from copy import deepcopy
from itertools import combinations


def get_W(y,hp,version,limit,corr_period = None):
    '''
    input:  y       : a pandas dataframe of the true directional movement of all 18 cases 
            hp      : the lambda hyperparameter to regulate correlation between directional movement
            version : 1 generates W that considers both metal type and prediction period
                      2 generates W that considers prediction period
                      3 generates W that considers metal type
    output: mat     : a numpy matrix that will be used to transform our predictions
    '''
    if corr_period is None:
            corr_period = len(y)
    y_ = deepcopy(y.iloc[len(y)-corr_period:,:])
    corr = y_.corr(method = "pearson")
    corr_ = deepcopy(corr)
    p = deepcopy(corr)
    columns = y_.columns.values.tolist()
    for comb in combinations(columns,2):
        p.loc[comb[0],comb[1]] = pearsonr(y.loc[:,comb[0]],y.loc[:,comb[1]])[1]
        p.loc[comb[1],comb[0]] = pearsonr(y.loc[:,comb[0]],y.loc[:,comb[1]])[1]
        if version == 1:
            if (comb[0][-1] != comb[1][-1] and comb[0][:-1] != comb[1][:-1]) or corr.loc[comb[0],comb[1]] < 0:
                corr.loc[comb[0],comb[1]] = 0
                corr.loc[comb[1],comb[0]] = 0
        elif version == 2:
            if comb[0][:-1] != comb[1][:-1] or corr.loc[comb[0],comb[1]] < 0:
                corr.loc[comb[0],comb[1]] = 0
                corr.loc[comb[1],comb[0]] = 0
        elif version == 3:
            if comb[0][-1] != comb[1][-1] or corr.loc[comb[0],comb[1]] < 0:
                corr.loc[comb[0],comb[1]] = 0
                corr.loc[comb[1],comb[0]] = 0
        if corr.loc[comb[0],comb[1]] < limit:
            corr.loc[comb[0],comb[1]] = 0
            corr.loc[comb[1],comb[0]] = 0            
    if hp == 0:
        pd.set_option('display.max_columns',None)
        print(p)
        print(corr)
    tri = (np.array(corr) - np.identity(np.shape(corr)[0]))*hp
    mat = np.matrix(np.sum(tri,axis = 1)*np.identity(np.shape(corr)[0]) + np.identity(np.shape(corr)[0]) - tri)
    return mat

def prediction_correction(W,y_pred):
    '''
    input:  W       : a numpy matrix that holds information regarding correlation between directional movements
            y_pred  : numpy array of original predictions
    output: ans     : numpy array of tweaked predictions
    '''
    return solve(W,y_pred)