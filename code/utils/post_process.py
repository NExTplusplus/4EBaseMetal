import numpy as np
from scipy.stats import pearsonr
from numpy.linalg import solve
import pandas as pd
from itertools import combinations


def get_W(y,hp,version):
    corr = y.corr(method = "pearson")
    columns = y.columns.values.tolist()
    for comb in combinations(columns,2):
        if version == 1:
            if (comb[0][-1] != comb[1][-1] and comb[0][:-1] != comb[1][:-1]) or corr.loc[comb[0],comb[1]] < 0:
                corr.loc[comb[0],comb[1]] = 0
                corr.loc[comb[1],comb[0]] = 0
        if version == 2:
            if comb[0][:-1] != comb[1][:-1] or corr.loc[comb[0],comb[1]] < 0:
                corr.loc[comb[0],comb[1]] = 0
                corr.loc[comb[1],comb[0]] = 0
        if version == 3:
            if comb[0][-1] != comb[1][-1] or corr.loc[comb[0],comb[1]] < 0:
                corr.loc[comb[0],comb[1]] = 0
                corr.loc[comb[1],comb[0]] = 0
    tri = (np.array(corr) - np.identity(np.shape(corr)[0]))*hp
    mat = np.matrix(np.sum(tri,axis = 1)*np.identity(np.shape(corr)[0]) + np.identity(np.shape(corr)[0]) - tri)
    return mat

def prediction_correction(W,y_pred):
    return solve(W,y_pred)