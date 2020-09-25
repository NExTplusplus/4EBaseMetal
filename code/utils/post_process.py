import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from numpy.linalg import solve
import pandas as pd
from copy import deepcopy
from itertools import combinations

#Return metal name reference under Analyst Report framework
def get_ind_metal(string):
    if 'Al' in string:
        return 'Al'
    elif 'Co' in string:
        return 'Cu'
    elif 'Ni' in string:
        return 'Ni'
    elif 'Le' in string:
        return 'Pb'
    elif 'Ti' in string:
        return 'Xi'
    elif 'Zi' in string:
        return 'Zn'

#read classification predictions that are pre-generated
def read_classification(ground_truth, horizon, date, version):
    return pd.read_csv(os.path.join('result','prediction','ensemble','_'.join([ground_truth,date,str(horizon),version+".csv"])),index_col = 0)

#read uncertainty values which have been precalculated
def read_uncertainty(ground_truth, horizon, date, version, model):
    validation_date = date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" 
    if model == "alstm":
        return pd.read_csv(os.path.join("result","uncertainty",model,version,'_'.join([ground_truth,date,str(horizon),version.split('_')[0],"True.csv"])),index_col = 0)
    elif model == "classification":
        return pd.read_csv(os.path.join("result","uncertainty",model,'_'.join([ground_truth,validation_date,str(horizon),version.split('_')[0]+".csv"])),index_col = 0)
    
#Generate signal based on the uncertainty value and threshold
def uncertainty_signal(pred_uncertainty, threshold, type = "classification", true_uncertainty = None):
    if type == "classification":
        return pred_uncertainty <= threshold
    elif type == "regression":
        return pred_uncertainty <= (true_uncertainty*threshold)

#Generate Regression Uncertainty Value
def generate_regression_uncertainty(spot_price, ground_truth, horizon, window):
    spot = spot_price.loc[:,ground_truth].to_frame()
    ret = (spot.shift(-horizon) - np.array(spot))/np.array(spot)
    return ret.rolling(window).std().shift(horizon)

#read regression predictions that are pre-generated
def read_regression(spot_price, ground_truth, horizon, date, version, t = "price"):
    pred = pd.read_csv(os.path.join('result','prediction','alstm',version,'_'.join([ground_truth,date,str(horizon),version.split("_")[0],"True.csv"])),index_col = 0)
    if t == "return":
        spot = spot_price.shift(horizon).loc[pred.index[0]:pred.index[-1],ground_truth].to_frame()
        pred = (pred - np.array(spot))/np.array(spot)
    return pred

#read substitution prediction from analyst report indicator
def read_substitution_analyst(ground_truth, horizon, date):
    new_ground_truth = get_ind_metal(ground_truth)
    new_date = "".join(date.split("-"))
    analyst_filepath = os.path.join('code','new_Analyst_Report_Chinese','step4_sentiment_analysis',"predict_result",ground_truth,str(horizon))
    f = list(filter(lambda x: new_date in x and len(x.split('_')) >= 6 , os.listdir(analyst_filepath)))[0]
    analyst_prediction = os.path.join('code','new_Analyst_Report_Chinese','step4_sentiment_analysis',"predict_result",ground_truth,str(horizon), f)
    substitute_prediction = analyst_prediction[['date','discrete_score']][analyst_prediction['discrete_score']!=0.0]
    substitute_prediction.set_index('date',inplace = True)
    return substitute_prediction

#generate classification signal for filter
def generate_class_signal(ground_truth, horizon, date, version, class_threshold):
    class_unc = read_uncertainty(ground_truth, horizon, date, version, "classification")
    return uncertainty_signal(class_unc, class_threshold, type = "classification")

#generate regression signal for filter
def generate_reg_signal(spot_price, ground_truth, horizon, date, version, reg_threshold, reg_window):
    reg_unc = read_uncertainty(ground_truth, horizon, date, version, "alstm")
    true_reg_unc = generate_regression_uncertainty(spot_price, ground_truth, horizon, reg_window).loc[reg_unc.index[0]:reg_unc.index[-1],:]
    true_reg_unc.columns = ["uncertainty"]
    reg_signal = uncertainty_signal(reg_unc, reg_threshold, type = "regression",true_uncertainty = true_reg_unc)

    return reg_signal
    
#generate final signal for filter
def generate_final_signal(spot_price, ground_truth, horizon, date, class_version, reg_version, class_threshold, reg_threshold, reg_window):
    class_signal = generate_class_signal(ground_truth, horizon, date, class_version, class_threshold)
    reg_signal = generate_reg_signal(spot_price, ground_truth, horizon, date, reg_version, reg_threshold, reg_window)

    class_pred = (read_classification(ground_truth,horizon,date, class_version)*2-1)*np.array(class_signal*1)
    reg_pred = read_regression(spot_price, ground_truth,horizon,date, reg_version)
    reg_ret_pred = read_regression(spot_price, ground_truth,horizon,date, reg_version,t = "return")*np.array(reg_signal*1)
    temp = []
    for i in reg_pred.index:
        temp.append(1 if reg_ret_pred.loc[i,"Prediction"] != 0 and np.sign(reg_ret_pred.loc[i,"Prediction"]) == np.sign(class_pred.loc[i,"result"]) else 0)
    temp = pd.DataFrame(data = temp, index = reg_pred.index, columns = ["Filter"])
    return temp
















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
    corr = y_.corr(method = matthews_corrcoef)
    corr_ = deepcopy(corr)
    columns = y_.columns.values.tolist()
    for comb in combinations(columns,2):
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
    sorted_corr = sorted(np.array(corr).flatten())[:limit]
    corr = corr.applymap(lambda x:  0 if x in sorted_corr else x )      
    tri = (np.array(corr) - np.identity(np.shape(corr)[0]))*hp
    mat = np.matrix(np.sum(tri,axis = 1)*np.identity(np.shape(corr)[0]) + np.identity(np.shape(corr)[0]) - tri)
    return mat, corr

def prediction_correction(W,y_pred):
    '''
    input:  W       : a numpy matrix that holds information regarding correlation between directional movements
            y_pred  : numpy array of original predictions
    output: ans     : numpy array of tweaked predictions
    '''
    return solve(W,y_pred)