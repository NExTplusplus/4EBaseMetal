import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from data.load_data_three_class import load_data
from model.logistic_regression import LogReg
from utils.transform_data import flatten
from utils.construct_data import rolling_half_year
from utils.log_reg_functions import objective_function, loss_function
import warnings
import xgboost as xgb
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn import metrics
from sklearn.model_selection import KFold
from utils.version_control_functions import generate_version_params
import matplotlib.pyplot as plt
if __name__ == '__main__':
    desc = 'the logistic regression model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--data_configure_file', '-c', type=str,
        help='configure file of the features to be read',
        default='exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
    )
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                        type=str, default="LME_Co_Spot")
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "NExT"
    )
    args = parser.parse_args()
    if args.ground_truth =='None':
        args.ground_truth = None
    os.chdir(os.path.abspath(sys.path[0]))
    # read data configure file
    with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)
    args.ground_truth = args.ground_truth.split(",")
    if args.action == 'train':
        comparison = None
        n = 0
        for f in fname_columns:
            #lag = args.lag
            if args.source == "NExT":
                from utils.read_data import read_data_NExT
                data_list, LME_dates = read_data_NExT(f, "2003-11-12")
                time_series = pd.concat(data_list, axis = 1, sort = True)
            elif args.source == "4E":
                from utils.read_data import read_data_v5_4E
                time_series, LME_dates = read_data_v5_4E("2003-11-12")
            #time_series = time_series.dropna()
            for ground_truth in ['LME_Co_Spot','LME_Al_Spot','LME_Ni_Spot','LME_Ti_Spot','LME_Zi_Spot','LME_Le_Spot']:
                origin = copy(time_series[ground_truth]).dropna()
                x_origin = np.log( origin.shift(-5) / origin )[:-5]
                y_origin = range(len(x_origin[100:]))
                x_mean = []
                for x in range(100,len(x_origin)):
                    x_mean.append(copy(x_origin)[x]-np.mean(copy(x_origin)[x-100:x]))
                mean = []
                for x in range(100,len(x_origin)):
                    mean.append(np.mean(copy(x_origin)[x-100:x]))
                fig, axs = plt.subplots(3, 1, constrained_layout=True)
                axs[0].set_title(ground_truth)
                axs[0].plot(y_origin,origin[100:-5])
                fig.suptitle('This is a comparison of the three figure', fontsize=16)
                axs[1].set_title(ground_truth+"_mean")
                axs[1].plot(y_origin,mean[:])
                axs[2].set_title(ground_truth+"_decress the mean")
                axs[2].plot(y_origin,x_mean[:])
                plt.show()
                #print("the time_series is {}".format(time_series))
