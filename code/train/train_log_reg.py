import argparse
from datetime import datetime
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tkinter
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

from data.load_rnn import load_pure_log_reg
from model.logistic_regression import LogReg
from utils.log_reg_functions import objective_function


import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    desc = 'the logistic regression model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--data_configure_file', '-c', type=str,
        help='configure file of the features to be read',
        default='../../exp/log_reg_data.conf'
    )
    parser.add_argument('-C', '--C', type=float, default=1,
                        help='inverse of regularization')
    parser.add_argument('-s', '--step', help='steps to make prediction',
                         type=int, default=1)
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                         type=str, default="Close.Price")
    parser.add_argument('-tol', '--tol', help='tolerance',
                        type=float, default=1e-4)
    parser.add_argument('-max_iter', '--max_iter', help='maximum number of iterations',
                        type=int, default=100)       
    # parser.add_argument('-verbose', '--verbose', help='verbosity',
    #                     type=int, default=1)
    # parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument(
        '-min', '--model_path', help='path to load model',
        type=str, default='../../exp/log_reg/model'
    )
    parser.add_argument(
        '-mout', '--model_save_path', type=str, help='path to save model',
        default='../../exp/log_reg/model'
    )
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    parser.add_argument('-f', '--fix_init', type=int, default=0,
                        help='use fixed initialization')
    parser.add_argument('-rl', '--reload', type=int, default=0,
                        help='use pre-trained parameters')
    args = parser.parse_args()



    tra_date = '2007-01-03'
    val_date = '2015-01-02'
    tes_date = '2016-01-04'
    split_dates = [tra_date, val_date, tes_date]

    # read data configure file
    with open(args.data_configure_file) as fin:
        fname_columns = json.load(fin)


    if args.action == 'train':
        model = None
        best_lag = 0
        max_acc = 0.0
        te_acc = 0.0

        
        
        
        for lag in (5, 10, 20, 40):
            # load data
            X_tr, y_tr, X_va, y_va, X_te, y_te = load_pure_log_reg(
                fname_columns, args.ground_truth, 'log_1d_return', split_dates, T = lag,
                S = args.step
            )

            pure_LogReg = LogReg(parameters={})
            parameters = {"penalty":"l2", "C":args.C, "solver":"lbfgs", "tol":args.tol,"max_iter":args.max_iter, "verbose" : 0,"warm_start":False}
            pure_LogReg.train(X_tr,y_tr, parameters)
            n_iter = pure_LogReg.n_iter()
            step = n_iter/10.0

            steps = np.zeros(10)
            tr_obj = np.zeros(10)
            va_obj = np.zeros(10)
            te_obj = np.zeros(10)
            tr_loss = np.zeros(10)
            va_loss = np.zeros(10)
            te_loss = np.zeros(10)

            # initialize and train the Logistic Regression model
            
            for j in range(10):
                steps[j] = int(np.round(step*(j+1)))
                if j > 0:
                    parameters["max_iter"] = steps[j] - steps[j-1]
                else:
                    parameters["max_iter"] = steps[j]

                pure_LogReg.train(X_tr,y_tr, parameters)
                parameters["warm_start"] = True

                tr_loss[j] = pure_LogReg.log_loss(X_tr,y_tr)
                va_loss[j] = pure_LogReg.log_loss(X_va,y_va)
                te_loss[j] = pure_LogReg.log_loss(X_te,y_te)
                tr_obj[j] = objective_function(pure_LogReg,X_tr,y_tr)
                va_obj[j] = objective_function(pure_LogReg,X_va,y_va)
                te_obj[j] = objective_function(pure_LogReg,X_te,y_te)
            
            acc = pure_LogReg.test(X_va,y_va)
            plt.plot(steps,tr_loss,"blue")
            plt.plot(steps,va_loss,"red")
            plt.plot(steps,te_loss,"green")
            plt.title("Loss")
            plt.savefig("Lag "+str(lag)+" Loss")
            plt.plot(steps,tr_obj, "blue")
            plt.plot(steps,va_obj,"red")
            plt.plot(steps,te_obj,"green")
            plt.title("Objective Function")
            plt.savefig("Lag "+str(lag)+" Objective Function")
            
            if acc > max_acc:
                best_lag = lag
                model = pure_LogReg
                max_acc = acc
                te_acc = pure_LogReg.test(X_te,y_te)

            
        
        print("Best Lag:" + str(lag))
        print("Accuracy on Validation Set:"+str(max_acc))
        print("Accuracy on Testing Set:"+str(te_acc))
        

      



