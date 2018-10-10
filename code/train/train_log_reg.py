import argparse
from datetime import datetime
import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

from data.load_rnn import load_pure_log_reg
from model.logistic_regression import LogReg
from utils._logistic_loss import _logistic_loss

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
        
        print(fname_columns)
        print(args.ground_truth)
        for lag in (5, 10, 20, 40):
            # load data
            X_tr, y_tr, X_va, y_va, X_te, y_te = load_pure_log_reg(
                fname_columns, args.ground_truth, 'log_nd_return', split_dates, T = lag,
                S = args.step
            )
            

            # initialize and train the Logistic Regression model
            parameters = {"penalty":"l2", "C":args.C, "solver":"liblinear", "tol":args.tol,"max_iter":50, "verbose" : 1}
            pure_LogReg = LogReg(parameters={})

            pure_LogReg.train(X_tr,y_tr, parameters)
            acc = pure_LogReg.test(X_va,y_va)
            if acc > max_acc:
                best_lag = lag
                model = pure_LogReg
                max_acc = acc
                te_acc = pure_LogReg.test(X_te,y_te)
            
        # print(np.shape(X_tr))
        # print(np.shape(X_va))
        # print(np.shape(X_te))
        
        print("Best Lag:" + str(lag))
        print("Accuracy on Validation Set:"+str(max_acc))
        print("Accuracy on Testing Set:"+str(te_acc))
        

      



