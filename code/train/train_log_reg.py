import argparse
from datetime import datetime
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

from data.load_rnn import load_pure_lstm
from model.logistic_regression import LogReg

if __name__ == '__main__':
    desc = 'the logsitc regression model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--data_configure_file', '-c', type=str,
        help='configure file of the features to be read',
        default='/home/ffl/nus/MM/fintech/4e_base_metal/exp/log_reg/log_reg.conf'
    )
    parser.add_argument('-C', '--C', type=float, default=1e-2,
                        help='inverse of regularization')
    parser.add_argument('-s', '--step', help='steps to make prediction',
                        type=int, default=1)
    parser.add_argument('-tol', '--tol', help='tolerance',
                        type=float, default=1e-4)
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument(
        '-min', '--model_path', help='path to load model',
        type=str, default='/home/ffl/nus/MM/fintech/4e_base_metal/exp/lstm/model'
    )
    parser.add_argument(
        '-mout', '--model_save_path', type=str, help='path to save model',
        default='/home/ffl/nus/MM/fintech/4e_base_metal/exp/lstm/model'
    )
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    parser.add_argument('-f', '--fix_init', type=int, default=0,
                        help='use fixed initialization')
    parser.add_argument('-rl', '--reload', type=int, default=0,
                        help='use pre-trained parameters')
    args = parser.parse_args()
    print(args)


    tra_date = '2007-01-03'
    val_date = '2015-01-02'
    tes_date = '2016-01-04'
    split_dates = [tra_date, val_date, tes_date]

    # read data configure file
    with open(args.data_configure_file) as fin:
        fname_columns = json.load(fin)

    if args.action == 'train':
        max_acc = 0.0
        for lag in (5, 10, 20, 40):
            # load data
            X_tr, y_tr, X_val, y_val, X_tes, y_tes = load_pure_log_reg(
                fname_columns, 'LMCADY', 'log_1d_return', split_dates, lag,
                args.step
            )

            # initialize and train the Logistic Regression model
            pure_LogReg = LogReg(parameters=args)

            pure_LogReg.train(X_tr,y_tr)




