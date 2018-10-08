import argparse
from datetime import datetime
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

from data.load_rnn import load_pure_lstm
from model.lstm import LSTM


if __name__ == '__main__':
    desc = 'the lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--data_configure_file', '-c', type=str,
        help='configure file of the features to be read',
        default='../../exp/lstm_data.conf'
    )
    parser.add_argument('-l', '--lag', help='lag size', type=int, default=10)
    parser.add_argument('-u', '--unit', help='number of hidden units in lstm',
                        type=int, default=8)
    parser.add_argument('-l2', '--alpha_l2', type=float, default=1e-2,
                        help='alpha for l2 regularizer')
    parser.add_argument('-s', '--step', help='steps to make prediction',
                        type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int,
                        default=1024)
    parser.add_argument('-e', '--epoch', help='epoch', type=int, default=150)
    parser.add_argument('-lr', '--learning_rate', help='learning rate',
                        type=float, default=1e-2)
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument(
        '-min', '--model_path', help='path to load model',
        type=str, default='../../exp/lstm/model'
    )
    parser.add_argument(
        '-mout', '--model_save_path', type=str, help='path to save model',
        default='../../exp/lstm/model'
    )
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    parser.add_argument('-f', '--fix_init', type=int, default=0,
                        help='use fixed initialization')
    parser.add_argument('-rl', '--reload', type=int, default=0,
                        help='use pre-trained parameters')
    args = parser.parse_args()
    print(args)

    # parameters = {
    #     'lag': int(args.lag),
    #     'unit': int(args.unit),
    #     'alp': float(args.alpha_l2),
    #     'lr': float(args.learning_rate)
    # }

    tra_date = '2007-01-03'
    val_date = '2015-01-02'
    tes_date = '2016-01-04'
    split_dates = [tra_date, val_date, tes_date]

    # read data configure file
    with open(args.data_configure_file) as fin:
        fname_columns = json.load(fin)

    # load data
    X_tr, y_tr, X_val, y_val, X_tes, y_tes = load_pure_lstm(
        fname_columns, 'LMCADY', 'log_1d_return', split_dates, args.lag,
        args.step
    )

    # initialize the LSTM model
    pure_LSTM = LSTM(parameters=args)


    if args.action == 'train':
        pure_LSTM.train(X_tr, y_tr, X_val, y_val)
    elif args.action == 'test':
        pure_LSTM.test(X_tes, y_tes)