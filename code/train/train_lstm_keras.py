import argparse
from datetime import datetime
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
import numpy as np
from data.load_rnn import load_pure_lstm
from model.cnn_lstm_keras import lstm_model, fit_model

if __name__ == '__main__':
    timestep = 5 
    lag_day = 1
    lstm_units = 64
    batch_size = 512
    epochs = 100
    tra_date = '2007-01-03'
    val_date = '2015-01-02'
    tes_date = '2016-01-04'
    split_dates = [tra_date, val_date, tes_date]
    with open("D:/project/4EBaseMetal/exp/keras_6metals_data.conf") as fin:
        fname_columns = json.load(fin)
    #form data
    X_train1, Y_train1, X_val1, Y_val1, X_test1, Y_test1, Y_daybefore_val1, Y_daybefore_tes1, unnormalized_bases_val1, unnormalized_bases_tes1, window_size = load_pure_lstm(fname_columns[0], 'LMAHDY', 'log_1d_return', split_dates, timestep, lag_day, using_frame = "keras")
    X_train2, Y_train2, X_val2, Y_val2, X_test2, Y_test2, Y_daybefore_val2, Y_daybefore_tes2, unnormalized_bases_val2, unnormalized_bases_tes2, window_size = load_pure_lstm(fname_columns[1], 'LMCADY', 'log_1d_return', split_dates, timestep, lag_day, using_frame = "keras")
    X_train3, Y_train3, X_val3, Y_val3, X_test3, Y_test3, Y_daybefore_val3, Y_daybefore_tes3, unnormalized_bases_val3, unnormalized_bases_tes3, window_size = load_pure_lstm(fname_columns[2], 'LMNIDY', 'log_1d_return', split_dates, timestep, lag_day, using_frame = "keras")
    X_train4, Y_train4, X_val4, Y_val4, X_test4, Y_test4, Y_daybefore_val4, Y_daybefore_tes4, unnormalized_bases_val4, unnormalized_bases_tes4, window_size = load_pure_lstm(fname_columns[3], 'LMPBDY', 'log_1d_return', split_dates, timestep, lag_day, using_frame = "keras")
    X_train5, Y_train5, X_val5, Y_val5, X_test5, Y_test5, Y_daybefore_val5, Y_daybefore_tes5, unnormalized_bases_val5, unnormalized_bases_tes5, window_size = load_pure_lstm(fname_columns[4], 'LMSNDY', 'log_1d_return', split_dates, timestep, lag_day, using_frame = "keras")
    X_train6, Y_train6, X_val6, Y_val6, X_test6, Y_test6, Y_daybefore_val6, Y_daybefore_tes6, unnormalized_bases_val6, unnormalized_bases_tes6, window_size = load_pure_lstm(fname_columns[5], 'LMZSDY', 'log_1d_return', split_dates, timestep, lag_day, using_frame = "keras")
    X_train = np.concatenate((X_train1,X_train2,X_train3,X_train4,X_train5,X_train6), axis = 0)
    Y_train = np.concatenate((Y_train1,Y_train2,Y_train3,Y_train4,Y_train5,Y_train6), axis = 0)
    X_val = np.concatenate((X_val1,X_val2,X_val3,X_val4,X_val5,X_val6), axis = 0)
    Y_val = np.concatenate((Y_val1,Y_val2,Y_val3,Y_val4,Y_val5,Y_val6), axis = 0)
    X_test = np.concatenate((X_test1,X_test2,X_test3,X_test4,X_test5,X_test6), axis = 0)
    Y_test = np.concatenate((Y_test1,Y_test2,Y_test3,Y_test4,Y_test5,Y_test6), axis = 0)
    Y_daybefore_val = np.concatenate((Y_daybefore_val1, Y_daybefore_val2, Y_daybefore_val3, Y_daybefore_val4, Y_daybefore_val5, Y_daybefore_val6), axis = 0)
    Y_daybefore_tes = np.concatenate((Y_daybefore_tes1, Y_daybefore_tes2, Y_daybefore_tes3, Y_daybefore_tes4, Y_daybefore_tes5, Y_daybefore_tes6), axis = 0)
    unnormalized_bases_val = np.concatenate((unnormalized_bases_val1,unnormalized_bases_val2,unnormalized_bases_val3,unnormalized_bases_val4,unnormalized_bases_val5,unnormalized_bases_val6), axis = 0)
    unnormalized_bases_tes = np.concatenate((unnormalized_bases_tes1,unnormalized_bases_tes2,unnormalized_bases_tes3,unnormalized_bases_tes4,unnormalized_bases_tes5,unnormalized_bases_tes6), axis = 0)
    #train
    model = lstm_model(lstm_units, window_size, 'linear', 'mse', 'adam')
    model.summary()
    model, training_time = fit_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size, epochs, lstm_units, window_size)

 