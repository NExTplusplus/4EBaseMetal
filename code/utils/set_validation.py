import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

from data.load_rnn import load_pure_log_reg


data_configure_file = "../../exp/al_log_reg_data.conf"
trading_dates_file = "/home/liangchen/Documents/trading_dates.txt"
ground_truth = "LMAHDY"

with open(data_configure_file) as fin:
    fname_columns = json.load(fin)
for x in fname_columns:
    with open(trading_dates_file) as f:
        dates = f.readlines()
        dates = [str.strip(date) for date in dates]
        dates_length = len(dates)
        horizons = np.zeros([dates_length,12])
        for ind in range(dates_length-1):

            tra_date = dates[0]
            val_date = dates[dates_length-ind-2]
            tes_date = dates[dates_length-ind-1]

            split_dates = [tra_date, val_date, tes_date]
            j = 0
            for horizon in (1,3,5,10,21,63):
                print(horizon)
                lag = 1
                X_tr, y_tr, X_va, y_va, X_te, y_te = load_pure_log_reg(
                    x, ground_truth, 'log_1d_return', split_dates, T = lag,
                    S = horizon
                )
                horizons[dates_length-ind-2,j] = sum(y_va+1)/2
                horizons[dates_length-ind-2,j+1] = -sum(y_va-1)/2
                if ind == 0:
                    horizons[dates_length-1,j] = sum(y_te+1)/2
                    horizons[dates_length-1,j+1] = -sum(y_te-1)/2
                j = j+2

    with open("/home/liangchen/Documents/al_up_and_downs.csv","w") as out:
        for i in range(dates_length):
            for j in range(12):
                if j == 11:
                    out.write(str(horizons[i,j])+"\n")
                else:
                    out.write(str(horizons[i,j])+",")