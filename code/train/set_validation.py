import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

from data.load_rnn import load_pure_log_reg


data_configure_file = "../../exp/log_reg_data.conf"
ground_truth = "LMCADY"
tra_date = '2005-01-04'
val_date = '2011-01-04'
tes_date = '2016-01-03'
split_dates = [tra_date, val_date, tes_date]
with open(data_configure_file) as fin:
    fname_columns = json.load(fin)

for horizon in (1,3,5,10,21,63):
    for lag in (5,10,20,40):
        X_tr, y_tr, X_va, y_va, X_te, y_te = load_pure_log_reg(
            fname_columns, ground_truth, 'log_1d_return', split_dates, T = lag,
            S = horizon
        )
    print(horizon)
    print("pos test:"+str(sum(y_te+1)/2))
    print("neg test:"+str(sum(y_te-1)/2))
    print("pos val:"+str(sum(y_va+1)/2))
    print("neg val:"+str(sum(y_va-1)/2))
    print("pos train:"+str(sum(y_tr+1)/2))
    print("neg train:"+str(sum(y_tr-1)/2))
    print("\n")