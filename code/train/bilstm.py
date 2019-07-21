#encoding:utf-8
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],'..')))
import time
import argparse
import json
from data.load_data_v5 import load_data_v5
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ALstm_pytorch import MultiHeadAttention, attention, bilstm
torch.manual_seed(0)


EPOCH = 100
batch_size = 256
# get the data
print("the path is {}".format(sys.path[0]))
data_configure_file='4EBaseMetal-master/exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
with open(os.path.join(sys.path[0],data_configure_file)) as fin:
        fname_columns = json.load(fin)

tra_date = '2003-11-12'
val_date = '2016-06-01'
tes_date = '2016-12-23'
split_dates = [tra_date, val_date, tes_date]
for f in fname_columns:
    for lag in [20]:
        for norm_volume in ["v2"]:
            horizon = 3
            norm_3m_spread = "v1"
            norm_ex = "v1"
            len_ma = 5
            len_update = 30
            tol = 1e-7
            norm_params = {'vol_norm':norm_volume, 'ex_spread_norm':norm_ex,'spot_spread_norm': norm_3m_spread, 
                'len_ma':len_ma, 'len_update':len_update, 'both':3,'strength':0.01
                }
            tech_params = {'strength':0.01,'both':3}
            X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params = load_data_v5(f, horizon, ["LME_Co_Spot"], lag, 
                                                               "NExT", split_dates, 
                                                               norm_params, tech_params)             
            train_X = X_tr[0]
            print('#training examples: ', len(train_X))
            train_Y = y_tr[0]
            test_X = X_va[0]
            test_Y = y_va[0]
            # Hyper Parameters
            validation_X = X_va[0]
            validation_Y = y_va[0]
            net = bilstm(input_dim=123,hidden_dim=16,output_dim=1,num_layers=2,lag=lag)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
            loss_func = torch.nn.BCEWithLogitsLoss()
            val_X_tensor = torch.from_numpy(validation_X)
            val_Y_tensor = torch.from_numpy(validation_Y)
            for epoch in range(EPOCH):
                print('current epoch:', epoch)
                batch_len=256
                while batch_len<len(train_X)-1:
                    train_X_repredict=train_X[batch_len-batch_size:batch_len]
                    train_Y_repredict=train_Y[batch_len-batch_size:batch_len]
                    train_X_tensor=torch.from_numpy(train_X_repredict)
                    train_Y_tensor=torch.from_numpy(train_Y_repredict)
                    #print("trian_Y is {}".format(train_Y_tensor))
                    output=net(train_X_tensor)
                    #print("the output is {}".format(output))
                    loss = loss_func(output, train_Y_tensor)
                    val_output = net(val_X_tensor)
                    val_loss = loss_func(val_output, val_Y_tensor)
                    print("train loss is {}    the validation loss is {}".format(loss,val_loss))
                    optimizer.zero_grad()
                    loss.backward()           
                    optimizer.step()                
                    batch_len+=batch_size
