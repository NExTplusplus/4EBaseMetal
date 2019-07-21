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
from ALstm_pytorch_model import MultiHeadAttention, attention, bilstm
from sklearn.metrics import accuracy_score
torch.manual_seed(0)


EPOCH = 10
batch_size = 256
# get the data
if __name__ == '__main__':
    desc = 'the Long Short Term Memory model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--data_configure_file', '-c', type=str,
        help='configure file of the features to be read',
        default='4EBaseMetal-master/exp/3d/Co/ALSTM/v5/LMCADY_v5.conf'
    )
    parser.add_argument('-s','--steps',type=int,default=1,
                        help='steps in the future to be predicted')
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                        type=str, default="LME_Co_Spot")
    parser.add_argument('-max_iter','--max_iter',type=int,default=100,
                        help='max number of iterations')
    parser.add_argument(
        '-min', '--model_path', help='path to load model',
        type=str, default='../../exp/ALSTM/model'
    )
    parser.add_argument(
        '-sou','--source', help='source of data', type=str, default = "NExT"
    )
    parser.add_argument(
        '-horizon','--horizon', help='the version of the day we predict', type=int, default=3
        )
    parser.add_argument(
        '-mout', '--model_save_path', type=str, help='path to save model',
        default='4EBaseMetal-master/exp/ALSTM/3d/co/'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    args = parser.parse_args()
    if args.ground_truth =='None':
        args.ground_truth = None
    #print("the path is {}".format(sys.path[0]))
    data_configure_file='4EBaseMetal-master/exp/3d/Co/ALSTM/v5/LMCADY_v5.conf'
    with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)
    tra_date = '2003-11-12'
    val_date = '2016-06-01'
    tes_date = '2016-12-23'
    split_dates = [tra_date, val_date, tes_date]
    args.ground_truth = args.ground_truth.split(",")
    for f in fname_columns:
        for lag in [5,10,20,30]:
            for norm_volume in ["v2","v1"]:
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
                X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params = load_data_v5(f, args.horizon, args.ground_truth, lag, 
                                                               args.source, split_dates, 
                                                               norm_params, tech_params)             
                train_X = X_tr[0]
                #print('#training examples: ', len(train_X))
                train_Y = y_tr[0]
                test_X = X_va[0]
                test_Y = y_va[0]
                # Hyper Parameters
                validation_X = X_va[0]
                validation_Y = y_va[0]
                # if you want to change the hidden_dim, you need to change the parameter.
                print("the lag is {}".format(lag))
                print("the norm_volume is {}".format(norm_volume))
                # the way we choose the parameter
                for hidden in [16,32,64]:
                    for drop in [0.1,0.3,0.5]:
                        for h in [2,4]:
                            print("hidden_dim is {} drop_rate is {} h is {}".format(hidden, drop,h))
                            # if you want to know what parameter means please see the wiki or the ALstm_pyotrch_model.py file
                            net = bilstm(input_dim=123,hidden_dim=16,num_layers=2,lag=lag,h=h,dropout=drop)
                            optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
                            loss_func = torch.nn.BCEWithLogitsLoss()
                            val_X_tensor = torch.from_numpy(validation_X)
                            val_Y_tensor = torch.from_numpy(validation_Y)
                            pre_accuracy = 0
                            for epoch in range(EPOCH):
                                print('current epoch:', epoch)
                                batch_len=256
                                while batch_len<len(train_X)-1:
                                    train_X_repredict=train_X[batch_len-batch_size:batch_len]
                                    train_Y_repredict=train_Y[batch_len-batch_size:batch_len]
                                    train_X_tensor=torch.from_numpy(train_X_repredict)
                                    train_Y_tensor=torch.from_numpy(train_Y_repredict)
                                    output=net(train_X_tensor)
                                    val_output = net(val_X_tensor)
                                    loss = loss_func(output, train_Y_tensor)
                                    val_loss = loss_func(val_output, val_Y_tensor)
                                    print("train loss is {}  the validation function loss is {}".format(loss,val_loss))
                                    optimizer.zero_grad()
                                    loss.backward()           
                                    optimizer.step()                
                                    batch_len+=batch_size
                                # test the validaiton
                                val_output = net(val_X_tensor)
                                val_output_sigmoid = F.sigmoid(val_output).data.numpy()
                                val_output_sigmoid[val_output_sigmoid>=0.5]=1
                                val_output_sigmoid[val_output_sigmoid<0.5]=0
                                print("the validation accuracy is {}".format(accuracy_score(val_output_sigmoid,validation_Y)))
                                # contrast the validation result if the validation is teh best we will save it.
                                if accuracy_score(val_output_sigmoid,validation_Y)>=pre_accuracy:
                                    pre_accuracy=accuracy_score(val_output_sigmoid,validation_Y)
                                    pre_net = net
                                else:
                                    torch.save(pre_net,os.path.join(sys.path[0],args.model_save_path)+"_"+str(lag)+"_"+norm_volume+"_"+str(hidden)+"_"+str(drop)+"_"+str(h)+".pkl")
                                    print("save the model when the validation accuracy is {}".format(pre_accuracy))
                                    break






                                    


