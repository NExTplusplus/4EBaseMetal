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

torch.manual_seed(0)

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim=123,
                 key_dim=123,
                 num_units=123,
                 dropout_p=0.1,
                 h=8,
                 is_masked=False):
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = float(key_dim)**-0.5
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)
        self.proj_layer = nn.Linear(num_units, num_units)
        self.ln = nn.LayerNorm(query_dim)

    def forward(self, query, keys):
        """
        Args:
            query (torch.Tensor): [batch, seq_len, embed_dim]
            keys (torch.Tensor): [batch, seq_len, embed_dim]
        Returns:
            torch.Tensor: [batch, seq_len, embed_dim]
        """
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        batch_size = query.size(0)
        seq_len = query.size(1)
        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = Q.view(batch_size * self._h, seq_len, chunk_size)
        K = K.view(batch_size * self._h, -1, chunk_size)
        V = V.view(batch_size * self._h, -1, chunk_size)

        # calculate QK^T
        attention = torch.bmm(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attention = attention * self._key_dim
        # use masking (usually for decoder) to prevent leftward
        # information flow and retains auto-regressive property
        # as said in the paper
        if self._is_masked:
            diag_vals = attention[0].sign().abs()
            diag_mat = diag_vals.tril(abs(Q.size(1) - K.size(1)))
            diag_mat = diag_mat.unsqueeze(0).expand(attention.size())
            mask = torch.ones(
                diag_mat.size(), device=query.device) * (-2**32 + 1)
            # this is some trick that I use to combine the lower diagonal
            # matrix and its masking. (diag_mat-1).abs() will reverse the value
            # inside diag_mat, from 0 to 1 and 1 to zero. with this
            # we don't need loop operation andn could perform our calculation
            # faster
            attention = (attention * diag_mat) + (mask * (diag_mat - 1).abs())
        # put it to softmax
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        attention = F.dropout(
            attention, p=self._dropout_p, training=self.training)
        # multiplyt it with V
        attention = torch.bmm(attention, V)
        # convert attention back to its input original size
        attention = attention.view(batch_size, seq_len, -1)

        # apply  projection
        attention = self.proj_layer(attention.view(-1, attention.size(-1)))
        attention = attention.view(batch_size, seq_len, -1)
        # residual connection
        attention += query
        # apply layer normalization
        attention = self.ln(attention)

        return attention

    def init_weight(self):
        nn.init.uniform_(self.query_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.key_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.value_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.proj_layer.weight, -0.1, 0.1)


class bilstm(nn.Module):
    def __init__(self,input_dim=123,hidden_dim=16,output_dim=1,num_layers=2,dropout=0.5):
        super(Net,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers

        self.layer=nn.LSTM(input_size=input_dim,hidden_size=16, \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=True)
        self.attention = MultiHeadAttention()
        query = torch.tensor([batch,lag,123], dtype=torch.long)
        keys = torch.tensor([batch,lag,123], dtype=torch.long)
        self.attention_encoder = nn.Sequential(
        		self.attention(query,keys),
        		nn.Dropout(p=dropout)
        	)

    def init_hidden(self,batch_size):
        return (torch.zeros(self.num_layers*self.bi_num,batch_size,self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers*self.bi_num,batch_size,self.hidden_dim).to(self.device))
    
    def forward(self,x):
        x = self.layer(x)
        x = self.attention_encoder(x)
        x = nn.MaxPool1d(kernel_size=12,stride = 12,padding=0,dilation=2)
        result = F.softmax(x)
        return x


# get the data
data_configure_file='exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
with open(os.path.join(sys.path[0],data_configure_file)) as fin:
        fname_columns = json.load(fin)

tra_date = '2003-11-12'
val_date = '2016-06-01'
tes_date = '2016-12-23'
split_dates = [tra_date, val_date, tes_date]
for f in fname_columns:
    for lag in [5,10,20,30]:
        for norm_volume in ["v1","v2"]:
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
            train_X = list(X_tr[0])
            train_Y = list(y_tr[0])
            test_X = list(X_va[0])
            test_Y = list(y_va[0])
            # Hyper Parameters
            EPOCH = 50           
            BATCH_SIZE = 256     
            INPUT_SIZE = 28     
            LR = 0.01           
            net = bilstm(input_dim=123,hidden_size=64,output_dim=1,num_layers=2)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
            loss_func = torch.nn.CrossEntropyLoss()

            for epoch in range(EPOCH):
                batch_len=5
                while batch_len<len(train_X)-1:
                    train_X=train_X[batch_len-5:batch_len]
                    train_Y=train_Y[batch_len-5:batch_len]
                    train_X_tensor=torch.tensor(train_X, dtype=torch.long)
                    train_Y_tensor=torch.tensor(train_Y, dtype=torch.long)
                    output=net(train_X)
                    loss = loss_func(output, train_Y)
                    optimizer.zero_grad()
                    loss.backward()           
                    optimizer.step()                
                    batch_len+=5

