#encoding:utf-8
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],'..')))
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# the multihead layer
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
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

# this class is the function to transfer the multiheadlayer
class attention(nn.Module):
    def __init__(self, dim, num_units,h,dropout):
        super(attention, self).__init__()
        """
        Args:
            dim: (int) the hidden layer output dimension which is the same with the num_units.
            lag: (int) the time length we trace
            dropout: (int) the dropout rate
        Returns:
            the multihead output:[batch_size, lag, hidden_dim]
        """
        self.dim = dim
        self.num_units = num_units
        self.h = h
        self.dropout = dropout
        self.encoders = self._build_model(self.dim, self.num_units,self.h,self.dropout)

    def _build_model(self, dim, num_units,h,dropout):
        layer = MultiHeadAttention(query_dim=dim, key_dim=dim, num_units=num_units,h=h,dropout_p=dropout)
        return layer

    def forward(self, inputs):
        net_inputs = inputs
        net_inputs = self.encoders(net_inputs, net_inputs)
        return net_inputs
# the lstm layer
class bilstm(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,lag,h,dropout,case_number,embedding_size,alpha=0.01):
        super(bilstm,self).__init__()
        """
        Args:
            input_dim: (int) the feature length
            hidden_dim: (int) the hidden layer length
            num_layers:(int) the number of the layer
            lag: (int) the time length we trace
            dropout: (int) the dropout rate
        Returns:
            the prediction percentage:(torch.Tensor) [batch, seq_len, 1]
        """
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.dropout = dropout
        self.lag = lag
        self.h=h
        self.case_number = case_number
        self.embedding_size = embedding_size
        self.layer=nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim, \
                        num_layers=self.num_layers,batch_first=True, \
                        dropout=self.dropout,bidirectional=True)
        self.selfattention=nn.Sequential(
                attention(self.hidden_dim*2,self.hidden_dim*2,self.h,self.dropout),
            )
        self.embedding = nn.Embedding(self.case_number,self.embedding_size)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.out = nn.Linear(self.hidden_dim*2+self.embedding_size,1,bias=False)
                
    def init_hidden(self,batch_size):
        return (torch.zeros(self.num_layers*self.bi_num,batch_size,self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers*self.bi_num,batch_size,self.hidden_dim).to(self.device))
    
    def forward(self, x, x_id):
        #print(x.shape)
        # train the bilstm layer
        #x = x.permute(1,0,2)
        output, (hn,cn) = self.layer(x)
        # train the selfattention layer
        x = self.selfattention(output)
        x = x.permute(0,2,1)
        # the max pooling layer
        pool = nn.MaxPool1d(kernel_size=self.lag,stride = 1,padding=0,dilation=1)
        x = pool(x)
        x = torch.squeeze(x)
        # the final prediction layer
        embed = self.embedding(x_id)
        embed = self.leakyrelu(embed)
        linear_in = torch.cat([x,embed],1)
        result = self.out(linear_in)
        return result

