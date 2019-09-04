#encoding:utf-8
import pandas as pd
import numpy as np
import os, sys, time, random
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import MultiHeadAttention, attention, bilstm
from sklearn.metrics import accuracy_score, f1_score
from dataset import Dataset
import config
import psutil

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

thresh = 0.02163874

def memory_usage():
    pid = os.getpid()
    py = psutil.Process(pid=pid)
    memory_use = py.memory_info()[0]/2.**30
    print('memory useage:', memory_use)

class Trainer:
    def __init__(self, hidden_state, time_step, split, lr, dropout, version, attention_size, embedding_size):
        # dataset
        self.dataset = Dataset(time_step, split, 0, version)
        self.train_size, self.test_size = self.dataset.get_size()
        self.window_size = time_step
        self.feature_size = self.dataset.get_num_features()
        self.case_number = self.dataset.case_size
        self.train_day, self.test_day = self.dataset.get_day_size()

        # Network
        self.split = split
        self.lr = lr
        self.hidden_state = hidden_state
        self.dropout = dropout
        self.loss_func = nn.MSELoss()
        self.embedding_size = embedding_size
        # attention
        self.attention_size = attention_size
        
        self.path_name = "../../../data/Pretrained/window_size-"+str(self.window_size)+"-hidden_size-"+str(self.hidden_state)+"-embed_size-"+str(embedding_size)
        

    def train_minibatch(self, num_epochs, batch_size, interval):
        start = time.time()
        net = bilstm(input_dim=self.feature_size, hidden_dim=self.hidden_state, num_layers=2,lag=self.window_size, h=self.attention_size, dropout=self.dropout,case_number=self.case_number,embedding_size=self.embedding_size)
        end = time.time()
        print("net initializing with time: {}".format(end-start))
        
        start = time.time()
        x_train, y_train = self.dataset.get_train_set()
        y_train_class = np.array([1 if ele>thresh else -1 for ele in y_train])

        x_test, y_test = self.dataset.get_test_set()
        y_test_class = np.array([1 if ele>thresh else -1 for ele in y_test])

        end = time.time()
        print("preparing training and testing date with time: {}".format(end-start))
        
        y_train = y_train.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_func = self.loss_func   
        
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        
        train_prediction = []
        test_prediction = []
        
        lowest_loss = 111111

        for epoch in range(num_epochs):
            current_train_pred = []
            current_test_pred = []

            '''start train'''
            net.train()
            start = time.time()
            print('current epoch:', epoch+1)
            
            loss_sum = 0
            
            for day in range(self.train_day):
                start_pos = day*self.case_number

                train_X_repredict = x_train[start_pos:start_pos+self.case_number]
                train_Y_repredict = y_train[start_pos:start_pos+self.case_number]
                train_X_tensor = torch.FloatTensor(train_X_repredict)
                train_Y_tensor = torch.FloatTensor(train_Y_repredict)
                var_x_train_id = torch.LongTensor(list(range(self.case_number)))
                
                output=net(train_X_tensor,var_x_train_id)
                #print(output.shape, train_Y_tensor.shape)
                loss = loss_func(output, train_Y_tensor)
                
                optimizer.zero_grad()
                loss.backward()           
                optimizer.step()                
                loss_sum += loss.detach()

            end = time.time()
            train_loss = loss_sum/self.train_day
            print("train loss is {} with time {}".format(train_loss, end-start))
            

            '''start eval'''
            start = time.time()
            net.eval()
            loss_sum = 0

            for day in range(self.train_day):
                start_pos = day*self.case_number

                train_X = torch.FloatTensor(x_train[start_pos:start_pos+self.case_number])
                train_Y = torch.FloatTensor(y_train[start_pos:start_pos+self.case_number])
                var_x_train_id = torch.LongTensor(list(range(self.case_number)))

                train_output = net(train_X,var_x_train_id)
                loss = loss_func(train_output, train_Y)
                loss_sum += loss.detach()
                
                current_train_pred += list(train_output.detach().view(-1,))

            current_train_class = [1 if ele>thresh else -1 for ele in current_train_pred]

            train_loss = loss_sum/self.train_day
            train_loss_list.append(float(train_loss))                       
            train_acc = accuracy_score(y_train_class, current_train_class)
            train_acc_list.append(train_acc)
            end = time.time()
            
            print('the average train loss is: {}, accuracy is {} with time: {}'.format(train_loss, train_acc,end-start))

            start = time.time()
            
            loss_sum = 0
            for day in range(self.test_day):
                start_pos = day*self.case_number

                test_X = torch.FloatTensor(x_test[start_pos:start_pos+self.case_number])
                test_Y = torch.FloatTensor(y_test[start_pos:start_pos+self.case_number])
                var_x_test_id = torch.LongTensor(list(range(self.case_number)))

                test_output = net(test_X,var_x_test_id)
                loss = loss_func(test_output, test_Y)
                loss_sum += loss.detach()
    
                current_test_pred += list(test_output.detach().view(-1,))
            
            current_test_class = [1 if ele>thresh else -1 for ele in current_test_pred]    
            
            test_loss = loss_sum/self.test_day
            test_loss_list.append(float(test_loss))
                        
            test_acc = accuracy_score(y_test_class, current_test_class)
            test_acc_list.append(test_acc)                        
            
            end = time.time()

            print('the average test loss is {}, accurary is {}, with time: {}'.format(test_loss, test_acc,end-start))
            
            if (epoch+1)%10 == 0:
                current_train_pred = np.array(current_train_pred).reshape(self.dataset.case_size,-1)
                current_test_pred = np.array(current_test_pred).reshape(self.dataset.case_size,-1)

                train_prediction.append([list(ele) for ele in current_train_pred][:10])
                test_prediction.append([list(ele) for ele in current_test_pred][:10])
            
            if interval == -2:
                if test_loss < lowest_loss:
                    torch.save(net.state_dict(),self.path_name)
                    lowest_loss = test_loss
            
        out_loss = pd.DataFrame()
        out_loss['train_loss'] = train_loss_list
        out_loss['test_loss'] = test_loss_list
        out_loss['train_acc'] = train_acc_list
        out_loss['test_acc'] = test_acc_list
        
        out_pred_train = pd.DataFrame()
        out_pred_test = pd.DataFrame()

        for epoch_index in range(int(num_epochs/10)):
            epoch = (epoch_index+1)*10
            out_pred_train[epoch] = train_prediction[epoch_index]
            out_pred_test[epoch] = test_prediction[epoch_index]

        return out_pred_train, out_pred_test, out_loss


def getArgParser():
    parser = argparse.ArgumentParser(description='Train the bi-LSTM + attention-based model on stock')
    parser.add_argument(
        '-e', '--epoch', type=int, default=1,
        help='the number of epochs')
    parser.add_argument(
        '-b', '--batch', type=int, default=1,
        help='the mini-batch size')
    parser.add_argument(
        '-s', '--split', type=float, default=0.8,
        help='the split ratio of validation set')
    parser.add_argument(
        '-i', '--interval', type=int, default=1,
        help='save models every interval epoch')
    parser.add_argument(
        '-l', '--lrate', type=float, default=0.01,
        help='learning rate')
    parser.add_argument(
        '-t', '--test', action='store_true',
        help='train or test')
    parser.add_argument(
        '-m', '--model', type=str, default='',
        help='the model name(after encoder/decoder)'
    )
    parser.add_argument(
        '-hidden','--hidden_state',type=int, default=128,
        help='number of hidden_state of encoder/decoder'
    )
    parser.add_argument(
        '-w','--window_size', type=int, default=10,
        help='the window size/lag of the model'
    )
    parser.add_argument(
        '-d','--drop_out', type=float, default = 0.1,
        help='the dropout rate of LSTM network'
    )
    parser.add_argument(
        '-v','--version', type=int, default = 0,
        help='the version of data set'
    )
    parser.add_argument(
        '-a','--attention_size', type = int, default = 2,
        help='the head number in MultiheadAttention Mechanism'
    )
    parser.add_argument(
        '-embed','--embedding_size', type=int, default = 10,
        help='the size of embedding layer'
    )
    parser.add_argument(
        '-lambd','--lambd',type=float,default = 0,
        help='the weight of classfication loss'
    )
    parser.add_argument(
        '-savep','--save_prediction',type=bool, default=0,
        help='whether to save prediction results'
    )
    parser.add_argument(
        '-savel','--save_loss',type=bool, default=0,
        help='whether to save loss results'
    )
    parser.add_argument(
        '-c','--comment',type=str,default='',
        help='the comment of model version'
    )
    return parser

if __name__ == '__main__':
    args = getArgParser().parse_args()
    num_epochs = args.epoch
    batch_size = args.batch
    split = args.split
    interval = args.interval
    lr = args.lrate
    test = args.test
    mname = args.model
    window_size = args.window_size
    hidden_state = args.hidden_state
    dropout = args.drop_out
    version = args.version
    attention_size = args.attention_size
    embedding_size = args.embedding_size
    lambd = args.lambd
    save_loss = args.save_loss
    save_prediction = args.save_prediction
    comment = args.comment
    
    if version == -2:
        thresh = -0.04344493
    
    start = time.time()
    trainer = Trainer(hidden_state, window_size, split, lr, dropout, version, attention_size, embedding_size)
    end = time.time()
    print("pre-processing time: {}".format(end-start))

    out_train_pred, out_test_pred, out_loss = trainer.train_minibatch(num_epochs, batch_size, interval)
    if save_prediction:
        out_train_pred.to_csv("results_pred/PredResult-batch_size-"+str(batch_size)+"-lr-"+str(lr)+"-window_size-"+str(window_size)+"-hidden_state-"+str(hidden_state)+"-embedding_size-"+str(embedding_size)+"-attention_head-"+str(attention_size)+"-drop_out-"+str(dropout)+"-data_version-"+str(version)+"-"+comment+"train.csv")
        out_test_pred.to_csv("results_pred/PredResult-batch_size-"+str(batch_size)+"-lr-"+str(lr)+"-window_size-"+str(window_size)+"-hidden_state-"+str(hidden_state)+"-embedding_size-"+str(embedding_size)+"-attention_head-"+str(attention_size)+"-drop_out-"+str(dropout)+"-data_version-"+str(version)+"-"+comment+"test.csv")
        print("Training prediction and test prediction saved! ")
    if save_loss:
        out_loss.to_csv("results_loss/LossResult-batch_size-"+str(batch_size)+"-lr-"+str(lr)+"-window_size-"+str(window_size)+"-hidden_state-"+str(hidden_state)+"-embedding_size-"+str(embedding_size)+"-attention_head-"+str(attention_size)+"-drop_out-"+str(dropout)+"-data_version-"+str(version)+"-"+comment+".csv")
