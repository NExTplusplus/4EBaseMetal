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
#from model.model_id_embedding import MultiHeadAttention, attention, bilstm
from sklearn.metrics import accuracy_score, f1_score
#from dataset import Dataset
#import config
import psutil
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from model.model_id_embedding import MultiHeadAttention, attention, bilstm
from data.load_data_v16_torch import load_data
from model.logistic_regression import LogReg
from utils.transform_data import flatten
from utils.construct_data import rolling_half_year
from utils.log_reg_functions import objective_function, loss_function
import warnings
import xgboost as xgb
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn import metrics
from sklearn.model_selection import KFold
from utils.version_control_functions import generate_version_params
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
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
    def __init__(self, input_dim, hidden_state, time_step, lr, dropout, split, attention_size, embedding_size, train_X, train_y, test_X, test_y):
        self.window_size = time_step
        self.feature_size = input_dim
        #the case number is the number of the metal we want to predict
        self.case_number = 6
        self.train_day = len(train_y)/self.case_number
        self.test_day = len(test_y)/self.case_number
        # Network
        self.split = split
        self.lr = lr
        self.hidden_state = hidden_state
        self.dropout = dropout
        self.loss_func = nn.MSELoss()
        self.embedding_size = embedding_size
        # attention
        self.attention_size = attention_size
        
        # get the train data and test data
        self.train_X = train_X
        self.train_y = train_y.values
        self.test_X = test_X
        self.test_y = test_y.values


    def train_minibatch(self, num_epochs, batch_size, interval):
        start = time.time()
        net = bilstm(input_dim=self.feature_size, hidden_dim=self.hidden_state, num_layers=2,lag=self.window_size, h=self.attention_size, dropout=self.dropout,case_number=self.case_number,embedding_size=self.embedding_size)
        end = time.time()
        batch_len = 125
        print("net initializing with time: {}".format(end-start))
        train_day =  int(self.train_day*self.split)
        val_day = self.train_day-train_day
        #start to prepare the train data
        start = time.time()
        test_y_class = []
        for item in self.test_y:
            if item >0:
                test_y_class.append(1)
            else:
                test_y_class.append(0)
        end = time.time()
        print("preparing training and testing date with time: {}".format(end-start))
        
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_func = self.loss_func   
        
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        

        train_prediction = []
        test_prediction = []
        #val_X_tensor = torch.from_numpy(X_validation)
        #val_Y_tensor = torch.from_numpy(y_validation)
        lowest_loss = 111111

        for epoch in range(num_epochs):
            current_train_pred = []
            current_test_pred = []

            '''start train'''
            net.train()
            start = time.time()
            print('current epoch:', epoch+1)
            
            loss_sum = 0
            #begin to train the data
            for day in range(train_day):
                start_pos = day*self.case_number
                train_X_repredict = self.train_X[start_pos:start_pos+self.case_number]
                train_Y_repredict = self.train_y[start_pos:start_pos+self.case_number]
                train_X_tensor = torch.FloatTensor(train_X_repredict)
                train_Y_tensor = torch.FloatTensor(train_Y_repredict)
                var_x_train_id = torch.LongTensor(list(range(self.case_number)))
                #print("the input size is {}".format(train_X_tensor.size(-1)))
                output=net(train_X_tensor,var_x_train_id)
                loss = loss_func(output, train_Y_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach()
            
            end = time.time()
            train_loss = loss_sum/train_day
            print("train loss is {} with time {}".format(train_loss, end-start))
            

            '''start eval'''
            start = time.time()
            net.eval()
            loss_sum = 0

            '''
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
        '''
        out_pred_train = 0
        out_pred_test = 0
        out_loss = 0
        return out_pred_train, out_pred_test, out_loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the bi-LSTM + attention-based model on stock')
    parser.add_argument(
        '-e', '--epoch', type=int, default=10,
        help='the number of epochs')
    parser.add_argument(
        '-b', '--batch', type=int, default=256,
        help='the mini-batch size')
    parser.add_argument(
        '-i', '--interval', type=int, default=1,
        help='save models every interval epoch')
    parser.add_argument(
        '-lrate', '--lrate', type=float, default=0.01,
        help='learning rate')
    parser.add_argument(
        '-t', '--test', action='store_true',
        help='train or test')
    parser.add_argument(
        '-m', '--model', type=str, default='',
        help='the model name(after encoder/decoder)'
    )
    parser.add_argument(
        '-hidden','--hidden_state',type=int, default=16,
        help='number of hidden_state of encoder/decoder'
    )
    parser.add_argument(
        '-split', '--split', type=float, default=0.8,
        help='the split ratio of validation set')
    parser.add_argument(
        '-d','--drop_out', type=float, default = 0.1,
        help='the dropout rate of LSTM network'
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
        '--data_configure_file', '-c', type=str,
        help='configure file of the features to be read',
        default='exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
    )
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                        type=str, default="LME_Co_Spot")
    parser.add_argument('-s','--steps',type=int,default=5,
                        help='steps in the future to be predicted')
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "NExT"
    )
    parser.add_argument(
        '-l','--lag', type=int, default = 5, help='lag'
    )
    parser.add_argument(
        '-v','--version', help='version', type = str, default = 'v10'
    )
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    parser.add_argument('-xgb','--xgboost',type = int,help='if you want to train the xgboost you need to inform us of that',default=0)
    args = parser.parse_args()
    if args.ground_truth =='None':
        args.ground_truth = None

    os.chdir(os.path.abspath(sys.path[1]))
    # read data configure file
    with open(os.path.join(os.path.abspath(sys.path[1]),args.data_configure_file)) as fin:
        fname_columns = json.load(fin)
    args.ground_truth = args.ground_truth.split(",")
    # prepare for the data
    time_horizon = args.steps
    if args.action == 'train':
        comparison = None
        n = 0
        for f in fname_columns:
            lag = args.lag
            if args.source == "NExT":
                from utils.read_data import read_data_NExT
                data_list, LME_dates = read_data_NExT(f, "2003-11-12")
                time_series = pd.concat(data_list, axis = 1, sort = True)
            elif args.source == "4E":
                from utils.read_data import read_data_v5_4E
                time_series, LME_dates = read_data_v5_4E("2003-11-12")
            length = 5
            split_dates = rolling_half_year("2009-07-01","2017-01-01",length)
            split_dates  =  split_dates[:]
            importance_list = []
            version_params=generate_version_params(args.version)
            for split_date in split_dates:
                horizon = args.steps
                norm_volume = "v1"
                norm_3m_spread = "v1"
                norm_ex = "v1"
                len_ma = 5
                len_update = 30
                tol = 1e-7
                if args.xgboost==1:
                    #print(args.xgboost)
                    norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                                'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':True}
                else:
                    norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                                'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
                tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                                                'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
                final_X_tr = []
                final_y_tr = []
                final_X_va = []
                final_y_va = []
                final_X_te = None
                final_y_te = None 
                ts = copy(time_series.loc[split_date[0]:split_date[2]])
                i = 0
                for ground_truth in ['LME_Co_Spot','LME_Al_Spot','LME_Le_Spot','LME_Ni_Spot','LME_Zi_Spot','LME_Ti_Spot']:
                    print(ground_truth)
                    metal_id = [0,0,0,0,0,0]
                    metal_id[i] = 1
                    X_tr, y_tr, X_va, y_va, X_te, y_te, norm_check,column_list,date_list = load_data(copy(ts),LME_dates,horizon,[ground_truth],lag,split_date,norm_params,tech_params,version_params,torch)

                    new_date_list = date_list[:len(date_list)]
                    new_time_series = time_series.loc[new_date_list]
                    Co_list = []
                    Al_list = []
                    Le_list = []
                    Mi_list = []
                    Zi_list = []
                    Ti_list = []
                    spot_list = np.array(new_time_series[ground_truth])
                    spot_list[1:]=np.log(np.true_divide(spot_list[1:], spot_list[:-1]))
                    normal_spot_list = spot_list[1:]
                    new_date_list = date_list[1:len(date_list)]
                    new_time_series = time_series.loc[new_date_list]
                    new_spot_list = []
                    for j in range(5,len(normal_spot_list)):
                        new_spot_list.append(list(normal_spot_list[j-5:j]))
                    spot_array = np.array(new_spot_list)
                    X_tr = np.concatenate(X_tr)[1:]
                    X_tr = X_tr.reshape(len(X_tr),lag*len(column_list[0]))
                    X_tr = np.concatenate((X_tr,spot_array[:len(X_tr)]),axis=1)
                    X_tr = np.append(X_tr,[metal_id]*len(X_tr),axis = 1)
                    y_tr = np.concatenate(y_tr)[1:]
                    X_va = np.concatenate(X_va)
                    X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
                    X_va = np.concatenate((X_va,spot_array[len(X_tr):]),axis=1)
                    X_va = np.append(X_va,[metal_id]*len(X_va),axis = 1)
                    y_va = np.concatenate(y_va)[1:]
                    final_X_tr.append(X_tr)
                    final_y_tr.append(y_tr)
                    final_X_va.append(X_va)
                    final_y_va.append(y_va)
                    i+=1
                final_X_tr = [np.transpose(arr) for arr in np.dstack(final_X_tr)]
                final_y_tr = [np.transpose(arr) for arr in np.dstack(final_y_tr)]
                final_X_tr = np.reshape(final_X_tr,[np.shape(final_X_tr)[0]*np.shape(final_X_tr)[1],np.shape(final_X_tr)[2]])
                final_y_tr = np.reshape(final_y_tr,[np.shape(final_y_tr)[0]*np.shape(final_y_tr)[1],np.shape(final_y_tr)[2]])
                final_X_va = [np.transpose(arr) for arr in np.dstack(final_X_va)]
                final_y_va = [np.transpose(arr) for arr in np.dstack(final_y_va)]
                final_X_va = np.reshape(final_X_va,[np.shape(final_X_va)[0]*np.shape(final_X_va)[1],np.shape(final_X_va)[2]])
                final_y_va = np.reshape(final_y_va,[np.shape(final_y_va)[0]*np.shape(final_y_va)[1],np.shape(final_y_va)[2]])
                column_lag_list = []
                column_name = []
                for i in range(lag):
                    for item in column_list[0]:
                        new_item = item+"_"+str(lag-i)
                        column_lag_list.append(new_item)
                for ground_truth in ['spot_price']:
                    for i in range(lag):
                        new_item = ground_truth+"_"+str(lag-i)
                        column_lag_list.append(new_item)
                print("the self feature size is {}".format(len(column_lag_list)))
                input_dim = int(len(column_lag_list)/lag)
                id_column = []
                id_column.append("Co")
                id_column.append("Al")
                id_column.append("Ni")
                id_column.append("Ti")
                id_column.append("Zi")
                id_column.append("Le")
                column_lag_list.append("Co")
                column_lag_list.append("Al")
                column_lag_list.append("Ni")
                column_lag_list.append("Ti")
                column_lag_list.append("Zi")
                column_lag_list.append("Le")
                train_dataframe = pd.DataFrame(final_X_tr,columns=column_lag_list)
                test_dataframe = pd.DataFrame(final_X_va,columns=column_lag_list)
                column_lag_list.remove("Co")
                column_lag_list.remove("Al")
                column_lag_list.remove("Ni")
                column_lag_list.remove("Ti")
                column_lag_list.remove("Zi")
                column_lag_list.remove("Le")
                # reshape the two dimensens data into three dimensons
                train_X = train_dataframe.loc[:,column_lag_list]
                test_X = test_dataframe.loc[:,column_lag_list]
                train_X_array = np.array(train_X)
                train_X_column = train_X_array[:,:len(train_X_array[0])-5]
                train_X_Spot = train_X_array[:,len(train_X_array[0])-5:]
                train_X_column = np.reshape(train_X_column, (len(train_X_column), lag, input_dim-1))
                train_X_Spot = np.reshape(train_X_Spot, (len(train_X_Spot), lag, 1))
                train_X = np.concatenate((train_X_column,train_X_Spot), axis=2)
                test_X_array = np.array(test_X)
                test_X_column = test_X_array[:,:len(test_X_array[0])-5]
                test_X_Spot = test_X_array[:,len(test_X_array[0])-5:]
                test_X_column = np.reshape(test_X_column, (len(test_X_column), lag, input_dim-1))
                test_X_Spot = np.reshape(test_X_Spot, (len(test_X_Spot), lag, 1))
                test_X = np.concatenate((test_X_column,test_X_Spot), axis=2)
                print("the length of the train is {}".format(len(train_X[0])))
                train_embedding = train_dataframe.loc[:,id_column]
                test_embedding = test_dataframe.loc[:,id_column]
                train_y = pd.DataFrame(final_y_tr,columns=['result'])
                test_y = pd.DataFrame(final_y_va,columns=['result'])
                num_epochs = args.epoch
                batch_size = args.batch
                split = args.split
                interval = args.interval
                lr = args.lrate
                test = args.test
                mname = args.model
                hidden_state = args.hidden_state
                dropout = args.drop_out
                attention_size = args.attention_size
                embedding_size = args.embedding_size
                lambd = args.lambd
                save_loss = args.save_loss
                save_prediction = args.save_prediction
                window_size = lag
                start = time.time()
                trainer = Trainer(input_dim, hidden_state, window_size, lr, dropout, args.split, attention_size, embedding_size,train_X,train_y,test_X,test_y)
                end = time.time()
                print("pre-processing time: {}".format(end-start))

                out_train_pred, out_test_pred, out_loss = trainer.train_minibatch(num_epochs, batch_size, interval)
                
