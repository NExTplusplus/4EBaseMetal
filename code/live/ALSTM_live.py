import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import time
# import psutil
from copy import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
print(sys.path)
from model.model_embedding import MultiHeadAttention, attention, bilstm
from train.grid_search import grid_search_alstm
import utils.general_functions as gn
from utils.data_preprocess_version_control import generate_version_params
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

thresh = 0


# def memory_usage():
#     pid = os.getpid()
#     py = psutil.Process(pid=pid)
#     memory_use = py.memory_info()[0]/2.**30
#     print('memory useage:', memory_use)

class Trainer:
    def __init__(self, input_dim, hidden_state, time_step, lr, dropout,
                case_number, attention_size, embedding_size,
                train_X, train_y,
                test_X, test_y,
                val_X, val_y,
                final_train_X_embedding,
                final_test_X_embedding,
                final_val_X_embedding,
                test_y_class_case,
                test_y_class_top_case,
                test_y_class_bot_case,
                test_y_top_ind_case,
                test_y_bot_ind_case
                ):
        # dataset
        self.window_size = time_step
        self.feature_size = input_dim
        #the case number is the number of the metal we want to predict
        self.case_number = case_number
        self.train_day = len(train_y)/self.case_number
        self.test_day = len(test_y)/self.case_number
        # Network
        self.lr = lr
        self.hidden_state = hidden_state
        self.dropout = dropout
        self.loss_func = nn.MSELoss()
        self.embedding_size = embedding_size
        # attention
        self.attention_size = attention_size
        # get the train data and test data
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.val_X = val_X
        self.val_y = val_y
        self.train_embedding = final_train_X_embedding
        self.test_embedding = final_test_X_embedding
        self.val_embedding = final_val_X_embedding

        self.test_y_class_case = test_y_class_case
        self.test_y_class_top_case = test_y_class_top_case
        self.test_y_class_bot_case = test_y_class_bot_case
        self.test_y_top_ind_case = test_y_top_ind_case
        self.test_y_bot_ind_case = test_y_bot_ind_case

    def evaluate_by_case(self, pred_class):
        pred_class_case = []
        # split predictions into cases and check whether length match
        start_index = 0
        for case in self.test_y_class_case:
            examples_case = len(case)
            pred_class_case.append(pred_class[start_index:start_index + examples_case])
            start_index += examples_case
        assert len(pred_class) == start_index, \
            'number of predictions {} does not match number of labels {}'.\
                format(len(pred_class), start_index)

        # performance by case
        for case in zip(self.test_y_class_case, pred_class_case):
            print('case acc:', accuracy_score(case[0], case[1]))

        # top & bottom performance by case
        for i, case in enumerate(pred_class_case):
            case = np.array(case)
            top_acc = accuracy_score(self.test_y_class_top_case[i],
                                     case[self.test_y_top_ind_case[i]])
            bot_acc = accuracy_score(self.test_y_class_bot_case[i],
                                     case[self.test_y_bot_ind_case[i]])
            print('top acc: {:.4f} ::: bot acc: {:.4f}'.format(top_acc, bot_acc))


    def train_minibatch(self, num_epochs, batch_size, interval, version, horizon, split_dates, drop_out, hidden_state, embedding_size, lag, method):
        start = time.time()
        net = bilstm(input_dim=self.feature_size,
                    hidden_dim=self.hidden_state,
                    num_layers=2,
                    lag=self.window_size,
                    h=self.attention_size,
                    dropout=self.dropout,
                    case_number=self.case_number,
                    embedding_size=self.embedding_size)
        end = time.time()
        print("net initializing with time: {}".format(end-start))
        start = time.time()
        val_y_class = []
        for item in self.val_y:
            if item >= thresh:
                val_y_class.append(1)
            else:
                val_y_class.append(0)
        test_y_class = []
        for item in self.test_y:
            if item >= thresh:
                test_y_class.append(1)
            else:
                test_y_class.append(0)
        end = time.time()
        print("preparing training and testing date with time: {}".format(end-start))

        # define the optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_func = self.loss_func   
        val_loss_list = []
        val_f1_list = []
        val_acc_list = []
        test_loss_list = []
        test_f1_list = []
        test_acc_list = []
        
        val_prediction = []
        test_prediction = []
        
        lowest_loss = 111111
        train_size = len(self.train_X)
        val_size = len(self.val_X)
        test_size = len(self.test_X)
        max_loss = 10
        #begin to train
        for epoch in range(num_epochs):
            current_val_pred = []
            current_test_pred = []

            '''start train'''
            net.train()
            print('current epoch:', epoch+1)
            
            i = 0
            loss_sum = 0

            while i < train_size:
                batch_end = i + batch_size
                if batch_end > train_size:
                    batch_end = train_size
                train_X_repredict=np.array(self.train_X[i:batch_end])
                train_Y_repredict=np.array(self.train_y[i:batch_end])
                train_X_tensor=torch.from_numpy(train_X_repredict).float()
                train_Y_tensor=torch.from_numpy(train_Y_repredict).float()
                var_x_train_id = torch.LongTensor(np.array(self.train_embedding[i:batch_end]))

                #get the output of the neural network
                output=net(train_X_tensor,var_x_train_id)
                loss = loss_func(output, train_Y_tensor)
                
                optimizer.zero_grad()
                loss.backward()           
                optimizer.step()                
                loss_sum += loss.detach()*(batch_end-i)
                i = batch_end
            train_loss = loss_sum/train_size
            print("train loss is {:.6f}".format(train_loss))
            
            if_eval_train = 1

            #start eval
            net.eval()
            loss_sum = 0
            if if_eval_train:
                val_X = torch.from_numpy(self.val_X).float()
                val_Y = torch.from_numpy(self.val_y).float()
                var_x_val_id = torch.LongTensor(np.array(self.val_embedding))
                val_output = net(val_X, var_x_val_id)
                loss = loss_func(val_output, val_Y)
                loss_sum = loss.detach()
                current_val_pred = list(val_output.detach().view(-1, ))
                current_val_class = [1 if ele>thresh else 0 for ele in current_val_pred]

                val_loss = loss_sum
                val_loss_list.append(float(val_loss))

                val_acc = accuracy_score(val_y_class, current_val_class)
                val_acc_list.append(val_acc)
            
                print('average val loss: {:.6f}, accuracy: {:.4f}'.format(
                    val_loss, val_acc)
                )

            test_X = torch.from_numpy(self.test_X).float()
            test_Y = torch.from_numpy(self.test_y).float()
            var_x_test_id = torch.LongTensor(np.array(self.test_embedding))

            test_output = net(test_X, var_x_test_id)
            loss = loss_func(test_output, test_Y)
            loss_sum = loss.detach()
            current_test_pred = list(test_output.detach().view(-1,))
            current_test_class = [1 if ele>thresh else 0 for ele in current_test_pred]
            if val_loss < max_loss:
                torch.save(net, os.path.join("result","model","alstm",version+"_"+method,split_dates[1]+"_"+str(horizon)+"_"+str(drop_out)+"_"+str(hidden_state)+"_"+str(embedding_size)+"_"+str(lag)+"_"+version+"_"+'alstm.pkl'))
                np.savetxt(os.path.join("result","probability","alstm",version+"_"+method,split_dates[1]+"_"+str(horizon)+"_"+str(drop_out)+"_"+str(hidden_state)+"_"+str(embedding_size)+"_"+str(lag)+"_"+version+"_"+"prediction.txt"),current_test_class)
                max_loss = val_loss
            test_loss = loss_sum
            test_loss_list.append(float(test_loss))
            
            test_acc = accuracy_score(test_y_class, current_test_class)
            test_acc_list.append(test_acc)

            print('average test loss: {:.6f}, accuracy: {:.4f}'.format(
                test_loss, test_acc)
            )

            # evaluate by case
            self.evaluate_by_case(current_test_class)
        return net




class ALSTM_online():
    """
    lag: the window size of the data feature
    horizon: the time horizon of the predict target
    version: the version of the feature
    gt: the ground_truth metal name
    date: the last date of the prediction
    source: the data source
    """
    def __init__(self,
        lag,
        horizon,
        version,
        gt,
        date,
        source):
        self.lag = lag
        self.horizon = horizon
        self.version = version
        self.gt = gt
        self.date = date
        self.source = source

  #this function is used to choose the parameter
    def tune(self,
            log='./tune.log',
            script='code/train/train_alstm.py',
            drop_out=0.0,
            hidden=50,
            embedding_size=5,
            batch=512):
        print("begin to choose the parameter")

        gn.generate_config_path(self.version)

        #the param we want to use
        selected_parameters = ['lag', 'hidden', 'embedding_size', 'drop_out', 'batch']
        # the range of the param we want to use
        parameter_values = [
            [2, 3, 4, 5],
            [10, 20, 30],
            [5, 10, 20],
            [0.2, 0.4, 0.6],
            [512]
        ]
        #we init the param
        init_para = {
            'drop_out': drop_out,
            'hidden': hidden,
            'embedding_size': embedding_size,
            'batch': batch,
            'lag':self.lag
        }
        #we begin to search the param
        grid_search_alstm(selected_parameters, parameter_values, init_para,
                script=script, log_file=log, steps=self.horizon, version=self.version,
                date = self.date, gt = self.gt, source = self.source)
  #-------------------------------------------------------------------------------------------------------------------------------------#
  
  #this function is used to train the model and save it
    def train(
        self,split = 0.9,
        num_epochs=50,
        drop_out=0.0,
        embedding_size=5,
        batch_size=512,
        hidden_state=50,
        lrate=0.001,
        attention_size=2,
        interval=1,
        lambd=0,
        save_loss=0,
        save_prediction=0,
        method =""):
        """
        drop_out: the dropout rate of LSTM network
        hidden: number of hidden_state of encoder/decoder
        embdedding_size: the size of embedding layer
        batch: the mini-batch size
        hidden_satte: number of hidden_state of encoder/decoder
        lrate: learning rate
        attention_size: the head number in MultiheadAttention Mechanism
        interval: save models every interval epoch
        lambd: the weight of classfication loss
        save_loss: whether to save loss results
        save_prediction: whether to save prediction results
        """
        sys.path[0] = os.curdir
        print(sys.path)
        print("begin to train")

        #identify the configuration file for data based on version
        self.path = gn.generate_config_path(self.version)

        #read the data from the 4E or NExT database with configuration file to determine columns to that are required
        time_series,LME_dates,config_length = gn.read_data_with_specified_columns(self.source,self.path,"2003-11-12")


        for date in self.date.split(","):

            torch.manual_seed(1)
            np.random.seed(1)
            random.seed(1)
            #generate list of dates for today's model training period
            today = date
            length = 5
            if gn.even_version(self.version) and self.horizon > 5:
                length = 4
            start_time,train_time,evalidate_date = gn.get_relevant_dates(today,length,"train")
            split_dates  =  [train_time,evalidate_date,str(today)]
        
        #generate the version
            version_params=generate_version_params(self.version)
            print("the train date is {}".format(split_dates[0]))
            print("the test date is {}".format(split_dates[1]))
            norm_volume = "v1"
            norm_3m_spread = "v1"
            norm_ex = "v1"
            len_ma = 5
            len_update = 30
            tol = 1e-7
            norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                    'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
            tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                            'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2,"live":None}
            #for versions that tune over 6 metals 
            final_X_tr = []
            final_y_tr = []
            final_X_val = []
            final_y_val = []
            final_X_te = []
            final_y_te = []
            final_y_te_class_list = []
            final_y_te_class_top_list = []
            final_y_te_top_ind_list = []
            final_y_te_class_bot_list = []
            final_y_te_bot_ind_list = []
            final_train_X_embedding = []
            final_test_X_embedding = []
            final_val_X_embedding = []

            i = 0
            #toggle metal id
            metal_id = False
            ground_truths_list = ["LME_Cu_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Xi_Spot","LME_Zn_Spot","LME_Pb_Spot"]
            for ground_truth in ground_truths_list:
                new_time_series = copy(time_series)
                spot_list = np.array(new_time_series[ground_truth])
                new_time_series['spot_price'] = spot_list
                ts = new_time_series.loc[start_time:split_dates[2]]

                #load data for use
                X_tr, y_tr, X_va, y_va, val_dates, column_lag_list = gn.prepare_data(ts,LME_dates,self.horizon,[ground_truth],self.lag,copy(split_dates),version_params,metal_id_bool = metal_id,reshape = False)
                        
                # split validation
                X_ta = X_tr[:int(len(X_tr) * split), :, :]
                y_ta = y_tr[:int(len(y_tr) * split)]

                X_val = X_tr[int(len(X_tr) * split):, :, :]
                y_val = y_tr[int(len(y_tr) * split):]

                X_te = X_va
                y_te = y_va

                # generate metal id for embedding lookup
                train_X_id_embedding = [i]*len(X_ta)
                val_X_id_embedding = [i]*len(X_val)
                test_X_id_embedding = [i]*len(X_te)

                if len(final_X_tr) == 0:
                    final_X_tr = copy(X_ta)
                else:
                    final_X_tr = np.concatenate((final_X_tr, X_ta), axis=0)
                if len(final_y_tr) == 0:
                    final_y_tr = copy(y_ta)
                else:
                    final_y_tr = np.concatenate((final_y_tr, y_ta), axis=0)

                if len(final_X_te) == 0:
                    final_X_te = copy(X_te)
                else:
                    final_X_te = np.concatenate((final_X_te, X_te), axis=0)
                if len(final_y_te) == 0:
                    final_y_te = copy(y_te)
                else:
                    final_y_te = np.concatenate((final_y_te, y_te), axis=0)

                y_te_rank = np.argsort(y_te[:,0])
                y_te_class = []
                for item in y_te:
                    if item >= thresh:
                        y_te_class.append(1)
                    else:
                        y_te_class.append(0)
                final_y_te_class_list.append(y_te_class)
                split_position = len(y_te) // 3
                final_y_te_bot_ind_list.append(y_te_rank[:split_position])
                final_y_te_top_ind_list.append(y_te_rank[-split_position:])
                y_te_class = np.array(y_te_class)
                final_y_te_class_bot_list.append(
                y_te_class[y_te_rank[:split_position]])
                final_y_te_class_top_list.append(
                y_te_class[y_te_rank[-split_position:]])

                if len(final_X_val) == 0:
                    final_X_val = copy(X_val)
                else:
                    final_X_val = np.concatenate((final_X_val, X_val), axis=0)
                if len(final_y_val) == 0:
                    final_y_val = copy(y_val)
                else:
                    final_y_val = np.concatenate((final_y_val, y_val), axis=0)

                final_train_X_embedding+=train_X_id_embedding
                final_test_X_embedding+=test_X_id_embedding
                final_val_X_embedding+=val_X_id_embedding

                # update metal index
                i+=1
        
            print('Dataset statistic: #examples')
            print('Train:', len(final_X_tr), len(final_y_tr), len(final_train_X_embedding))
            print(np.max(final_X_tr), np.min(final_X_tr), np.max(final_y_tr), np.min(final_y_tr))
            print('Validation:', len(final_X_val), len(final_y_val), len(final_val_X_embedding))
            print('Testing:', len(final_X_te), len(final_y_te), len(final_test_X_embedding))
            # begin to train the model
            input_dim = final_X_tr.shape[-1]
            window_size = self.lag
            case_number = len(ground_truths_list)
            start = time.time()
            trainer = Trainer(input_dim, hidden_state, window_size, lrate,
                    drop_out, case_number, attention_size,
                    embedding_size,
                    final_X_tr, final_y_tr,
                    final_X_te, final_y_te,
                    final_X_val, final_y_val,
                    final_train_X_embedding,
                    final_test_X_embedding,
                    final_val_X_embedding,
                    final_y_te_class_list,
                    final_y_te_class_top_list,
                    final_y_te_class_bot_list,
                    final_y_te_top_ind_list,
                    final_y_te_bot_ind_list
                    )

            end = time.time()

            print("pre-processing time: {}".format(end-start))
            print("the split date is {}".format(split_dates[1]))
            save = 1
            net=trainer.train_minibatch(num_epochs, batch_size, interval, self.version, self.horizon, split_dates, drop_out, hidden_state, embedding_size, self.lag,method)
  #-------------------------------------------------------------------------------------------------------------------------------------#

    #this function is used to generate the predictions
    def test(self,split = 0.9,
        num_epochs=50,
        drop_out=0.0,
        embedding_size=5,
        batch_size=512,
        hidden_state=50,
        lrate=0.001,
        attention_size=2,
        interval=1,
        lambd=0,
        save_loss=0,
        save_prediction=0,
        method = ""):
        sys.path[0] = os.curdir
        print(sys.path)
        print("begin to test")

        #identify the configuration file for data based on version
        self.path = gn.generate_config_path(self.version)

        #read the data from the 4E or NExT database with configuration file to determine columns to that are required
        time_series,LME_dates,config_length = gn.read_data_with_specified_columns(self.source,self.path,"2003-11-12")

        for date in self.date.split(","):
            torch.manual_seed(1)
            np.random.seed(1)
            random.seed(1)

            today = date
            length = 5
            if gn.even_version(self.version) and self.horizon > 5:
                length = 4
            start_time,train_time,evalidate_date = gn.get_relevant_dates(today,length,"test")
            split_dates  =  [train_time,evalidate_date,str(today)]

            

            #generate the version parameters
            version_params=generate_version_params(self.version)
            print("the test date is {}".format(split_dates[1]))
            norm_volume = "v1"
            norm_3m_spread = "v1"
            norm_ex = "v1"
            len_ma = 5
            len_update = 30
            tol = 1e-7
            norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                    'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
            tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                            'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2,"live":None}
            #for versions that tune over 6 metals 
            final_X_tr = []
            final_y_tr = []
            final_X_val = []
            final_y_val = []
            final_X_te = []
            final_y_te = []
            final_y_te_class_list = []
            final_y_te_class_top_list = []
            final_y_te_top_ind_list = []
            final_y_te_class_bot_list = []
            final_y_te_bot_ind_list = []
            final_train_X_embedding = []
            final_test_X_embedding = []
            final_val_X_embedding = []

            i = 0
            #toggle metal id
            metal_id = False
            ground_truths_list = ["LME_Cu_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Xi_Spot","LME_Zn_Spot","LME_Pb_Spot"]
            for ground_truth in ground_truths_list:
                new_time_series = copy(time_series)
                spot_list = np.array(new_time_series[ground_truth])
                new_time_series['spot_price'] = spot_list
                ts = new_time_series.loc[start_time:split_dates[2]]

                #load data for use
                X_tr, y_tr, X_va, y_va, val_dates, column_lag_list = gn.prepare_data(ts,LME_dates,self.horizon,[ground_truth],self.lag,copy(split_dates),version_params,metal_id_bool = metal_id,reshape = False,live = True)
                        
                # split validation
                X_ta = X_tr[:int(len(X_tr) * split), :, :]
                y_ta = y_tr[:int(len(y_tr) * split)]

                X_val = X_tr[int(len(X_tr) * split):, :, :]
                y_val = y_tr[int(len(y_tr) * split):]

                X_te = X_va
                y_te = y_va

                # generate metal id for embedding lookup
                train_X_id_embedding = [i]*len(X_ta)
                val_X_id_embedding = [i]*len(X_val)
                test_X_id_embedding = [i]*len(X_te)

                if len(final_X_tr) == 0:
                    final_X_tr = copy(X_ta)
                else:
                    final_X_tr = np.concatenate((final_X_tr, X_ta), axis=0)
                if len(final_y_tr) == 0:
                    final_y_tr = copy(y_ta)
                else:
                    final_y_tr = np.concatenate((final_y_tr, y_ta), axis=0)

                if len(final_X_te) == 0:
                    final_X_te = copy(X_te)
                else:
                    final_X_te = np.concatenate((final_X_te, X_te), axis=0)
                if len(final_y_te) == 0:
                    final_y_te = copy(y_te)
                else:
                    final_y_te = np.concatenate((final_y_te, y_te), axis=0)

                y_te_rank = np.argsort(y_te[:,0])
                y_te_class = []
                for item in y_te:
                    if item >= thresh:
                        y_te_class.append(1)
                    else:
                        y_te_class.append(0)
                final_y_te_class_list.append(y_te_class)
                split_position = len(y_te) // 3
                final_y_te_bot_ind_list.append(y_te_rank[:split_position])
                final_y_te_top_ind_list.append(y_te_rank[-split_position:])
                y_te_class = np.array(y_te_class)
                final_y_te_class_bot_list.append(
                y_te_class[y_te_rank[:split_position]])
                final_y_te_class_top_list.append(
                y_te_class[y_te_rank[-split_position:]])

                if len(final_X_val) == 0:
                    final_X_val = copy(X_val)
                else:
                    final_X_val = np.concatenate((final_X_val, X_val), axis=0)
                if len(final_y_val) == 0:
                    final_y_val = copy(y_val)
                else:
                    final_y_val = np.concatenate((final_y_val, y_val), axis=0)

                final_train_X_embedding+=train_X_id_embedding
                final_test_X_embedding+=test_X_id_embedding
                final_val_X_embedding+=val_X_id_embedding

                # update metal index
                i+=1
            print('Dataset statistic: #examples')
            print('Testing:', len(final_X_te), len(final_y_te), len(final_test_X_embedding))
            # begin to train the model
            input_dim = final_X_tr.shape[-1]
            window_size = self.lag
            case_number = len(ground_truths_list)
            # begin to predict
            start = time.time()
            test_loss_list = []
            test_X = torch.from_numpy(final_X_te).float()
            test_Y = torch.from_numpy(final_y_te).float()
            var_x_test_id = torch.LongTensor(np.array(final_test_X_embedding))
            net = torch.load(os.path.join('result','model','alstm',self.version+"_"+method,split_dates[1]+"_"+str(self.horizon)+"_"+str(drop_out)+"_"+str(hidden_state)+"_"+str(embedding_size)+"_"+str(self.lag)+"_"+self.version+"_"+'alstm.pkl'))
            net.eval()
            test_output = net(test_X, var_x_test_id)
            current_test_pred = list(test_output.detach().view(-1,))

            current_test_class = [1 if ele>thresh else 0 for ele in current_test_pred]
            np.savetxt(os.path.join('result','probability','alstm',self.version+"_"+method,split_dates[1]+"_"+str(self.horizon)+"_"+self.version+".txt"),current_test_class)

            pred_length = int(len(current_test_class)/6)
            for num,gt in enumerate(ground_truths_list):
                final_list = pd.DataFrame(current_test_class[num*pred_length:(num+1)*pred_length],index = val_dates, columns = ["Prediction"])
                final_list.to_csv(os.path.join(os.getcwd(),"result","prediction","alstm",self.version+"_"+method,"_".join([gt,date,str(self.horizon),self.version])+".csv"))
            end = time.time()
            print("predict time: {}".format(end-start))
