#encoding:utf-8
import pandas as pd
import os, sys, time, random
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score
from copy import copy
import psutil
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from model.model_embedding import MultiHeadAttention, attention, bilstm
from utils.data_preprocess_version_control import generate_version_params
import utils.general_functions as gn
import numpy as np
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

thresh = 0



def memory_usage():
    pid = os.getpid()
    py = psutil.Process(pid=pid)
    memory_use = py.memory_info()[0]/2.**30
    print('memory useage:', memory_use)

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

        self.window_size = time_step
        self.feature_size = input_dim
        #the case number is the number of the metal we want to predict
        self.case_number = case_number
        self.train_day = len(train_y)/self.case_number
        self.test_day = len(test_y)/self.case_number
        # Network
        # self.split = split
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


    def train_minibatch(self, num_epochs, batch_size, interval):
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
            
            test_loss = loss_sum
            test_loss_list.append(float(test_loss))

            test_acc = accuracy_score(test_y_class, current_test_class)
            test_acc_list.append(test_acc)

            print('average test loss: {:.6f}, accuracy: {:.4f}'.format(
                test_loss, test_acc)
            )

            # evaluate by case
            self.evaluate_by_case(current_test_class)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the bi-LSTM + attention-based model on stock')
    parser.add_argument(
        '-e', '--epoch', type=int, default=50,
        help='the number of epochs')
    parser.add_argument(
        '-b', '--batch', type=int, default=1024,
        help='the mini-batch size')
    parser.add_argument(
        '-i', '--interval', type=int, default=1,
        help='save models every interval epoch')
    parser.add_argument(
        '-lrate', '--lrate', type=float, default=0.001,
        help='learning rate')
    parser.add_argument(
        '-t', '--test', action='store_true',
        help='train or test')
    parser.add_argument(
        '-m', '--model', type=str, default='',
        help='the model name(after encoder/decoder)'
    )
    parser.add_argument(
        '-hidden','--hidden_state',type=int, default=50,
        help='number of hidden_state of encoder/decoder'
    )
    parser.add_argument(
        '-split', '--split', type=float, default=0.9,
        help='the split ratio of validation set')
    parser.add_argument(
        '-d','--drop_out', type=float, default = 0.3,
        help='the dropout rate of LSTM network'
    )
    parser.add_argument(
        '-a','--attention_size', type = int, default = 2,
        help='the head number in MultiheadAttention Mechanism'
    )
    parser.add_argument(
        '-embed','--embedding_size', type=int, default = 5,
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
        default='exp/online_v10.conf'
    )
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                        type=str, default="LME_Co_Spot")
    parser.add_argument('-s','--steps',type=int,default=5,
                        help='steps in the future to be predicted')
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "NExT"
    )
    parser.add_argument(
        '-l','--lag', type=int, default=4, help='lag'
    )
    parser.add_argument(
        '-v','--version', help='version', type = str, default = 'v16'
    )
    parser.add_argument(
        '-date','--date',type = str, default= "2017-06-30"
    )
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    args = parser.parse_args()
    if args.ground_truth =='None':
        args.ground_truth = None

    os.chdir(os.path.abspath(sys.path[1]))
    sys.path[0] = os.path.abspath(os.path.join(sys.path[0],"4EBaseMetal"))
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

    # prepare for the data
    time_horizon = args.steps
    if args.action == 'train':
        comparison = None
        n = 0

        #read the data from the 4E or NExT database
        time_series,LME_dates,config_length = gn.read_data_with_specified_columns(args.source,args.data_configure_file,"2003-11-12")

        #generate list of list of dates to be used to roll over 5 half years
        today = args.date
        length = 5
        if gn.even_version(args.version) and time_horizon > 5:
            length = 4
        start_time,end_time = gn.get_relevant_dates(today,length,"tune")
        split_dates = gn.rolling_half_year(start_time,end_time,length)
        split_dates  =  split_dates[:]
        
        importance_list = []
        #generate the version
        version_params=generate_version_params(args.version)


        for s, split_date in enumerate(split_dates):
            lag = args.lag
            horizon = args.steps
            norm_volume = "v1"
            norm_3m_spread = "v1"
            norm_ex = "v1"
            len_ma = 5
            len_update = 30
            tol = 1e-7
            norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                            'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
            tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                                            'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
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
            ground_truth_list = [args.ground_truth]
            if gn.even_version(args.version):
                ground_truth_list = ["LME_Co_Spot","LME_Al_Spot","LME_Ni_Spot","LME_Ti_Spot","LME_Zi_Spot","LME_Le_Spot"]

            for ground_truth in ground_truth_list:
                print(ground_truth)
                print('Before Load Data')
                print(split_date)
                new_time_series = copy(time_series)
                spot_list = np.array(new_time_series[ground_truth])
                new_time_series['spot_price'] = spot_list
                #extract copy of data to process
                ts = new_time_series.loc[split_date[0]:split_date[-1]]
                tvt_date = split_date[1:-1]

                #load data for use
                X_tr, y_tr, X_va, y_va, val_dates, column_lag_list = gn.prepare_data(ts,LME_dates,horizon,[ground_truth],lag,copy(tvt_date),version_params,metal_id_bool = metal_id,reshape = False)
                
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

            input_dim = final_X_tr.shape[-1]
            window_size = lag
            case_number = len(ground_truth_list)
            start = time.time()
            trainer = Trainer(input_dim, hidden_state, window_size, lr,
                                dropout, case_number, attention_size,
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
            print("the split date is {}".format(split_date[1]))
            trainer.train_minibatch(num_epochs, batch_size, interval)

