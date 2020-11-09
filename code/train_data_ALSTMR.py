import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
import warnings
from live.ALSTMR_live import ALSTMR_online

if __name__ == '__main__':
    desc = 'the ALSTM Regression model'
    parser = argparse.ArgumentParser(description=desc)

    #script parameter
    parser.add_argument(
        '-o', '--action', type=str, default='train', help='action that we wish to take, has potential values of : train, test, tune'
        )

    #results parameter
    parser.add_argument(
      '-s', '--horizon', type=int, default=5, help='the prediction horizon'
        )
    parser.add_argument(
      '-gt', '--ground_truth', help='the name of the column that we are predicting either value or direction', type=str, default="LME_Al_Spot"
        )
    parser.add_argument(
      '-sou', '--source', help='source of data', type = str, default = "NExT"
        )
    parser.add_argument(
        '-v', '--version', help='feature version for data', type = str, default = 'v10'
        )
    parser.add_argument(
        '-d', '--date', help = "string of comma-separated dates which identify the total period of deployment by half-years", type=str
        )	
    parser.add_argument(
      '-method', '--method', type=str, default="best_acc", help='method to choose hyperparameter'
        )

    #parameters for input or output of tuning
    parser.add_argument(
        '-script', '--script', default='code/train/train_alstm_reg.py', help = 'script that generates the tuning logs of a single hyperparameter combination', type=str
        )
    parser.add_argument(
        '-log', '--log', help = 'output file for logs', default='./tune.log', type=str
        )

    #hyperparameters
    parser.add_argument(
      '-max_iter', '--max_iter', type=int, default=100, help='max number of iterations'
        )
    parser.add_argument(
      '-l', '--lag', type=int, default = 5, help='lag'
        )
    parser.add_argument(
      '-e', '--epoch', type=int, default=50, help='the number of epochs'
        )
    parser.add_argument(
      '-b', '--batch', type=int, default=512, help='the mini-batch size'
        )
    parser.add_argument(
      '-i', '--interval', type=int, default=1, help='save models every interval epoch'
        )
    parser.add_argument(
      '-lrate', '--lrate', type=float, default=0.001, help='learning rate'
        )
    parser.add_argument(
      '-t', '--test', action='store_true', help='train or test'
        )
    parser.add_argument(
      '-m', '--model', type=str, default='', help='the model name(after encoder/decoder)'
        )
    parser.add_argument(
      '-hidden', '--hidden_state', type=int, default=50, help='number of hidden_state of encoder/decoder'
        )
    parser.add_argument(
      '-split', '--split', type=float, default=0.9, help='the split ratio of validation set'
        )
    parser.add_argument(
      '-drop', '--drop_out', type=float, default = 0.0, help='the dropout rate of LSTM network'
        )
    parser.add_argument(
      '-a', '--attention_size', type = int, default = 2, help='the head number in MultiheadAttention Mechanism'
        )
    parser.add_argument(
      '-embed', '--embedding_size', type=int, default = 5, help='the size of embedding layer'
        )
    parser.add_argument(
      '-lambd', '--lambd', type=float, default = 0, help='the weight of classfication loss'
        )
    parser.add_argument(
      '-savep', '--save_prediction', type=bool, default=0, help='whether to save prediction results'
        )
    parser.add_argument(
      '-savel', '--save_loss', type=bool, default=0, help='whether to save loss results'
        )
    parser.add_argument(
      '--drop_out_mc', type = float, default= 0.0, help = 'dropout rate of monte-carlo'
    )
    parser.add_argument(
      '--repeat_mc', type = int, default= 10, help = 'number of monte-carlo repetitions'
    )
    parser.add_argument(
      '--mc', type = int, default= 0, help = 'trigger for monte-carlo, if value not equal to 0 then monte-carlo is activated'
    )
    args = parser.parse_args()
    args.mc = args.mc != 0

    #initialize model
    model = ALSTMR_online(lag = args.lag, horizon = args.horizon, version = args.version, gt = args.ground_truth, date = args.date, source = args.source, mc = args.mc)
    
    #case if action is tune
    if args.action=="tune":
        model.tune(log = args.log, script = args.script, drop_out = args.drop_out, hidden = args.hidden_state, \
                                embedding_size = args.embedding_size, batch = args.batch, drop_out_mc = args.drop_out_mc, \
                                repeat_mc = args.repeat_mc    
                                )
    
    #case if action is train
    elif args.action=='train':
        model.train(
            num_epochs=args.epoch, 
            batch_size=args.batch, 
            split=args.split, 
            drop_out=args.drop_out, 
            drop_out_mc = args.drop_out_mc, 
            repeat_mc = args.repeat_mc, 
            hidden_state=args.hidden_state, 
            embedding_size=args.embedding_size, 
            lrate=args.lrate, 
            attention_size=args.attention_size, 
            interval=args.interval, 
            lambd=args.lambd, 
            save_loss=args.save_loss, 
            save_prediction=args.save_prediction, 
            method = args.method)

    #case if action is test
    else:
        final = model.test(num_epochs=args.epoch, 
            batch_size=args.batch, 
            split=args.split, 
            drop_out=args.drop_out, 
            drop_out_mc = args.drop_out_mc, 
            repeat_mc = args.repeat_mc, 
            hidden_state=args.hidden_state, 
            embedding_size=args.embedding_size, 
            lrate=args.lrate, 
            attention_size=args.attention_size, 
            interval=args.interval, 
            lambd=args.lambd, 
            save_loss=args.save_loss, 
            save_prediction=args.save_prediction, 
            method = args.method)
        print("the result of the test is {}".format(final))
