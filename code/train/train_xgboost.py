#encoding:utf-8
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],'..')))
import time
import argparse
import json
from data.load_rnn import load_pure_log_reg
from model.logistic_regression import LogReg
from utils.log_reg_functions import objective_function, loss_function
import xgboost as xgb

if __name__ == '__main__':
    desc = 'the Xgboost model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--data_configure_file', '-c', type=str,
        help='configure file of the features to be read',
        default='exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
    )
    parser.add_argument('-s','--steps',type=int,default=3,
                        help='steps in the future to be predicted')
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                         type=str, default="LME_Co_Spot")
    parser.add_argument('-max_iter','--max_iter',type=int,default=100,
                        help='max number of iterations')
    parser.add_argument(
        '-min', '--model_path', help='path to load model',
        type=str, default='../../exp/3d/Co/Xgboost/'
    )
    parser.add_argument(
        '-v','--version', help='version', type = int, default = 5
    )
    parser.add_argument(
        '-mout', '--model_save_path', type=str, help='path to save model',
        default='../../exp/3d/Co/Xgboost/'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="exp/3d/Co/Xgboost/v5/result.txt")
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    args = parser.parse_args()
    if args.ground_truth =='None':
        args.ground_truth = None

    tra_date = '2005-01-10'
    val_date = '2016-06-01'
    tes_date = '2016-12-16'
    split_dates = [tra_date, val_date, tes_date]

    # read data configure file
    with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)

    with open(args.output,"w") as out:
        out.write("C,Lag,Volume,Spread,Exchange,Validation,Training,Testing,\n")
        if args.action == 'train':
            comparison = None
            n = 0
            for f in fname_columns:
                horizon = args.steps
                # best_C  = 0
                # model = None
                # best_lag = 0
                # max_acc = 0.0
                # te_acc = 0.0
                # best_nv = ""
                # best_ns = ""
                # best_ne = ""
                for lag in [5,10, 20, 30]:
                    for norm_volume in ["v1","v2","v4"]:
                            n+=1
                            norm_3m_spread = "v1"
                            norm_ex = "v2"
                            len_ma = 5
                            len_update = 30
                            tol = 1e-7
                            # start_time = time.time()
                            # load data
                            X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params= load_pure_log_reg(
                                f, 'log_1d_return', split_dates, gt_column = args.ground_truth, T = lag, S = horizon,
                                vol_norm = norm_volume, ex_spread_norm = norm_ex, spot_spread_norm = norm_3m_spread,
                                len_ma = len_ma, len_update = len_update, version = args.version
                            )
                            # print(X_tr[0])
                            #print("fname columns is {}".format(f))            
                            X_tr = np.concatenate(X_tr)
                            y_tr = np.concatenate(y_tr)
                            X_va = np.concatenate(X_va)
                            y_va = np.concatenate(y_va)
                            #print(len(X_tr[0]))
                            #pure_LogReg = LogReg(parameters={})
                            #max_iter = args.max_iter
                            #parameters = {"penalty":"l2", "C":C, "solver":"lbfgs", "tol":tol,"max_iter":6*4*len(f)*max_iter, "verbose" : 0,"warm_start": False, "n_job": -1}
                            #pure_LogReg.train(X_tr,y_tr.flatten(), parameters)
                            #n_iter = pure_LogReg.n_iter()
                            train = np.append(X_tr,y_tr,axis=1)
                            test = np.append(X_va,y_va,axis=1)
                            #print(len(train[0]))
                            train_X = train[:,:len(train[0])-2]
                            train_Y = train[:,len(train[0])-1]
                            for i in range(len(train_Y)):
                                if train_Y[i]==-1:
                                    train_Y[i]=0
                            test_X = test[:,:len(train[0])-2]
                            test_Y = test[:,len(train[0])-1]
                            for i in range(len(test_Y)):
                                if test_Y[i]==-1:
                                    test_Y[i]=0
                            xg_train = xgb.DMatrix(train_X, label=train_Y)
                            xg_test = xgb.DMatrix(test_X,label=test_Y)
                            param={}
                            param['objective']='multi:softmax'
                            for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                                param['eta']=i
                            param['max_depth']=20
                            param['silent']=1
                            param['nthread']=4
                            param['num_class']=2
                            for j in [3,4,5]:
                                param['min_child_weight']=j
                            for k in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]: 
                                param['gamma']=k
                            watchlist = [(xg_train,'train'),(xg_test,'test')]
                            num_round=10
                            bst = xgb.train(param, xg_train, num_round, watchlist)
                            pred = bst.predict(xg_test)
                            error_rate = np.sum(pred != test_Y)/test_Y.shape[0]
                            print('Test error using softmax={}'.format(error_rate))
                            print("lag is {}".format(lag))
                            print("C is {}".format(C))
                            print('norm_volume is {}'.format(norm_volume))
                            # steps = np.zeros(20)
                            # step = int(n_iter/20)
                            # tr_obj = np.zeros(20)
                            # va_obj = np.zeros(20)
                            # te_obj = np.zeros(20)
                            # tr_loss = np.zeros(20)
                            # va_loss = np.zeros(20)
                            # te_loss = np.zeros(20)
                            # total = 0
                            # for j in range(20):
                            #     steps[j] = step*(j+1)
                            #     if j > 0:
                            #         parameters["max_iter"] = int(np.round(steps[j]) - np.round(steps[j-1]))
                            #     else:
                            #         parameters["max_iter"] = int(np.round(steps[j]))
                            #     total += parameters["max_iter"]
                            #     parameters["max_iter"] = total
                            #     pure_LogReg.train(X_tr,y_tr.flatten(), parameters)


                            #     tr_loss[j] = loss_function(pure_LogReg,X_tr,y_tr)/np.shape(X_tr)[0]
                            #     va_loss[j] = loss_function(pure_LogReg,X_va,y_va)/np.shape(X_va)[0]
                            #     te_loss[j] = pure_LogReg.log_loss(X_te,y_te)
                            #     tr_obj[j] = objective_function(pure_LogReg,X_tr,y_tr,C)
                            #     va_obj[j] = objective_function(pure_LogReg,X_va,y_va,C)
                            #     te_obj[j] = objective_function(pure_LogReg,X_te,y_te,C)
                            # warnings.filterwarnings("once")
                            
                            # plt.plot(steps,tr_loss,"blue")
                            # plt.plot(steps,va_loss,"red")

                            # plt.plot(steps,te_loss,"green")
                            # plt.title("Loss")
                            # plt.savefig(os.path.join("Experiment","Loss","f_"+str(horizon)+" l_"+str(lag)+" t_"+str(tol)+" c_"+str(C)+" e_"+norm_ex+" 3m_"+norm_3m_spread+" n_"+str(n)+".png"))
                            # plt.close()
                            # plt.plot(steps,tr_obj, "blue")
                            # if max(tr_obj) - min(tr_obj) > 0.5:
                            #     plt.ylim(np.floor(min(tr_obj)),np.ceil(max(tr_obj)))
                            # plt.plot(steps,va_obj,"red")
                            # plt.plot(steps,te_obj,"green")
                            # plt.title("Objective Function")
                            # plt.savefig(os.path.join("Experiment","Objective Function","f_"+str(horizon)+" l_"+str(lag)+" t_"+str(tol)+" c_"+str(C)+" e_"+norm_ex+" 3m_"+norm_3m_spread+" n_"+str(n)+".png"))
                            # plt.close()

                            #acc = pure_LogReg.test(X_va,y_va.flatten())
                            #if args.ground_truth is not None:
                                #pure_LogReg.save(os.path.join(sys.path[0],args.model_save_path,"exp",str(horizon)+"d",args.ground_truth[4:6],"logistic_regression","v"+str(args.version),"n"+str(n)+".joblib"))
                            #else:
                                #pure_LogReg.save(os.path.join(sys.path[0],args.model_save_path,"exp",str(horizon)+"d","All","logistic_regression","v"+str(args.version),"n"+str(n)+".joblib"))
                            #out.write(str(C)+",")
                            #out.write(str(lag)+",")
                            #out.write(str(norm_volume)+",")
                            #out.write(str(norm_3m_spread)+",")
                            #out.write(str(norm_ex)+",")
                            #out.write(str(acc)+",")
                            #out.write(str(pure_LogReg.test(X_tr,y_tr.flatten()))+",")
                            # if X_te is not None:
                            #     X_te = np.concatenate(X_te)
                            #     y_te = np.concatenate(y_te)
                            #     out.write(str(pure_LogReg.test(X_te,y_te.flatten()))+",")
                            #out.write("\n")
                            #if norm_params["nVol"] is False:
                                #break  

                #     break
                # break

        out.close()
