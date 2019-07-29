'''
    
'''
import os
import sys


import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))


from data.load_data_v5_ex2 import load_data_v5_ex2
from model.logistic_regression import LogReg
from utils.transform_data import flatten
from utils.log_reg_functions import objective_function, loss_function


import warnings


if __name__ == '__main__':
    desc = 'the logistic regression model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--data_configure_file', '-c', type=str,
        help='configure file of the features to be read',
        default='log_reg_data.conf'
    )
    parser.add_argument('-s','--steps',type=int,default=1,
                        help='steps in the future to be predicted')
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                        type=str, default="LMCADY")
    parser.add_argument('-max_iter','--max_iter',type=int,default=100,
                        help='max number of iterations')
    parser.add_argument(
        '-min', '--model_path', help='path to load model',
        type=str, default='../../exp/log_reg/model'
    )
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "NExT"
    )
    parser.add_argument(
        '-v','--version', help='version', type = int, default = 1
    )
    parser.add_argument(
        '-mout', '--model_save_path', type=str, help='path to save model',
        default='../../exp/log_reg/model'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    args = parser.parse_args()
    if args.ground_truth =='None':
        args.ground_truth = None

    tra_date = '2003-11-12'
    val_date = '2016-01-04'
    tes_date = '2016-12-23'
    split_dates = [tra_date, val_date, tes_date]

    os.chdir(os.path.abspath(sys.path[0]))

    # read data configure file
    with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)

    with open(args.output,"w") as out:
        out.write("C,Lag,Volume,Spread,Exchange,Validation,Training,Testing,\n")
        args.ground_truth = args.ground_truth.split(",")
        if args.action == 'train':
            comparison = None
            n = 0
            for f in fname_columns:
                horizon = args.steps
                for lag in [5,10,20,30]:
                  for norm_volume in ["v1","v2"]:
                    norm_3m_spread = "v1"
                    norm_ex = "v1"
                    len_ma = 5
                    len_update = 30
                    tol = 1e-7

                    norm_params = {'vol_norm':norm_volume, 'ex_spread_norm':norm_ex,'spot_spread_norm': norm_3m_spread, 
                                    'len_ma':len_ma, 'len_update':len_update, 'both':3,'strength':0.01
                                    }
                    tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                                   'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
                    # start_time = time.time()
                    # load data
                    X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params = load_data_v5_ex2(f, horizon, args.ground_truth, lag, 
                                                                                    args.source, split_dates, 
                                                                                    norm_params, tech_params)
                    for ind in range(len(X_tr)):
                        X_tr[ind] = flatten(X_tr[ind])
                        X_va[ind] = flatten(X_va[ind])
                        
                        if X_te is not None:
                            X_te[ind] = flatten(X_te[ind])
                    # print(X_tr[0])              
                    X_tr = np.concatenate(X_tr)
                    y_tr = np.concatenate(y_tr)
                    X_va = np.concatenate(X_va)
                    y_va = np.concatenate(y_va)
                    pure_LogReg = LogReg(parameters={})
                    for C in [0.00001,0.0001,0.001,0.01]:
                    
                        n+=1

                        max_iter = args.max_iter
                        parameters = {"penalty":"l2", "C":C, "solver":"lbfgs", "tol":tol,"max_iter":6*4*len(f)*max_iter, "verbose" : 0,"warm_start": False, "n_jobs": -1}
                        pure_LogReg.train(X_tr,y_tr.flatten(), parameters)
                        n_iter = pure_LogReg.n_iter()
                        

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

                        acc = pure_LogReg.test(X_va,y_va.flatten())
                        if len(args.ground_truth) == 1:
                            pure_LogReg.save(os.path.join(sys.path[0],args.model_save_path,"exp",str(horizon)+"d",args.ground_truth[0][4:6],"logistic_regression_ex2","v"+str(args.version),"n"+str(n)+".joblib"))
                        else:
                            pure_LogReg.save(os.path.join(sys.path[0],args.model_save_path,"exp",str(horizon)+"d","All","logistic_regression_ex2","v"+str(args.version),"n"+str(n)+".joblib"))
                        out.write(str(C)+",")
                        out.write(str(lag)+",")
                        out.write(str(norm_volume)+",")
                        out.write(str(norm_3m_spread)+",")
                        out.write(str(norm_ex)+",")
                        out.write(str(acc)+",")
                        out.write(str(pure_LogReg.test(X_tr,y_tr.flatten()))+",")
                        # if X_te is not None:
                        #     X_te = np.concatenate(X_te)
                        #     y_te = np.concatenate(y_te)
                        #     out.write(str(pure_LogReg.test(X_te,y_te.flatten()))+",")
                        out.write("\n")
                    if norm_params["nVol"] is False:
                        break  

                #     break
                # break

        out.close()
        

                



