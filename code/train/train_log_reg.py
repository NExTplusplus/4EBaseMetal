import argparse
from datetime import datetime
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tkinter
import time
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

from data.load_rnn import load_pure_log_reg
from model.logistic_regression import LogReg
from utils.log_reg_functions import objective_function, loss_function


import warnings


if __name__ == '__main__':
    desc = 'the logistic regression model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--data_configure_file', '-c', type=str,
        help='configure file of the features to be read',
        default='../../exp/log_reg_data.conf'
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
        '-mout', '--model_save_path', type=str, help='path to save model',
        default='../../exp/log_reg/model'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    args = parser.parse_args()



    tra_date = '2005-01-10'
    val_date = '2015-01-06'
    tes_date = '2016-01-05'
    split_dates = [tra_date, val_date, tes_date]

    # read data configure file
    with open(args.data_configure_file) as fin:
        fname_columns = json.load(fin)

    with open(args.output+".csv","w") as out:
        out.write("C,Lag,Volume,Spread,Exchange,Validation,Testing,\n")
        if args.action == 'train':
            comparison = None
            for f in fname_columns:
                horizon = args.steps
                best_C  = 0
                model = None
                best_lag = 0
                max_acc = 0.0
                te_acc = 0.0
                best_nv = ""
                best_ns = ""
                best_ne = ""
                for lag in (5,10,20,40):
                    for C in (0.01,0.1,1,10,100):
                        for norm_ex in ("v1","v2"):
                            for norm_volume in ("v1","v2","v3","v4"):
                                norm_3m_spread = "v1"
                                len_ma = 5
                                len_update = 30
                                tol = 1e-4
                                # start_time = time.time()
                                # load data
                                X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params= load_pure_log_reg(
                                    f, args.ground_truth, 'log_1d_return', split_dates, T = lag,
                                    S = horizon,
                                    vol_norm = norm_volume, ex_spread_norm = norm_ex, spot_spread_norm = norm_3m_spread,
                                    inc = True
                                )                  
                    

                                pure_LogReg = LogReg(parameters={})
                                # check = True
                                max_iter = args.max_iter
                                parameters = {"penalty":"l2", "C":C, "solver":"lbfgs", "tol":tol,"max_iter":4*len(f)*max_iter, "verbose" : 0,"warm_start":False}
                                pure_LogReg.train(X_tr,y_tr.flatten(), parameters)
                                n_iter = pure_LogReg.n_iter()
                                # if n_iter == max_iter:
                                #     max_iter += 100
                                # else:
                                #     check = False
                                # print(n_iter)
                                # step = n_iter/10.0
                                # print(n_iter)

                                # print(time.time()-start_time)
                                

                                # steps = np.zeros(10)
                                # tr_obj = np.zeros(10)
                                # va_obj = np.zeros(10)
                                # te_obj = np.zeros(10)
                                # tr_loss = np.zeros(10)
                                # va_loss = np.zeros(10)
                                # te_loss = np.zeros(10)
                                # total = 0
                                # warnings.filterwarnings("ignore")
                                # for j in range(10):
                                    
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
                                #     # te_loss[j] = pure_LogReg.log_loss(X_te,y_te)
                                #     tr_obj[j] = objective_function(pure_LogReg,X_tr,y_tr,C)
                                #     # va_obj[j] = objective_function(pure_LogReg,X_va,y_va,C)
                                #     # te_obj[j] = objective_function(pure_LogReg,X_te,y_te)
                                # warnings.filterwarnings("once")
                                acc = pure_LogReg.test(X_va,y_va.flatten())
                                # plt.plot(steps,tr_loss,"blue")
                                # plt.plot(steps,va_loss,"red")

                                # # plt.plot(steps,te_loss,"green")
                                # plt.title("Loss")
                                # plt.savefig(os.path.join("..","..","..","..","Experiment","C","Loss","Forecast "+str(horizon),"Lag "+str(lag),"tol "+str(tol)+" C "+str(C)+".png"))
                                # plt.close()
                                # plt.plot(steps,tr_obj, "blue")
                                # if max(tr_obj) - min(tr_obj) > 0.5:
                                #     plt.ylim(np.floor(min(tr_obj)),np.ceil(max(tr_obj)))
                                # # plt.plot(steps,va_obj,"red")
                                # # plt.plot(steps,te_obj,"green")
                                # plt.title("Objective Function")
                                # plt.savefig(os.path.join("..","..","..","..","Experiment","C","Objective Function","Forecast "+str(horizon),"Lag "+str(lag),"F"+str(horizon)+"L"+str(lag)+"C"+str(C)+".png"))
                                # plt.close()
                                if acc > max_acc:
                                    best_C = C
                                    best_lag = lag
                                    model = pure_LogReg
                                    max_acc = acc
                                    te_acc = pure_LogReg.test(X_te,y_te.flatten())
                                    best_nv = norm_volume
                                    best_ns = norm_3m_spread
                                    best_ne = norm_ex
                                t_acc = pure_LogReg.test(X_te,y_te.flatten())
                                # out.write(str(horizon)+",")
                                # out.write(str(C)+",")
                                # out.write(str(lag)+",")
                                # out.write(str(norm_volume)+",")
                                # out.write(str(norm_3m_spread)+",")
                                # out.write(str(norm_ex)+",")
                                # out.write(str(n_iter)+",")
                                # out.write(str(acc)+",")
                                # out.write(str(t_acc)+",\n")
                                if norm_params["nVol"] is False:
                                    break
                            if norm_params["nEx"] is False:
                                break       

                #         break
                #     break
                # break
                out.write(str(best_C)+",")
                out.write(str(best_lag)+",")
                out.write(str(best_nv)+",")
                out.write(str(best_ns)+",")
                out.write(str(best_ne)+",")
                out.write(str(max_acc)+",")
                out.write(str(te_acc)+",")
                if comparison == None:
                    comparison = max_acc
                else:
                    if max_acc > comparison:
                        out.write("TRUE")
                    else:
                        out.write("FALSE")
                out.write("\n")
                    
        out.close()
                    

                



