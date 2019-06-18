import argparse
from datetime import datetime
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tkinter
import time
from joblib import dump
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
    if args.ground_truth =='None':
        args.ground_truth = None

    tra_date = '2005-01-10'
    val_date = '2015-01-05'
    tes_date = '2016-01-06'
    split_dates = [tra_date, val_date, tes_date]

    # read data configure file
    with open(args.data_configure_file) as fin:
        fname_columns = json.load(fin)

    with open(args.output+".csv","w") as out:
        out.write("C,Lag,Volume,Spread,Exchange,Validation,Training,Testing,\n")
        if args.action == 'train':
            comparison = None
            n = 0
            for f in fname_columns:
                n+=1
                print(f)
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
                    for C in (0.0001,1,1000,1000000):
                        for norm_ex in ("v1","v2"):
                            for norm_volume in ("v1","v2","v3","v4"):
                                norm_3m_spread = "v1"
                                len_ma = 5
                                len_update = 30
                                tol = 1e-7
                                # start_time = time.time()
                                # load data
                                X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params= load_pure_log_reg(
                                    f, 'log_1d_return', split_dates, gt_column = args.ground_truth, T = lag,
                                    S = horizon,
                                    vol_norm = norm_volume, ex_spread_norm = norm_ex, spot_spread_norm = norm_3m_spread
                                )
                                # print(X_tr[0])                  
                                X_tr= np.concatenate(X_tr)
                                y_tr = np.concatenate(y_tr)
                                X_va = np.concatenate(X_va)
                                y_va = np.concatenate(y_va)
                                X_te = np.concatenate(X_te)
                                y_te = np.concatenate(y_te)
                                pure_LogReg = LogReg(parameters={})
                                # check = True
                                max_iter = args.max_iter
                                parameters = {"penalty":"l2", "C":C, "solver":"lbfgs", "tol":tol,"max_iter":6*4*len(f)*max_iter, "verbose" : 0,"warm_start": False}
                                pure_LogReg.train(X_tr,y_tr.flatten(), parameters)
                                n_iter = pure_LogReg.n_iter()
                                

                                steps = np.zeros(20)
                                step = int(n_iter/20)
                                tr_obj = np.zeros(20)
                                va_obj = np.zeros(20)
                                te_obj = np.zeros(20)
                                tr_loss = np.zeros(20)
                                va_loss = np.zeros(20)
                                te_loss = np.zeros(20)
                                total = 0
                                for j in range(20):
                                    steps[j] = step*(j+1)
                                    if j > 0:
                                        parameters["max_iter"] = int(np.round(steps[j]) - np.round(steps[j-1]))
                                    else:
                                        parameters["max_iter"] = int(np.round(steps[j]))
                                    total += parameters["max_iter"]
                                    parameters["max_iter"] = total
                                    pure_LogReg.train(X_tr,y_tr.flatten(), parameters)


                                    tr_loss[j] = loss_function(pure_LogReg,X_tr,y_tr)/np.shape(X_tr)[0]
                                    va_loss[j] = loss_function(pure_LogReg,X_va,y_va)/np.shape(X_va)[0]
                                    te_loss[j] = pure_LogReg.log_loss(X_te,y_te)
                                    tr_obj[j] = objective_function(pure_LogReg,X_tr,y_tr,C)
                                    va_obj[j] = objective_function(pure_LogReg,X_va,y_va,C)
                                    te_obj[j] = objective_function(pure_LogReg,X_te,y_te,C)
                                warnings.filterwarnings("once")
                                acc = pure_LogReg.test(X_va,y_va.flatten())
                                plt.plot(steps,tr_loss,"blue")
                                plt.plot(steps,va_loss,"red")

                                plt.plot(steps,te_loss,"green")
                                plt.title("Loss")
                                plt.savefig(os.path.join("Experiment","Loss","f_"+str(horizon)+" l_"+str(lag)+" t_"+str(tol)+" c_"+str(C)+" e_"+norm_ex+" 3m_"+norm_3m_spread+" n_"+str(n)+".png"))
                                plt.close()
                                plt.plot(steps,tr_obj, "blue")
                                if max(tr_obj) - min(tr_obj) > 0.5:
                                    plt.ylim(np.floor(min(tr_obj)),np.ceil(max(tr_obj)))
                                plt.plot(steps,va_obj,"red")
                                plt.plot(steps,te_obj,"green")
                                plt.title("Objective Function")
                                plt.savefig(os.path.join("Experiment","Objective Function","f_"+str(horizon)+" l_"+str(lag)+" t_"+str(tol)+" c_"+str(C)+" e_"+norm_ex+" 3m_"+norm_3m_spread+" n_"+str(n)+".png"))
                                plt.close()
                                if acc > max_acc:
                                    best_C = C
                                    best_lag = lag
                                    model = pure_LogReg
                                    max_acc = acc
                                    tr_acc = pure_LogReg.test(X_tr,y_tr.flatten())
                                    te_acc = pure_LogReg.test(X_te,y_te.flatten())
                                    best_nv = norm_volume
                                    best_ns = norm_3m_spread
                                    best_ne = norm_ex
                                if norm_params["nVol"] is False:
                                    break
                            if norm_params["nEx"] is False:
                                break       

                #         break
                #     break
                # break
                X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params= load_pure_log_reg(
                    f, 'log_1d_return', split_dates,gt_column = args.ground_truth, T = best_lag,
                    S = horizon,
                    vol_norm = best_nv, ex_spread_norm = best_ne, spot_spread_norm = best_ns
                )
                X_tr = np.concatenate(X_tr)
                y_tr = np.concatenate(y_tr)
                X_va = np.concatenate(X_va)
                y_va = np.concatenate(y_va)
                X_te = np.concatenate(X_te)
                y_te = np.concatenate(y_te)
                if args.ground_truth is None:
                    args.ground_truth = "None"
                model.save("4EBaseMetal/exp/log_reg/"+args.ground_truth+"_h"+str(args.steps)+"_n"+str(n)+".joblib")
                out.write(str(best_C)+",")
                out.write(str(best_lag)+",")
                out.write(str(best_nv)+",")
                out.write(str(best_ns)+",")
                out.write(str(best_ne)+",")
                out.write(str(max_acc)+",")
                out.write(str(tr_acc)+",")
                out.write(str(te_acc)+",")
                prediction = model.predict(X_va).reshape(X_va.shape[0],1)
                total_no = prediction.shape[0]
                # with open("h.csv","w") as out1:
                #     out1.write("Prediction,True Value\n")
                #     for i in range(X_te.shape[0]):
                #         # outputs a boolean value stating whether the prediction was an increase in spot price.
                #         out1.write(str(prediction[i] == 1).strip("[ ").strip("]") + ","+ str(y_te[i] == 1).strip("[ ").strip("]") + "\n")
                #     out1.close()
                # print(X_te[0:10])
                
                no_true = sum(np.equal(prediction,y_va))
                no_TT = sum(np.multiply(prediction + 1,y_va + 1))/4
                no_FF = sum(np.multiply(prediction - 1,y_va - 1))/4
                no_TF = -sum(np.multiply(prediction + 1,y_va - 1))/4
                no_FT = -sum(np.multiply(prediction - 1,y_va + 1))/4
                
                
                print("Overall Accuracy:%d",no_true/total_no )
                print("TT:%d", no_TT)
                print("TF:%d", no_TF)
                print("FT:%d", no_FT)
                print("FF:%d", no_FF)
                if comparison == None:
                    comparison = max_acc
                else:
                    if max_acc > comparison:
                        out.write("TRUE")
                    else:
                        out.write("FALSE")
                out.write("\n")
                    
        out.close()
        

                



