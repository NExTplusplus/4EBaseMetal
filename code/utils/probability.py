import argparse
from datetime import datetime
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tkinter
import time
import joblib
from copy import copy
import statistics
sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],"..")))
from data.load_data_v5 import load_data_v5
from utils.transform_data import flatten

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
    parser.add_argument(
        '-v','--version', help='version', type = int, default = 1
    )
    args = parser.parse_args()
    if args.ground_truth =='None':
        args.ground_truth = None
    
    gt = ""

    if args.ground_truth == "LMCADY":
        gt = "Co"
    if args.ground_truth == "LMAHDY":
        gt = "Al"
    if args.ground_truth == "LMPBDY":
        gt = "Le"
    if args.ground_truth == "LMZSDY":
        gt = "Zi"
    if args.ground_truth == "LMSNDY":
        gt = "Ti"
    if args.ground_truth == "LMNIDY":
        gt = "Ni"
    if args.ground_truth == "All":
        gt = "All"

    directory = os.listdir(os.path.abspath(os.path.join(sys.path[0],"exp",str(args.steps)+"d",gt,"logistic_regression","v"+str(args.version))))

    tra_date = '2003-11-12'
    val_date = '2016-06-01'
    tes_date = '2016-12-23'
    split_dates = [tra_date, val_date, tes_date]
    len_ma = 5
    len_update = 30
    tol = 1e-7


    with open(os.path.join(sys.path[0],"exp",str(args.steps)+"d",gt,"logistic_regression","v"+str(args.version),args.ground_truth+"_v"+str(args.version)+".conf")) as fin:
        fname_columns = json.load(fin)[0]

    for n in range(len(directory)):
        print(n)
        if "n"+str(n+1)+".joblib" not in directory:
            continue
        model = joblib.load(os.path.abspath(os.path.join(sys.path[0],"exp",str(args.steps)+"d",gt,"logistic_regression","v"+str(args.version),"n"+str(n+1)+".joblib")))
        with open(os.path.abspath(os.path.join(sys.path[0],"Results",args.ground_truth+"_h"+str(args.steps)+"_v"+str(args.version)+".csv"))) as f:
            lines = f.readlines()
            rel_line = lines[n+1].split(",")
            lag = int(rel_line[1])
            norm_volume = rel_line[2]
            norm_3m_spread = rel_line[3]
            norm_ex = rel_line[4]


            norm_params = {'vol_norm':norm_volume, 'ex_spread_norm':norm_ex,'spot_spread_norm': norm_3m_spread, 
                            'len_ma':5, 'len_update':30, 'both':3,'strength':0.01
                            }
            tech_params = {'strength':0.01,'both':3}
            # start_time = time.time()
            # load data
            X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params = load_data_v5(fname_columns, args.steps, ["LME_"+gt+"_Spot"], lag, 
                                                                            "NExT", split_dates, 
                                                                            norm_params, tech_params)
            for ind in range(len(X_tr)):
                neg_y_tr = y_tr[ind] - 1
                y_tr[ind] = y_tr[ind] + neg_y_tr
                X_tr[ind] = flatten(X_tr[ind])
                
                
                neg_y_va = y_va[ind] - 1
                y_va[ind] = y_va[ind] + neg_y_va
                X_va[ind] = flatten(X_va[ind])
                
                if X_te is not None:
                    neg_y_te = y_te[ind] - 1
                    y_te[ind] = y_te[ind] + neg_y_te
                    X_te[ind] = flatten(X_te[ind])
            # print(X_tr[0])              
            X_tr = np.concatenate(X_tr)
            y_tr = np.concatenate(y_tr).flatten()
            X_va = np.concatenate(X_va)
            y_va = np.concatenate(y_va).flatten()

            # print(model.predict_proba(X_tr))

            with open(os.path.abspath(os.path.join(sys.path[0],"Results",args.ground_truth+"_h"+str(args.steps)+"_v"+str(args.version)+"_n"+str(n+1)+"_probs.csv")),"w") as out:
                out.write("Negative Prob,Positive Prob,Prob Diff,Pred,TrueVal\n")
                prob = model.predict_proba(X_va)
                prob_diff = [abs(p[1]-p[0]) for p in prob]
                pred = model.predict(X_va)
                med = statistics.median(prob_diff)
                top_pred = []
                top_true = []
                bot_pred = []
                bot_true = []
                for i in range(len(prob)):
                    out.write(str(prob[i][0])+","+str(prob[i][1])+","+str(prob_diff[i])+","+str(pred[i])+","+str(y_va[i])+"\n")
                    if (prob_diff[i] > med):
                        top_pred.append(pred[i])
                        top_true.append(y_va[i])
                    else:
                        bot_pred.append(pred[i])
                        bot_true.append(y_va[i])
                top_pred = np.array(top_pred)
                top_true = np.array(top_true)
                bot_pred = np.array(bot_pred)
                bot_true = np.array(bot_true)


                out.write("\n\n\n")
                no_true = sum(np.equal(pred,y_va))
                no_TT = sum(np.multiply(pred + 1,y_va + 1))/4
                no_FF = sum(np.multiply(pred - 1,y_va - 1))/4
                no_TF = -sum(np.multiply(pred + 1,y_va - 1))/4
                no_FT = -sum(np.multiply(pred - 1,y_va + 1))/4
                top_no_true = sum(np.equal(top_pred,top_true))
                top_no_TT = sum(np.multiply(top_pred + 1,top_true + 1))/4
                top_no_FF = sum(np.multiply(top_pred - 1,top_true - 1))/4
                top_no_TF = -sum(np.multiply(top_pred + 1,top_true - 1))/4
                top_no_FT = -sum(np.multiply(top_pred - 1,top_true + 1))/4
                bot_no_true = sum(np.equal(bot_pred,bot_true))
                bot_no_TT = sum(np.multiply(bot_pred + 1,bot_true + 1))/4
                bot_no_FF = sum(np.multiply(bot_pred - 1,bot_true - 1))/4
                bot_no_TF = -sum(np.multiply(bot_pred + 1,bot_true - 1))/4
                bot_no_FT = -sum(np.multiply(bot_pred - 1,bot_true + 1))/4
                out.write(",Total,Top,Bot\n")
                out.write(",".join(["Acc",str(no_true/len(pred)),str(top_no_true/len(top_pred)),str(bot_no_true/len(bot_pred))]))
                out.write("\n")
                out.write(",".join(["TT",str(no_TT),str(top_no_TT),str(bot_no_TT)]))
                out.write("\n")
                out.write(",".join(["TF",str(no_TF),str(top_no_TF),str(bot_no_TF)]))
                out.write("\n")
                out.write(",".join(["FT",str(no_FT),str(top_no_FT),str(bot_no_FT)]))
                out.write("\n")
                out.write(",".join(["FF",str(no_FF),str(top_no_FF),str(bot_no_FF)]))

                out.close()


