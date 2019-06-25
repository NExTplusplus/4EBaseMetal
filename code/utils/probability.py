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
from data.load_rnn import load_pure_log_reg

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

    directory = os.listdir(os.path.abspath(os.path.join(sys.path[0],"..","..","mingwei","NExT","4EBaseMetal","exp",str(args.steps)+"d",gt,"logistic_regression","v"+str(args.version))))

    tra_date = '2005-01-10'
    val_date = '2016-06-01'
    tes_date = '2016-12-16'
    split_dates = [tra_date, val_date, tes_date]
    len_ma = 5
    len_update = 30
    tol = 1e-7


    with open(os.path.join(sys.path[0],"..","..","mingwei","NExT","4EBaseMetal","exp",str(args.steps)+"d",gt,"logistic_regression","v"+str(args.version),args.ground_truth+"_v"+str(args.version)+".conf")) as fin:
        fname_columns = json.load(fin)[0]

    for n in range(60):
        print(n)
        if "n"+str(n+1)+".joblib" not in directory:
            continue
        model = joblib.load(os.path.abspath(os.path.join(sys.path[0],"..","..","mingwei","NExT","4EBaseMetal","exp",str(args.steps)+"d",gt,"logistic_regression","v"+str(args.version),"n"+str(n+1)+".joblib")))
        with open(os.path.abspath(os.path.join(sys.path[0],"..","..","mingwei","NExT","Results",args.ground_truth+"_h"+str(args.steps)+"_v"+str(args.version)+".csv"))) as f:
            lines = f.readlines()
            rel_line = lines[n+1].split(",")
            lag = int(rel_line[1])
            norm_volume = rel_line[2]
            norm_3m_spread = rel_line[3]
            norm_ex = rel_line[4]

            X_tr, y_tr, X_va, y_va, X_te, y_te,norm_params= load_pure_log_reg(
                fname_columns, 'log_1d_return', split_dates, gt_column = "LME_"+gt+"_Spot", T = lag, S = args.steps,
                vol_norm = norm_volume, ex_spread_norm = norm_ex, spot_spread_norm = norm_3m_spread,
                len_ma = 5, len_update = 30, version = args.version
            )

            X_va = np.concatenate(X_va)
            y_va = np.concatenate(y_va).flatten()

            # print(model.predict_proba(X_tr))

            with open(os.path.abspath(os.path.join(sys.path[0],"..","..","mingwei","NExT","Results",args.ground_truth+"_h"+str(args.steps)+"_v"+str(args.version)+"_n"+str(n+1)+"_probs.csv")),"w") as out:
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
                out.write(",".join(["Acc",str(no_true/len(pred)),str(top_no_true/len(top_pred)),str(bot_no_true/len(top_pred))]))
                out.write("\n")
                out.write(",".join(["TT",str(no_TT),str(top_no_TT),str(bot_no_TT)]))
                out.write("\n")
                out.write(",".join(["TF",str(no_TF),str(top_no_TF),str(bot_no_TF)]))
                out.write("\n")
                out.write(",".join(["FT",str(no_FT),str(top_no_FT),str(bot_no_FT)]))
                out.write("\n")
                out.write(",".join(["FF",str(no_FF),str(top_no_FF),str(bot_no_FF)]))

                out.close()


