import os 
import sys
import numpy as np
import ast
import argparse

if __name__ == '__main__':
    desc = 'the script for ALSTM tuning'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-gt', '--ground_truth_list', help='list of ground truths, separated by ","',
                        type=str, default="LME_Co_Spot,LME_Al_Spot,LME_Ni_Spot,LME_Ti_Spot,LME_Zi_Spot,LME_Le_Spot")
    parser.add_argument(
        '-sou','--source', help='source of data to be inserted into commands', type = str, default = "4E"
    )
    parser.add_argument(
        '-f','--files', help='comma-separated files', type = str, 
        default = 'v16_h1_para.txt,v16_h3_para.txt,v16_h5_para.txt,v16_h10_para.txt,v16_h20_para.txt,v16_h60_para.txt,v26_h1_para.txt,v26_h3_para.txt,v26_h5_para.txt,v26_h10_para.txt,v26_h20_para.txt,v26_h60_para.txt'
    )
    parser.add_argument('-r','--regression', default = 0, type = int)
    
    parser.add_argument('-o','--action', default = 'train', type = str)
    parser.add_argument('-mc','--mc',type =int, default = 0)
    parser.add_argument('-d','--date',type = str, help = "dates", default = "2014-12-31,2015-06-30,2015-12-31,2016-06-30,2016-12-31,2017-06-30,2017-12-31,2018-06-30,2018-12-31")
    parser.add_argument('-m','--method',type = str, help = "method of extraction", default = "best_loss")

    args = parser.parse_args()
    args.mc = args.mc != 0
    files =args.files.split(",")
    best_loss_parameters = ""
    best_acc_parameters = ""
    average_acc_parameters = {}
    average_loss_parameters = {}
    train = "train_data_ALSTM"
    full_str = ""
    if args.regression != 0:
        train = "train_data_ALSTMR"

    #iterate through files
    for f in files:
        step = f.split("_")[1]
        version = f.split("/")[-1].split("_")[0]
        val_loss = 100.0
        val_acc = 0
        curr = ""
        print(f)

        #analyze file with give nstructure
        with open(f) as fl:
            lines = fl.readlines()
            for l,line in enumerate(lines):
                if "{'drop_out'" == line[:11]: 
                    if val_loss > float(lines[l+1].split(" ")[-1]):
                        val_loss = float(lines[l+1].split(" ")[-1])
                        best_loss_parameters = ast.literal_eval(line.strip())
                    if val_acc < float(lines[l+2].split(" ")[-1]):
                        val_acc = float(lines[l+2].split(" ")[-1])
                        best_acc_parameters = ast.literal_eval(line.strip())
                if "(" == line[0]:
                    if l+1 < len(lines):
                        curr = lines[l+1].strip()
                    if '(' == curr[0]:
                        key = line.split("'")[1]
                        val = float(line.split("[")[1].split(']')[0])
                        if val.is_integer():
                            val = int(val)
                        average_acc_parameters[key] = val
                        average_loss_parameters[key] = val
                if "[" == line[0] and curr != "":
                    va_loss = [float(i) for i in lines[l+1][10:-2].split(" ") if i != ""]
                    va_acc = [float(i) for i in lines[l+2][9:-2].split(" ") if i != ""]
                    va_loss = np.argmin(va_loss)
                    va_acc = np.argmax(va_acc)
                    param_combination = [float(i.strip()) if curr == "drop_out" or curr == "drop_out_mc" else int(i.strip()) for i in line[1:-2].split(",")]
                    average_acc_parameters[curr]=param_combination[va_acc]
                    average_loss_parameters[curr]=param_combination[va_loss]

            #choose method of choosing hpyerparameter
            if args.method == "best_loss":
                dc = best_loss_parameters
            elif args.method == "best_acc":
                dc = best_acc_parameters
            elif args.method == "ave_loss":
                dc = average_loss_parameters
            elif args.method == "ave_acc":
                dc = average_acc_parameters
            else:
                dc = None

            #generate tuning commands for alstmr with monte carlo
            if args.action == "tune":
                assert args.mc, "tune is only for monte carlo"
                full_str += ' '.join(["python code/"+train+".py","-a 2","-b",str(dc["batch"]), \
                                    '-drop',str(dc["drop_out"]),'-embed',str(dc['embedding_size']), \
                                    '-e 50','-hidden', str(dc['hidden']),'-i 1','-l',str(dc['lag']), \
                                    '-lambd 0','-lrate 0.001','-savel 0','-savep 0', '-split 0.9','-s', step[1:],'-v',version, \
                                    '-sou '+args.source,'-o '+args.action,'-d '+args.date,'--mc',str(args.mc*1),'-log', \
                                    "./result/validation/alstm/"+version+"_h"+step[1:]+"_mc_tune.log",'>',"/dev/null",'2>&1 &']) \
                                    +"\n"
            else:
                #genreate commands with action for alstm with monte carlo
                if args.mc:
                    if "drop_out_mc" not in dc.keys():
                        dc["drop_out_mc"] = 0.0
                        dc["repeat_mc"] = 10
                    full_str += ' '.join(["python code/"+train+".py","-a 2","-b",str(dc["batch"]), \
                                        '-drop',str(dc["drop_out"]),'-embed',str(dc["embedding_size"]), \
                                        '-e 50','-hidden', str(dc['hidden']),'-i 1','-l',str(dc["lag"]), \
                                        '--drop_out_mc',str(dc["drop_out_mc"]), \
                                        '--repeat_mc',str(dc["repeat_mc"]), \
                                        '-lambd 0','-lrate 0.001','-savel 0','-savep 0', '-split 0.9','-s', step[1:],'-v',version,'--mc', str(args.mc*1), \
                                        '-sou '+args.source,'-o '+args.action,'-d '+args.date,'-method '+args.method,'>',"/dev/null",'2>&1 &']) \
                                        +"\n"
                #genreate commands with action for alstm without monte carlo
                else:
                    full_str += ' '.join(["python code/"+train+".py","-a 2","-b",str(dc["batch"]), \
                                        '-drop',str(dc["drop_out"]),'-embed',str(dc["embedding_size"]), \
                                        '-e 50','-hidden', str(dc['hidden']),'-i 1','-l',str(dc["lag"]), \
                                        '-lambd 0','-lrate 0.001','-savel 0','-savep 0', '-split 0.9','-s', step[1:],'-v',version, \
                                        '-sou '+args.source,'-o '+args.action,'-d '+args.date,'-method '+args.method,'>',"/dev/null",'2>&1 &']) \
                                        +"\n"
            

    with open(args.method +"_"+args.action+".sh","w") as bl:
        bl.write("#!/bin/bash\n")
        bl.write(full_str)
        bl.close()
