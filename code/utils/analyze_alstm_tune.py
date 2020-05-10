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
    parser.add_argument('-o','--action', default = 'train')
    parser.add_argument('-d','--date',type = str, help = "dates", default = "2014-12-31,2015-06-30,2015-12-31,2016-06-30,2016-12-31,2017-06-30,2017-12-31,2018-06-30,2018-12-31")

    args = parser.parse_args()
    files =args.files.split(",")
    best_loss_parameters = ""
    best_acc_parameters = ""
    average_acc_parameters = {'batch':512}
    average_loss_parameters = {'batch':512}
    best_loss_str = ""
    best_acc_str = ""
    average_loss_str = ""
    average_acc_str = ""
    for f in files:
        step = f.split("_")[1]
        version = f.split("_")[0]
        val_loss = 100.0
        val_acc = 0
        curr = ""
        print(f)
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
                    curr = lines[l+1].strip()
                if "[" == line[0] and curr != "":
                    va_loss = [float(i) for i in lines[l+1][10:-2].split(" ") if i != ""]
                    va_acc = [float(i) for i in lines[l+2][9:-2].split(" ") if i != ""]
                    va_loss = np.argmin(va_loss)
                    va_acc = np.argmax(va_acc)
                    param_combination = [float(i.strip()) if curr == "drop_out" else int(i.strip()) for i in line[1:-2].split(",")]
                    average_acc_parameters[curr]=param_combination[va_acc]
                    average_loss_parameters[curr]=param_combination[va_loss]
            best_loss_str += ' '.join(["python code/train_data_ALSTM.py","-a 2","-b",str(best_loss_parameters["batch"]),'-drop',str(best_loss_parameters["drop_out"]),'-embed',str(best_loss_parameters['embedding_size']),'-e 50','-hidden', str(best_loss_parameters['hidden']),'-i 1','-l',str(best_loss_parameters['lag']),'-lambd 0','-lrate 0.001','-savel 0','-savep 0', '-split 0.9','-s', step[1:],'-v',version,'-c exp/online_v10.conf','-sou '+args.source,'-o '+args.action,'-d '+args.date,'-method best_loss','>',"/dev/null",'2>&1 &'])+"\n"

            best_acc_str += ' '.join(["python code/train_data_ALSTM.py","-a 2","-b",str(best_acc_parameters["batch"]),'-drop',str(best_acc_parameters["drop_out"]),'-embed',str(best_acc_parameters['embedding_size']),'-e 50','-hidden', str(best_acc_parameters['hidden']),'-i 1','-l',str(best_acc_parameters['lag']),'-lambd 0','-lrate 0.001','-savel 0','-savep 0', '-split 0.9','-s', step[1:],'-v',version,'-c exp/online_v10.conf','-sou '+args.source,'-o '+args.action,'-d '+args.date,'-method best_acc','>',"/dev/null",'2>&1 &'])+"\n"
            
            average_loss_str += ' '.join(["python code/train_data_ALSTM.py","-a 2","-b",str(average_loss_parameters["batch"]),'-drop',str(average_loss_parameters["drop_out"]),'-embed',str(average_loss_parameters['embedding_size']),'-e 50','-hidden', str(average_loss_parameters['hidden']),'-i 1','-l',str(average_loss_parameters['lag']),'-lambd 0','-lrate 0.001','-savel 0','-savep 0', '-split 0.9','-s', step[1:],'-v',version,'-c exp/online_v10.conf','-sou '+args.source,'-o '+args.action,'-d '+args.date,'-method ave_loss','>',"/dev/null",'2>&1 &'])+"\n"

            average_acc_str += ' '.join(["python code/train_data_ALSTM.py","-a 2","-b",str(average_acc_parameters["batch"]),'-drop',str(average_acc_parameters["drop_out"]),'-embed',str(average_acc_parameters['embedding_size']),'-e 50','-hidden', str(average_acc_parameters['hidden']),'-i 1','-l',str(average_acc_parameters['lag']),'-lambd 0','-lrate 0.001','-savel 0','-savep 0', '-split 0.9','-s', step[1:],'-v',version,'-c exp/online_v10.conf','-sou '+args.source,'-o '+args.action,'-d '+args.date,'-method ave_acc','>',"/dev/null",'2>&1 &'])+"\n"

        with open("best_loss_"+args.action+".sh","w") as bl:
            bl.write("#!/bin/bash\n")
            bl.write(best_loss_str)
            bl.close()

        with open("best_acc_"+args.action+".sh","w") as bl:
            bl.write("#!/bin/bash\n")
            bl.write(best_acc_str)
            bl.close()

        with open("ave_loss_"+args.action+".sh","w") as bl:
            bl.write("#!/bin/bash\n")
            bl.write(average_loss_str)
            bl.close()


        with open("ave_acc_"+args.action+".sh","w") as bl:
            bl.write("#!/bin/bash\n")
            bl.write(average_acc_str)
            bl.close()