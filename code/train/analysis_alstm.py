import copy
import json
import os
import numpy as np
import sys
import pandas as pd
if __name__ == '__main__':
    files = sys.argv
    print(sys.argv)
    exp_path = files[1]
    fnames = []
    for ind, file in enumerate(files):
        if ind > 1:
            fnames.append(os.path.join(exp_path, file))
    action = None
    # fnames = ['/Users/ffl/Research/exp_4e/alstm_log_all/tune_alstm_h5.log']
    # action = 'para'
    print('fnames:', fnames)

    # !!! should be exactly same as the "ground_truths_list" in train_alstm.py
    #cases = ['LME_Co_Spot', 'LME_Al_Spot', 'LME_Le_Spot', 'LME_Ni_Spot',
    #         'LME_Zi_Spot', 'LME_Ti_Spot']
    #case_number = len(cases)
    #paras = []
    Co = []
    Co.append('Co')
    Al = []
    Al.append('Al')
    Le = []
    Le.append('Le')
    Ni = []
    Ni.append('Ni')
    Zi = []
    Zi.append('Zi')
    Ti = []
    Ti.append('Ti')
    for fname in fnames:
        with open(fname) as fin:
            lines = fin.readlines()
            paras_count = 0
            test_value_list = []
            val_value_list = []
            Co_value_list = []
            Al_value_list = []
            Le_value_list = []
            Ni_value_list = []
            Zi_value_list = []
            Ti_value_list = []
            epoch = 0
            for ind, line in enumerate(lines):
                #line = line.replace('\'', '"')
                # the start of a hyper-parameter combination
                #if 'the split date is ' in line:
                    #print(line)
                if 'current epoch' in line:
                    if epoch==50:
                        val_value_list.sort(reverse=False, key=lambda x:x[1])
                        #print(val_value_list)
                        index = val_value_list[0][0]
                        print(index)
                        val_value_list=[]
                        for item in Co_value_list:
                            if item[0]==index:
                                Co.append(item[1])
                        Co_value_list=[]
                        for item in Al_value_list:
                            if item[0]==index:
                                Al.append(item[1])
                        Al_value_list=[]
                        for item in Le_value_list:
                            if item[0]==index:
                                Le.append(item[1])
                        Le_value_list=[]
                        for item in Ni_value_list:
                            if item[0]==index:
                                Ni.append(item[1])
                        Ni_value_list=[]
                        for item in Zi_value_list:
                            if item[0]==index:
                                Zi.append(item[1])
                        Zi_value_list=[]
                        for item in Ti_value_list:
                            if item[0]==index:
                                Ti.append(item[1])
                        Ti_value_list=[] 
                    #epoch=0
                    epoch = int(line.strip("\n").split(" ")[-1])
                    #val_value_list=[]
                    #print(epoch)
                    val_value_list.append((epoch, float(lines[ind+2].strip("\n").split(" ")[3].strip(","))))
                    #print(float(lines[ind+2].strip("\n").split(" ")[3].strip(",")))
                    #line = lines[ind+3]
                    #print(line)
                    test_value_list.append((epoch, float(lines[ind+3].strip("\n").split(" ")[5])))
                    #print(ind)
                    #print(lines[ind+3].strip("\n").split(" "))
                    #print(epoch)
                    #print(float(lines[ind+3].strip("\n").split(" ")[5].strip(",")))
                    #print(ind)
                    Co_value_list.append((epoch, float(lines[ind+4].strip("\n").split(" ")[2])))
                    #print(float(lines[ind+4].strip("\n").split(" ")[2]))
                    Al_value_list.append((epoch, float(lines[ind+5].strip("\n").split(" ")[2])))
                    #print(float(lines[ind+5].strip("\n").split(" ")[2]))
                    Le_value_list.append((epoch, float(lines[ind+6].strip("\n").split(" ")[2])))
                    #print(float(lines[ind+6].strip("\n").split(" ")[2]))
                    Ni_value_list.append((epoch, float(lines[ind+7].strip("\n").split(" ")[2])))
                    #print(float(lines[ind+7].strip("\n").split(" ")[2]))
                    Zi_value_list.append((epoch, float(lines[ind+8].strip("\n").split(" ")[2])))
                    #print(float(lines[ind+8].strip("\n").split(" ")[2]))
                    Ti_value_list.append((epoch, float(lines[ind+9].strip("\n").split(" ")[2]))) 
                    #print(float(lines[ind+9].strip("\n").split(" ")[2]))
    val_value_list.sort(reverse=False, key=lambda x:x[1])
    #print(val_value_list)
    index = val_value_list[0][0]
    print(index)
    val_value_list=[]
    for item in Co_value_list:
        if item[0]==index:
            Co.append(item[1])
    Co_value_list=[]
    for item in Al_value_list:
        if item[0]==index:
            Al.append(item[1])
    Al_value_list=[]
    for item in Le_value_list:
        if item[0]==index:
            Le.append(item[1])
    Le_value_list=[]
    for item in Ni_value_list:
        if item[0]==index:
            Ni.append(item[1])
    Ni_value_list=[]
    for item in Zi_value_list:
        if item[0]==index:
            Zi.append(item[1])
    Zi_value_list=[]
    for item in Ti_value_list:
        if item[0]==index:
            Ti.append(item[1])
    Ti_value_list=[]    
    data_list = []
    #Co.append('Co')
    data_list.append(Co)
    #data_list.append('Al')
    data_list.append(Al)
    #data_list.append('Le')
    data_list.append(Le)
    #data_list.append('Ni')
    data_list.append(Ni)
    #data_list.append('Zi')
    data_list.append(Zi)
    #data_list.append('Ti')
    data_list.append(Ti)
    #print(data_list)
    data_frame = pd.DataFrame(data=data_list,columns=['metal','2017-01-01','2017-07-01','2018-01-01','2018-07-01'])
    data_frame.to_csv(fnames[-1].split(".")[0]+".csv",index=False)

