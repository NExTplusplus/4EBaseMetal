import pandas as pd
import numpy as np
path_list = ['Three_10years_h1.txt', 'Three_10years_h3.txt', 'Three_10years_h5.txt']
for new_path in path_list:
    metal_dictionary = {}
    for item in ["LMCADY","LMAHDY","LMNIDY","LMSNDY","LMZSDY","LMPBDY"]:
        metal_dictionary[item]=[]
    length_dictionary = {}
    for item in ["LMCADY","LMAHDY","LMNIDY","LMSNDY","LMZSDY","LMPBDY"]:
        length_dictionary[item]=[]
    metal_list = []
    path = new_path
    all_file = []
    sub_file = []
    all_voting_Str = 'ground truth is'
    precison_str = 'the all folder voting precision is'
    with open(path,"r") as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            if all_voting_Str.lower() in line.lower():
                metal_list.append(line.strip("\n").split(" ")[-1])
                #file.append(line.strip("\n").split(" ")[-1])
            if precison_str.lower() in line.lower():
                length_dictionary[metal_list[-1]].append(int(lines[i-1].strip("\n")[-1]))
                metal_dictionary[metal_list[-1]].append(float(line.strip("\n").split(" ")[-1]))
                #file.append(float(line.strip("\n").split(" ")[-1]))
    metal_result = []
    for key in metal_dictionary.keys():
        new_data = []
        new_data.append(key)
        new_data+=metal_dictionary[key]
        all_sum = 0
        for i in range(len(metal_dictionary[key])):
            all_sum+=metal_dictionary[key][i]*length_dictionary[key][i]
        new_data.append(all_sum/np.sum(length_dictionary[key]))
        metal_result.append(new_data)
    print(metal_result)
    final_data = pd.DataFrame(metal_result,columns=['metal', '2017-01-01', '2017-07-01', '2018-01-01', '2018-07-01', 'average'])
    final_data.to_csv(new_path.split("_")[2]+".csv",index=False)