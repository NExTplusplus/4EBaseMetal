import copy
import json
import numpy as np
import subprocess

def _gen_grid_search_all_para(parameter_combinations, paras, para_values,
                              selected_paras, para_index):
    if len(selected_paras) == 0:
        return
    if para_index == len(selected_paras) - 1:
        for value in para_values[para_index]:
            paras[selected_paras[para_index]] = value
            parameter_combinations.append(paras.copy())
    else:
        # print(para_index, para_values)
        for value in para_values[para_index]:
            paras[selected_paras[para_index]] = value
            _gen_grid_search_all_para(
                parameter_combinations, paras, para_values, selected_paras,
                para_index + 1
            )


def grid_search_alstm(sel_paras, cand_values, init_paras, script='code/train/train_alstm.py',
                      log_file='./tune.log', steps=5, version='v16', date = "2017-06-30",
                         gt = "LME_Co_Spot",source = "NExT"):
    # init_paras = model_paras
    print('selected parameters:', sel_paras)
    print('parameter candidates:', cand_values)
    print('init parameters:', init_paras)
    parameter_combinations = []
    _gen_grid_search_all_para(parameter_combinations, copy.copy(init_paras),
                              cand_values, sel_paras, 0)
    print('#parameter combinations:', len(parameter_combinations))
    ofile = open(log_file, 'wb')
    for cur_paras in parameter_combinations:
        output = subprocess.check_output(
            ['python', script,
             '--lag', str(cur_paras['lag']),
             '--batch', str(cur_paras['batch']),
             '--drop_out', str(cur_paras['drop_out']),
             '--hidden', str(cur_paras['hidden']),
             '--embedding_size', str(cur_paras['embedding_size']),
             '--steps', str(steps),
             '--version', str(version),
             '--ground_truth', gt,
             '--source', source,
             '--date', date
             ]
        )
        print('\n\t\t', cur_paras)
        ofile.write(('\n\t\t' + json.dumps(cur_paras) + '\n').encode())
        ofile.write(output)
        print(output)


def grid_search_alstm_mc(sel_paras, cand_values, init_paras,
                         script='code/train/train_alstm_mc.py',
                         log_file='./tune.log', steps=5, version ="v16", date = "2017-06-30",
                         gt = "LME_Co_Spot",source = "NExT"):
    # init_paras = model_paras
    print('selected parameters:', sel_paras)
    print('parameter candidates:', cand_values)
    print('init parameters:', init_paras)
    parameter_combinations = []
    _gen_grid_search_all_para(parameter_combinations, copy.copy(init_paras),
                              cand_values, sel_paras, 0)
    print('#parameter combinations:', len(parameter_combinations))
    ofile = open(log_file, 'wb')
    for cur_paras in parameter_combinations:
        output = subprocess.check_output(
            ['python', script,
             '--lag', str(cur_paras['lag']),
             '--batch', str(cur_paras['batch']),
             '--drop_out', str(cur_paras['drop_out']),
             '--drop_out_mc', str(cur_paras['drop_out_mc']),
             '--repeat_mc', str(cur_paras['repeat_mc']),
             '--hidden', str(cur_paras['hidden']),
             '--embedding_size', str(cur_paras['embedding_size']),
             '--steps', str(steps),
             '--version',version,
             '--ground_truth', gt,
             '--source', source,
             '--date', date,
             '--mc', '1'
             ]
        )
        print('\n\t\t', cur_paras)
        ofile.write(('\n\t\t' + json.dumps(cur_paras) + '\n').encode())
        ofile.write(output)
        print(output)
