import copy
import json
import numpy as np
import subprocess

from evaluator import evaluate, compare


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


def grid_search(model, selected_parameters, parameter_values, repeat=False,
                best_performance=None
                ):
    # init_paras = model_paras
    print('selected parameters:', selected_parameters)
    print('parameter candidates:', parameter_values)
    if best_performance is None:
        best_performance = {'acc': 0, 'mcc': -2}
    # print 'init parameter:', init_paras
    print('init parameters:', model.paras)
    print('init performance:', best_performance)
    parameter_combinations = []
    _gen_grid_search_all_para(parameter_combinations,
                              copy.copy(model.paras),
                              parameter_values, selected_parameters, 0)
    print('#parameter combinations:', len(parameter_combinations))
    # best_parameters = copy.copy(model.parameters)
    best_parameters = {}
    for metric_name in best_performance.keys():
        best_parameters[metric_name] = copy.copy(model.paras)
    for current_parameters in parameter_combinations:
        print('\n\t\t', current_parameters)
        if not model.update_model(current_parameters):
            continue
        # try:
        cur_val_perf, cur_tes_perf = model.train(tune_para=True)
        if repeat:
            rep_cou = 3
            for i in range(rep_cou - 1):
                temp_val_perf, temp_tes_perf = model.train(tune_para=True)
                for met in cur_val_perf.keys():
                    cur_val_perf[met] = cur_val_perf[met] + temp_val_perf[met]
                    cur_tes_perf[met] = cur_tes_perf[met] + temp_tes_perf[met]
            for met in cur_val_perf.keys():
                cur_val_perf[met] = cur_val_perf[met] / rep_cou
                cur_tes_perf[met] = cur_tes_perf[met] / rep_cou

        print(cur_val_perf)
        is_better = compare(cur_val_perf, best_performance)
        print(is_better)
        for metric_name in is_better.keys():
            if is_better[metric_name]:
                best_performance[metric_name] = \
                    cur_val_perf[metric_name]
                best_parameters[metric_name] = copy.copy(current_parameters)
                print('better paras:', current_parameters, 'on:', metric_name)
                print('val perf:', cur_val_perf)
                print('tes perf:', cur_tes_perf)
        # except:
        #     print('Exception happends during training')
    print('\nbest performance:', best_performance)
    print('\nbest parameter:', best_parameters)
    return best_parameters


def grid_search_shit(model, selected_parameters, parameter_values, script='lstm.py',
                     log_file='./tune.log', repeat=False, best_performance=None):
    # init_paras = model_paras
    print('selected parameters:', selected_parameters)
    print('parameter candidates:', parameter_values)
    if best_performance is None:
        best_performance = {
            'acc': 0, 'mcc': -2
        }
    # print 'init parameter:', init_paras
    print('init parameters:', model.paras)
    print('init performance:', best_performance)
    parameter_combinations = []
    _gen_grid_search_all_para(parameter_combinations,
                              copy.copy(model.paras),
                              parameter_values, selected_parameters, 0)
    print('#parameter combinations:', len(parameter_combinations))
    # best_parameters = copy.copy(model.parameters)
    # best_parameters = {}
    # for metric_name in best_performance.keys():
    #     best_parameters[metric_name] = copy.copy(model.paras)
    ofile = open(log_file, 'wb')
    for cur_paras in parameter_combinations:
        print('\n\t\t', cur_paras)
        ofile.write(('\n\t\t' + json.dumps(cur_paras) + '\n').encode())
        # if not model.update_model(current_parameters):
        #     continue
        output = subprocess.check_output(
            ['python', script,
             '-rl', '1', '-v', '1', '-o', 'train', '-r', str(cur_paras['lr']),
             '-l', str(cur_paras['seq']), '-u', str(cur_paras['unit']),
             '-l2', str(cur_paras['alp']), '-la', str(cur_paras['bet']), '-le', str(cur_paras['eps']),
             '-q', model.model_path, '-qs', model.model_save_path,
             '-p', model.data_path
             ])
        ofile.write(output)
        print(output)
        # cur_val_perf, cur_tes_perf = model.train(tune_para=True)
        # if repeat:
        #     for i in range(4):
        #         temp_val_perf, temp_tes_perf = model.train(tune_para=True)
        #         for met in cur_val_perf.keys():
        #             cur_val_perf[met] = cur_val_perf[met] + temp_val_perf[met]
        #             cur_tes_perf[met] = cur_tes_perf[met] + temp_tes_perf[met]
        #     for met in cur_val_perf.keys():
        #         cur_val_perf[met] = cur_val_perf[met] / 5
        #         cur_tes_perf[met] = cur_tes_perf[met] / 5
        #
        # print(cur_val_perf)
        # is_better = compare(cur_val_perf, best_performance)
        # print(is_better)
        # for metric_name in is_better.keys():
        #     if is_better[metric_name]:
        #         best_performance[metric_name] = \
        #             cur_val_perf[metric_name]
        #         best_parameters[metric_name] = copy.copy(current_parameters)
        #         print('better paras:', current_parameters, 'on:', metric_name)
        #         print('val perf:', cur_val_perf)
        #         print('tes perf:', cur_tes_perf)
        # except:
        #     print('Exception happends during training')
    # print('\nbest performance:', best_performance)
    # print('\nbest parameter:', best_parameters)
    # return best_parameters
