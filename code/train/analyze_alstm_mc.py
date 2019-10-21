import copy
import json
import os
import numpy as np
import sys


def _check_para_order(par1, par2):
    if not par1['lag'] == par2['lag']:
        return par1['lag'] > par2['lag']
    elif not par1['hidden'] == par2['hidden']:
        return par1['hidden'] > par2['hidden']
    elif not par1['embedding_size'] == par2['embedding_size']:
        return par1['embedding_size'] > par2['embedding_size']
    elif not par1['drop_out'] == par2['drop_out']:
        return par1['drop_out'] > par2['drop_out']
    elif not par1['batch'] == par2['batch']:
        return par1['batch'] > par2['batch']
    else:
        return False


def _parse_test_line(line):
    line = line.replace(',', '')
    tokens = line.split(' ')
    perf = {}
    for ind, tok in enumerate(tokens):
        if 'accuracy' in tok:
            perf['acc'] = float(tokens[ind + 1])
        if 'loss' in tok:
            perf['loss'] = float(tokens[ind + 1])
    return perf


def _parse_case_lines(case_lines):
    # case acc: 0.6341463414634146
    case_perfs = []
    for line in case_lines:
        tokens = line.split(' ')
        case_perfs.append(float(tokens[-1]))
    return case_perfs


def _parse_top_bot_lines(tb_lines):
    # top acc: 0.4390 ::: bot acc: 0.7805
    tb_perfs = []
    for line in tb_lines:
        sub_lines = line.split(' ::: ')
        tb_perf = []
        tokens = sub_lines[0].split(' ')
        tb_perf.append(float(tokens[-1]))
        tokens = sub_lines[1].split(' ')
        tb_perf.append(float(tokens[-1]))
        tb_perfs.append(tb_perf)
    return tb_perfs


def _get_stop_perf(perfs, stop, stop_window=10):
    if stop == 'last':
        # need to be updated
        return perfs[0][-1]['acc'], perfs[1][-1]['acc']
    elif stop == 'best_val':
        # val_loss_mean = 0
        # val_acc_mean = 0
        #
        # test_loss_mean = 0
        # test_acc_mean = 0
        # test_case_mean = np.zeros(len(perfs[1][0][0]['case']), dtype=float)
        # test_tp_mean = copy.copy(test_case_mean)
        #
        # rollings = len(perfs[0])
        # for rolling in range(rollings):
        #     val_perfs = perfs[0][rolling]
        #     tes_perfs = perfs[1][rolling]
        #
        #     best_val_loss = 1e10
        #     best_val_acc = 0
        #     best_test_perf = None
        #
        #     for val_perf, tes_perf in zip(val_perfs, tes_perfs):
        #         # get the best epoch
        #         if val_perf['loss'] < best_val_loss:
        #             best_val_loss = val_perf['loss']
        #             best_val_acc = val_perf['acc']
        #             best_test_perf = tes_perf
        #
        #     # get average val performance across rollings at selected epoch
        #     val_loss_mean += best_val_loss
        #     val_acc_mean += best_val_acc
        #
        #     # get average test performance across rollings at selected epoch
        #     test_loss_mean += best_test_perf['loss']
        #     test_acc_mean += best_test_perf['acc']
        #     test_case_mean += best_test_perf['case']
        #     for case in range(len(best_test_perf['tb'])):
        #         test_tp_mean[case] = test_tp_mean[case] + (best_test_perf['tb'][case][0] + best_test_perf['tb'][case][1]) / 2
        #     # test_tp_mean += test_perf_rolling['tb']
        # val_loss_mean /= rollings
        # val_acc_mean /= rollings
        # test_loss_mean /= rollings
        # test_acc_mean /= rollings
        # test_case_mean /= rollings
        # test_tp_mean /= rollings
        #
        # return val_loss_mean, val_acc_mean, \
        #        test_loss_mean, test_acc_mean, test_case_mean, test_tp_mean

        rollings = len(perfs[0])

        val_loss_rolls = np.zeros(rollings, dtype=float)
        val_acc_rolls = np.zeros(rollings, dtype=float)

        test_loss_rolls = np.zeros(rollings, dtype=float)
        test_acc_rolls = np.zeros(rollings, dtype=float)

        test_case_rolls = np.zeros([rollings, len(perfs[1][0][0]['case'])],
                                   dtype=float)
        test_tp_rolls = copy.copy(test_case_rolls)
        for rolling in range(rollings):
            val_perfs = perfs[0][rolling]
            tes_perfs = perfs[1][rolling]

            best_val_loss = 1e10
            best_val_acc = 0
            best_test_perf = None

            for val_perf, tes_perf in zip(val_perfs, tes_perfs):
                # get the best epoch
                if val_perf['loss'] < best_val_loss:
                    best_val_loss = val_perf['loss']
                    best_val_acc = val_perf['acc']
                    best_test_perf = tes_perf

            # record the validation performance at selected epoch
            val_loss_rolls[rolling] = best_val_loss
            val_acc_rolls[rolling] = best_val_acc

            # record the testing performance at selected epoch
            test_loss_rolls[rolling] = best_test_perf['loss']
            test_acc_rolls[rolling] = best_test_perf['acc']
            test_case_rolls[rolling] = copy.copy(best_test_perf['case'])

            test_tp_mean = np.zeros(len(perfs[1][0][0]['case']), dtype=float)
            for case in range(len(best_test_perf['tb'])):
                test_tp_mean[case] = (best_test_perf['tb'][case][0] +
                                      best_test_perf['tb'][case][1]) / 2
            test_tp_rolls[rolling] = copy.copy(test_tp_mean)

        # format stop performance
        stop_perf = {}
        stop_perf['va_loss'] = val_loss_rolls
        stop_perf['va_acc'] = val_acc_rolls
        stop_perf['te_loss'] = test_loss_rolls
        stop_perf['te_acc'] = test_acc_rolls
        stop_perf['cases'] = test_case_rolls
        stop_perf['tps'] = test_tp_rolls
        return stop_perf

        # val_loss_mean /= rollings
        # val_acc_mean /= rollings
        # test_loss_mean /= rollings
        # test_acc_mean /= rollings
        # test_case_mean /= rollings
        # test_tp_mean /= rollings
        #
        # return val_loss_mean, val_acc_mean, \
        #        test_loss_mean, test_acc_mean, test_case_mean, test_tp_mean
    elif stop == 'early_va_loss':
        # need to be updated
        va_losses = [val_perf['loss'] for val_perf in perfs[0]]
        for epoch in range(stop_window, len(va_losses)):
            if va_losses[epoch] > np.mean(va_losses[epoch - stop_window: epoch]):
                return perfs[0][epoch]['acc'], perfs[1][epoch]['acc']
        return perfs[0][-1]['acc'], perfs[1][-1]['acc']
    else:
        raise NotImplementedError


def parse_performance(fnames, case_number=6, is_test=False):
    # parse performances
    paras = []
    perfs = []
    val_perfs_rolling = []
    tes_perfs_rolling = []
    val_perfs = []
    tes_perfs = []
    for fname in fnames:
        with open(fname) as fin:
            lines = fin.readlines()
            for ind, line in enumerate(lines):
                line = line.replace('\'', '"')
                # the start of a hyper-parameter combination
                if '\t\t {"drop_out":' in line or '\t\t{"drop_out":' in line or\
                        (is_test and 'drop_out=' in line):
                    # insert the performance of the previous parameter
                    # combination into the list
                    if not len(paras) == 0:
                        val_perfs_rolling.append(val_perfs)
                        tes_perfs_rolling.append(tes_perfs)
                        perfs.append([val_perfs_rolling, tes_perfs_rolling])
                    val_perfs = []
                    tes_perfs = []
                    if is_test:
                        para = line
                    else:
                        para = json.loads(line)
                    paras.append(para)

                    val_perfs_rolling = []
                    tes_perfs_rolling = []

                # the start of a rolling time window
                if 'preparing training' in line:
                    # insert the performance of previous rolling
                    if not len(val_perfs) == 0:
                        val_perfs_rolling.append(val_perfs)
                        tes_perfs_rolling.append(tes_perfs)
                    val_perfs = []
                    tes_perfs = []

                # parse performance of each epoch
                '''
                 train loss is 0.150938
                 average val loss: 0.059852, accuracy: 0.5448
                 average test loss: 0.067483, accuracy: 0.4621
                 case acc: 0.6341463414634146
                 case acc: 0.45528455284552843
                 case acc: 0.3983739837398374
                 case acc: 0.4146341463414634
                 case acc: 0.4878048780487805
                 case acc: 0.3821138211382114
                 top acc: 0.4390 ::: bot acc: 0.7805
                 top acc: 0.6341 ::: bot acc: 0.2927
                 top acc: 1.0000 ::: bot acc: 0.0000
                 top acc: 0.9024 ::: bot acc: 0.1463
                 top acc: 0.8049 ::: bot acc: 0.0488
                 top acc: 1.0000 ::: bot acc: 0.0000
                '''
                if 'current epoch' in line:
                    val_perfs.append(_parse_test_line(lines[ind + 2]))
                    test_perf = _parse_test_line(lines[ind + 3])
                    test_perf['case'] = _parse_case_lines(
                        lines[ind + 4: ind + 4 + case_number])
                    test_perf['tb'] = _parse_top_bot_lines(
                        lines[ind + 4 + case_number: ind + 4 + 2 * case_number])
                    tes_perfs.append(test_perf)

                # last case
                if ind == len(lines) - 1:
                    val_perfs_rolling.append(val_perfs)
                    tes_perfs_rolling.append(tes_perfs)
                    perfs.append([val_perfs_rolling, tes_perfs_rolling])

    assert len(paras) == len(perfs), 'length of paras and perfs do not ' \
                                     'mathch %d VS %d' % (len(paras), len(perfs))

    if is_test:
        return paras, perfs
    # sort parameters
    for i in range(len(paras)):
        for j in range(i + 1, len(paras)):
            if _check_para_order(paras[i], paras[j]):
                temp = paras[i]
                paras[i] = paras[j]
                paras[j] = temp

                temp = perfs[i]
                perfs[i] = perfs[j]
                perfs[j] = temp
    return paras, perfs


def get_performance_stop_epoch(paras, perfs, stops, stop_window=10):
    select_perfs_stops = []
    assert len(paras) == len(perfs), 'length mismatch before stop inference'
    # report results
    for stop in stops:
        select_perfs = []
        for i in range(len(paras)):
            print(paras[i])
            stop_perf = _get_stop_perf(perfs[i], stop, stop_window=stop_window)
            print('va_loss:', stop_perf['va_loss'], np.mean(stop_perf['va_loss']))
            print('va_acc:', stop_perf['va_acc'], np.mean(stop_perf['va_acc']))
            print('te_loss:', stop_perf['te_loss'], np.mean(stop_perf['te_loss']))
            print('te_acc:', stop_perf['te_acc'], np.mean(stop_perf['te_acc']))
            select_perfs.append(stop_perf)
        select_perfs_stops.append(select_perfs)
    return select_perfs_stops


def select_parameter_average_test_loss(ofname, paras, perfs, perfs_stops, stop, stops):
    if stop not in stops:
        print('unexpected stop method: ', stop)
        exit(1)
    assert len(paras) == len(perfs), 'length mismatch before para selection'

    perfs_stop = None
    for i, s in enumerate(stops):
        if s == stop:
            perfs_stop = perfs_stops[i]
    assert len(paras) == len(perfs_stop), 'length mismatch on stop perfs'

    best_val_loss = 0
    best_val_acc = 0
    best_test_loss = 1e10
    best_test_acc = 0
    best_test_tp_acc = 0
    best_test_cases = None
    best_test_tps = None
    best_para = None
    fout = open(ofname, 'w+')
    fout.write('para_combination, va_loss, va_acc, te_loss, te_acc, te_tp_acc')
    for i in range(perfs_stop[0]['cases'].shape[1]):
        fout.write(', case' + str(i))
    for i in range(perfs_stop[0]['cases'].shape[1]):
        fout.write(', tp_case' + str(i))
    fout.write('\n')

    # traverse all parameter combinations
    for i in range(len(paras)):
        print(paras[i])
        fout.write(json.dumps(paras[i]).replace(',', '') + ',')
        stop_perf = perfs_stop[i]
        val_loss_mean = np.mean(stop_perf['va_loss'])
        val_acc_mean = np.mean(stop_perf['va_acc'])
        test_loss_mean = np.mean(stop_perf['te_loss'])
        test_acc_mean = np.mean(stop_perf['te_acc'])
        test_tp_mean = np.mean(stop_perf['tps'])
        print(
            'va_loss: {:.6f}'.format(val_loss_mean),
            'va_acc: {:.4f}'.format(val_acc_mean),
            'te_loss: {:.6f}'.format(test_loss_mean),
            'te_acc: {:.4f}'.format(test_acc_mean),
            'te_tp_acc: {:.4f}'.format(test_tp_mean)
        )
        fout.write('{:.6f}, {:.4f}, {:.6f}, {:.4f}, {:.4f}'.format(
            val_loss_mean, val_acc_mean, test_loss_mean, test_acc_mean,
            test_tp_mean
        ))
        test_cases = np.mean(stop_perf['cases'], axis=0)
        for case in test_cases:
            fout.write(', {:.4f}'.format(case))
        test_tp_cases = np.mean(stop_perf['tps'], axis=0)
        for case in test_tp_cases:
            fout.write(', {:.4f}'.format(case))
        fout.write('\n')

        # if stop_perf[0] < best_val_loss:
        if test_loss_mean < best_test_loss:
            best_val_loss = val_loss_mean
            best_val_acc = val_acc_mean

            best_test_loss = test_loss_mean
            best_test_acc = test_acc_mean
            best_test_tp_acc = test_tp_mean

            best_test_cases = np.mean(stop_perf['cases'], axis=0)
            best_test_tps = np.mean(stop_perf['tps'], axis=0)

            best_para = paras[i]
        # break

    print('Best valid loss: {:.6f} acc: {:.4f}'.format(best_val_loss, best_val_acc))
    print('Best test loss: {:.6f} acc: {:.4f} tp_acc: {:.4f}'.format(
        best_test_loss, best_test_acc, best_test_tp_acc)
    )
    print('Best test acc cases:', best_test_cases)
    print('Best test acc tps:', best_test_tps)
    print('Best para:', best_para)


def select_parameter_test_loss_ratio(ofname, paras, perfs, perfs_stops, stop, stops):
    if stop not in stops:
        print('unexpected stop method: ', stop)
        exit(1)
    assert len(paras) == len(perfs), 'length mismatch before para selection'

    perfs_stop = None
    for i, s in enumerate(stops):
        if s == stop:
            perfs_stop = perfs_stops[i]
    assert len(paras) == len(perfs_stop), 'length mismatch on stop perfs'

    best_val_loss = 0
    best_val_acc = 0
    best_test_loss = 1e10
    best_test_acc = 0
    best_test_tp_acc = 0
    best_test_cases = None
    best_test_tps = None
    best_para = None
    fout = open(ofname, 'w+')
    fout.write('para_combination, va_loss, va_acc, te_loss, te_acc, te_tp_acc')
    for i in range(perfs_stop[0]['cases'].shape[1]):
        fout.write(', case' + str(i))
    for i in range(perfs_stop[0]['cases'].shape[1]):
        fout.write(', tp_case' + str(i))
    fout.write('\n')

    # scale val loss
    val_loss_all = np.zeros([len(paras), perfs_stop[0]['cases'].shape[0]],
                           dtype=float)
    test_loss_all = np.zeros([len(paras), perfs_stop[0]['cases'].shape[0]],
                             dtype=float)
    for i in range(len(paras)):
        val_loss_all[i] = perfs_stop[i]['va_loss']
        test_loss_all[i] = perfs_stop[i]['te_loss']

    min_val_loss_rolls = np.min(val_loss_all, axis=0)
    print('min val loss:', min_val_loss_rolls)
    min_test_loss_rolls = np.min(test_loss_all, axis=0)
    print('min test loss:', min_test_loss_rolls)
    val_loss_all /= min_val_loss_rolls
    test_loss_all /= min_test_loss_rolls

    for i in range(len(paras)):
        perfs_stop[i]['va_loss'] = val_loss_all[i]
        perfs_stop[i]['te_loss'] = test_loss_all[i]

    # traverse all parameter combinations
    for i in range(len(paras)):
        print(paras[i])
        fout.write(json.dumps(paras[i]).replace(',', '') + ',')
        stop_perf = perfs_stop[i]
        val_loss_mean = np.mean(stop_perf['va_loss'])
        val_acc_mean = np.mean(stop_perf['va_acc'])
        test_loss_mean = np.mean(stop_perf['te_loss'])
        test_acc_mean = np.mean(stop_perf['te_acc'])
        test_tp_mean = np.mean(stop_perf['tps'])
        print(
            'va_loss: {:.6f}'.format(val_loss_mean),
            'va_acc: {:.4f}'.format(val_acc_mean),
            'te_loss: {:.6f}'.format(test_loss_mean),
            'te_acc: {:.4f}'.format(test_acc_mean),
            'te_tp_acc: {:.4f}'.format(test_tp_mean)
        )
        fout.write('{:.6f}, {:.4f}, {:.6f}, {:.4f}, {:.4f}'.format(
            val_loss_mean, val_acc_mean, test_loss_mean, test_acc_mean,
            test_tp_mean
        ))
        test_cases = np.mean(stop_perf['cases'], axis=0)
        for case in test_cases:
            fout.write(', {:.4f}'.format(case))
        test_tp_cases = np.mean(stop_perf['tps'], axis=0)
        for case in test_tp_cases:
            fout.write(', {:.4f}'.format(case))
        fout.write('\n')

        # if stop_perf[0] < best_val_loss:
        if test_loss_mean < best_test_loss:
            best_val_loss = val_loss_mean
            best_val_acc = val_acc_mean

            best_test_loss = test_loss_mean
            best_test_acc = test_acc_mean
            best_test_tp_acc = test_tp_mean

            best_test_cases = np.mean(stop_perf['cases'], axis=0)
            best_test_tps = np.mean(stop_perf['tps'], axis=0)

            best_para = paras[i]
        # break

    print('Best valid loss: {:.6f} acc: {:.4f}'.format(best_val_loss, best_val_acc))
    print('Best test loss: {:.6f} acc: {:.4f} tp_acc: {:.4f}'.format(
        best_test_loss, best_test_acc, best_test_tp_acc)
    )
    print('Best test acc cases:', best_test_cases)
    print('Best test acc tps:', best_test_tps)
    print('Best para:', best_para)


def report_test_performance(ofname, cases, perfs_stop):
    fout = open(ofname, 'w+')
    fout.write('Metal, ' + ', '.join(np.arange(perfs_stop['cases'].shape[0]).astype(str)) +
               ', Mean, , ' + ', '.join(np.arange(perfs_stop['cases'].shape[0]).astype(str)) +
               ', Mean\n')
    for i in range(len(cases)):
        fout.write(cases[i].replace('_spot', '').replace('LME_', '') + ', ')
        fout.write(', '.join(perfs_stop['cases'][:, i].astype(str)))
        fout. write(', {:.6f}, , '.format(np.mean(perfs_stop['cases'][:, i])))
        fout.write(', '.join(perfs_stop['tps'][:, i].astype(str)))
        fout.write(', {:.6f}\n'.format(np.mean(perfs_stop['tps'][:, i])))
    fout.close()


def report_parameter_performance(ofname, paras, perfs_stop):
    # get values for each parameter
    para_values = {}
    for para in paras:
        for kv in para.items():
            if kv[0] not in para_values.keys():
                para_values[kv[0]] = [kv[1]]
            elif kv[1] not in para_values[kv[0]]:
                para_values[kv[0]].append(kv[1])

    # report peformance by parameter values
    for kv in para_values.items():
        print(kv)
        if len(kv[1]) == 1:
            continue
        else:
            print(kv[0])

        perf_values = []
        para_values = []
        for value in kv[1]:
            perf_value = []
            para_value = []
            for para, perf in zip(paras, perfs_stop):
                if para[kv[0]] == value:
                    temp_para = copy.copy(para)
                    temp_para.pop(kv[0])
                    perf_value.append(copy.copy(perf))
                    para_value.append(temp_para)
            perf_values.append(perf_value)
            para_values.append(para_value)
            print('number of paras:', len(perf_value))

        # check number of parameter combinations under each parameter value
        for i in range(1, len(kv[1])):
            assert len(para_values[i]) == len(para_values[i - 1]), 'lengths mismatch'

        # check order of parameter combinations under each parameter value
        for j in range(len(para_values[-1])):
            for i in range(1, len(kv[1])):
                assert para_values[i - 1][j] == para_values[i][j], \
                    'order mismatch: ' + json.dumps(para_values[i - 1][j]) + \
                    json.dumps(para_values[i][j])

        # write out performance
        fout = open(ofname + kv[0] + '.csv', 'w+')
        fout.write('Parameter_Combination, ')
        fout.write('_va_loss, '.join(list(map(str, kv[1]))))
        fout.write(', ')
        fout.write('_va_acc, '.join(list(map(str, kv[1]))))
        fout.write(', ')
        fout.write('_te_loss, '.join(list(map(str, kv[1]))))
        fout.write(', ')
        fout.write('_te_acc, '.join(list(map(str, kv[1]))))
        fout.write(', ')
        fout.write('_te_tp, '.join(list(map(str, kv[1]))))
        fout.write('\n')

        va_loss_mean = np.zeros(len(kv[1]), dtype=float)
        va_acc_mean = np.zeros(len(kv[1]), dtype=float)
        te_loss_mean = np.zeros(len(kv[1]), dtype=float)
        te_acc_mean = np.zeros(len(kv[1]), dtype=float)
        te_tp_mean = np.zeros(len(kv[1]), dtype=float)
        for j in range(len(para_values[-1])):
            fout.write(json.dumps(para_values[-1][j]).replace(',', ''))
            # write validation loss
            for i in range(len(kv[1])):
                fout.write(', {:.6f}'.format(np.mean(perf_values[i][j]['va_loss'])))
                va_loss_mean[i] += np.mean(perf_values[i][j]['va_loss'])

            # write validation accuracy
            for i in range(len(kv[1])):
                fout.write(', {:.4f}'.format(np.mean(perf_values[i][j]['va_acc'])))
                va_acc_mean[i] += np.mean(perf_values[i][j]['va_acc'])

            # write testing loss
            for i in range(len(kv[1])):
                fout.write(', {:.6f}'.format(np.mean(perf_values[i][j]['te_loss'])))
                te_loss_mean[i] += np.mean(perf_values[i][j]['te_loss'])

            # write testing accuracy
            for i in range(len(kv[1])):
                fout.write(', {:.4f}'.format(np.mean(perf_values[i][j]['te_acc'])))
                te_acc_mean[i] += np.mean(perf_values[i][j]['te_acc'])

            # write testing accuracy tps
            for i in range(len(kv[1])):
                fout.write(', {:.4f}'.format(np.mean(perf_values[i][j]['tps'])))
                te_tp_mean[i] += np.mean(perf_values[i][j]['tps'])

            fout.write('\n')
        fout.close()

        # compare the performance of different values
        print(kv[1])
        print('va_loss:', va_loss_mean / len(para_values[-1]))
        print('va_acc:', va_acc_mean / len(para_values[-1]))
        print('te_loss:', te_loss_mean / len(para_values[-1]))
        print('te_acc:', te_acc_mean / len(para_values[-1]))
        print('te_tp:', te_tp_mean / len(para_values[-1]))
        print('---------------------------------------------')


if __name__ == '__main__':
    '''
            format of command: 
                python script.py path file1 file2 ... action
            action is an optional parameter for special analysis, which could be:
                'para'
        '''
    files = sys.argv
    print(sys.argv)
    exp_path = files[1]
    fnames = []
    for ind, file in enumerate(files):
        if ind > 1:
            fnames.append(os.path.join(exp_path, file))
    action = None
    if files[-1] == 'para':
        action = 'para'
        fnames.pop()
    # fnames = ['/Users/ffl/Research/exp_4e/alstm_log_all/tune_alstm_h5.log']
    # action = 'para'
    print('fnames:', fnames)
    stop_method = 'best_val'
    stop_methods = ['best_val']

    # !!! should be exactly same as the "ground_truths_list" in train_alstm.py
    cases = ['LME_Co_Spot', 'LME_Al_Spot', 'LME_Le_Spot', 'LME_Ni_Spot',
             'LME_Zi_Spot', 'LME_Ti_Spot']
    case_number = len(cases)

    if 'tune' in fnames[-1] and not action == 'para':
        # report_performance(fnames, 6, ['best_val'])
        # analyze the log of grid_search
        paras, perfs = parse_performance(fnames, case_number)
        perfs_stops = get_performance_stop_epoch(paras, perfs, stop_methods,
                                                 stop_window=10)
        select_parameter_average_test_loss(
            fnames[-1].replace('.log', '_mean_loss_perf.csv'), paras, perfs,
            perfs_stops, stop_method, stop_methods
        )
        select_parameter_test_loss_ratio(
            fnames[-1].replace('.log', '_mean_loss_ratio_perf.csv'), paras,
            perfs,
            perfs_stops, stop_method, stop_methods
        )
    elif 'txt' in fnames[-1]:
        # analyze the log of online_test
        paras, perfs = parse_performance(fnames, case_number, is_test=True)
        perfs_stops = get_performance_stop_epoch(paras, perfs, stop_methods,
                                                 stop_window=10)

        perfs_stop = None
        for i, s in enumerate(stop_methods):
            if s == stop_method:
                perfs_stop = perfs_stops[i][-1]
        print('case:', np.mean(perfs_stop['cases']))
        print('tp:', np.mean(perfs_stop['tps']))
        report_test_performance(fnames[-1].replace('.txt', '_test_perf.csv'),
                                cases, perfs_stop)
    elif 'tune' in fnames[-1] and action == 'para':
        # analyze the log of grid_search, but report parameter comparison
        paras, perfs = parse_performance(fnames, case_number)
        perfs_stops = get_performance_stop_epoch(paras, perfs, stop_methods,
                                                 stop_window=10)
        perfs_stop = None
        for i, s in enumerate(stop_methods):
            if s == stop_method:
                perfs_stop = perfs_stops[i]
        report_parameter_performance(
            fnames[-1].replace('.log', '_para_perf_'), paras, perfs_stop
        )
    else:
        pass
