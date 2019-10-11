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
        val_loss_mean = 0
        val_acc_mean = 0

        test_loss_mean = 0
        test_acc_mean = 0
        test_case_mean = np.zeros(len(perfs[1][0][0]['case']), dtype=float)
        test_tp_mean = copy.copy(test_case_mean)

        rollings = len(perfs[0])
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

            # get average val performance across rollings at selected epoch
            val_loss_mean += best_val_loss
            val_acc_mean += best_val_acc

            # get average test performance across rollings at selected epoch
            test_loss_mean += best_test_perf['loss']
            test_acc_mean += best_test_perf['acc']
            test_case_mean += best_test_perf['case']
            for case in range(len(best_test_perf['tb'])):
                test_tp_mean[case] = test_tp_mean[case] + (best_test_perf['tb'][case][0] + best_test_perf['tb'][case][1]) / 2
            # test_tp_mean += test_perf_rolling['tb']
        val_loss_mean /= rollings
        val_acc_mean /= rollings
        test_loss_mean /= rollings
        test_acc_mean /= rollings
        test_case_mean /= rollings
        test_tp_mean /= rollings

        return val_loss_mean, val_acc_mean, \
               test_loss_mean, test_acc_mean, test_case_mean, test_tp_mean
    elif stop == 'early_va_loss':
        # need to be updated
        va_losses = [val_perf['loss'] for val_perf in perfs[0]]
        for epoch in range(stop_window, len(va_losses)):
            if va_losses[epoch] > np.mean(va_losses[epoch - stop_window: epoch]):
                return perfs[0][epoch]['acc'], perfs[1][epoch]['acc']
        return perfs[0][-1]['acc'], perfs[1][-1]['acc']
    else:
        raise NotImplementedError


def report_performance(fnames, case_number=6, stops=None):
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
                if '\t\t {"drop_out":' in line or '\t\t{"drop_out":' in line:
                    # insert the performance of the previous parameter
                    # combination into the list
                    if not len(paras) == 0:
                        val_perfs_rolling.append(val_perfs)
                        tes_perfs_rolling.append(tes_perfs)
                        perfs.append([val_perfs_rolling, tes_perfs_rolling])
                    val_perfs = []
                    tes_perfs = []
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

    stop_perfs = []
    # report results
    for stop in stops:
        best_val_loss = 0
        best_val_acc = 0
        best_test_loss = 1e10
        best_test_acc = 0
        best_test_tp_acc = 0
        best_para = None
        for i in range(len(paras)):
            print(paras[i])
            stop_perf = _get_stop_perf(perfs[i], stop, stop_window=10)
            print('va_loss: {:.6f}'.format(stop_perf[0]),
                  'va_acc: {:.4f}'.format(stop_perf[1]),
                  'te_loss: {:.6f}'.format(stop_perf[2]),
                  'te_acc: {:.4f}'.format(stop_perf[3]),
                  'te_tp_acc: {:.4f}'.format(np.mean(stop_perf[5])))

            # if stop_perf[0] < best_val_loss:
            if stop_perf[2] < best_test_loss:
                best_val_loss = stop_perf[0]
                best_val_acc = stop_perf[1]
                best_test_loss = stop_perf[2]
                best_test_acc = stop_perf[3]
                best_test_tp_acc = np.mean(stop_perf[5])
                best_para = paras[i]

            stop_perfs.append(stop_perf)
            # break

        print('Best valid loss: {:.6f} acc: {:.4f}'.format(best_val_loss, best_val_acc))
        print('Best test loss: {:.6f} acc: {:.4f} tp_acc: {:.4f}'.format(
            best_test_loss, best_test_acc, best_test_tp_acc)
        )
        print('Best para:', best_para)

        # write out
        print('------------')
    return paras, perfs, stop_perfs


def report_hyperpara(sel_paras, fname):
    with open(fname) as fin:
        lines = fin.readlines()
        paras = []
        perfs = []
        for ind, line in enumerate(lines):
            line = line.replace('\'', '"')
            if line.startswith('{"dropout'):
                para = json.loads(line)
                paras.append(para)
                toks = lines[ind + 1].split(' ')
                perfs.append([float(toks[1]), float(toks[3])])
            if 'Best para' in line:
                line = line.replace('Best para: ', '')
                best_para = json.loads(line)
                for pname in sel_paras:
                    print('tune:', pname)
                    for pind, para in enumerate(paras):
                        # select satisfied paras
                        satisfy = True
                        for qname in sel_paras:
                            if qname == pname:
                                continue
                            if not best_para[qname] == para[qname]:
                                satisfy = False
                        if satisfy:
                            print(para[pname], perfs[pind][1])
                    print('--------------')
                paras = []
                perfs = []
                print('++++++++++++++')


def _get_mean_perf(cur_perfs, repeats):
    mean_per = copy.copy(cur_perfs[0])
    for j in range(1, repeats):
        for metric in mean_per.keys():
            mean_per[metric] = mean_per[metric] + cur_perfs[j][metric]
    for metric in mean_per.keys():
        mean_per[metric] = mean_per[metric] / repeats
    return mean_per


def report_curve(fnames, ofname=''):
    all_accs = []
    for fname in fnames:
        with open(fname) as fin:
            tes_accs = []
            val_accs = []
            lines = fin.readlines()
            for ind, line in enumerate(lines):
                line = line.replace('\'', '"')
                if line.startswith('Epoch'):
                    val_acc = _parse_epoch_line(line)['acc']
                    tes_acc = _parse_test_line(lines[ind + 1])['acc']
                    val_accs.append(val_acc)
                    tes_accs.append(tes_acc)
            best_val_acc = 0
            best_epoch = 0
            for epoch, val_acc in enumerate(val_accs):
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
            print(fname)
            print('best epoch:', best_epoch)
            all_accs.append(tes_accs)
    np.savetxt(ofname + '_test_accs.csv', all_accs, fmt='%.6f', delimiter=',')

if __name__ == '__main__':
    files = sys.argv
    print(sys.argv)
    exp_path = files[1]
    fnames = []
    for ind, file in enumerate(files):
        if ind > 1:
            fnames.append(os.path.join(exp_path, file))
    print(fnames)
    if 'ana' in fnames[-1]:
        report_hyperpara(['dropout', 'hidden1', 'weight_decay'], fnames[-1])
    elif 'tune' in fnames[-1]:
        report_performance(fnames, 6, ['best_val'])
    else:
        report_curve(fnames, os.path.join(exp_path, 'gcns'))
