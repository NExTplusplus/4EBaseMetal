import copy
import json
import os
import numpy as np

def report_performance(fnames, repeats=5):
    paras = []
    perfs = []
    perf_count = 0
    val_perfs = []
    tes_perfs = []
    for fname in fnames:
        with open(fname) as fin:
            lines = fin.readlines()
            for ind, line in enumerate(lines):
                line = line.replace('\'', '"')
                if '\t\t {"seq":' in line or '\t\t{"seq":' in line:
                    para = json.loads(line)
                    paras.append(para)
                    perf_count = 0
                    val_perfs = []
                    tes_perfs = []
                if 'Best Valid' in line:
                    val_per_str = line.replace('Best Valid performance: ', '')
                    val_per = json.loads(val_per_str)
                    tes_per_str = lines[ind + 1].replace('\'', '"').replace('\tBest Test performance: ', '')
                    # print(tes_per_str)
                    tes_per = json.loads(tes_per_str)
                    val_perfs.append(val_per)
                    tes_perfs.append(tes_per)
                    perf_count += 1
                    if perf_count == repeats:
                        perfs.append([val_perfs, tes_perfs])

    assert len(paras) == len(perfs), 'length of paras and perfs do not ' \
                                     'mathch %d VS %d' % (len(paras), len(perfs))
    # sort
    for i in range(len(paras)):
        for j in range(i + 1, len(paras)):
            if _check_para_order(paras[i], paras[j]):
                temp = paras[i]
                paras[i] = paras[j]
                paras[j] = temp

                temp = perfs[i]
                perfs[i] = perfs[j]
                perfs[j] = temp

    best_val_per = perfs[0][0][0]
    best_tes_per = perfs[0][1][0]
    for i in range(len(paras)):
        if paras[i]['alp'] > 5:
            continue
        # acl aw & pred
        if paras[i]['eps'] < 0.00075:
            continue
        # if paras[i]['bet'] > 1.05:
        #     continue
        if not (
                abs(paras[i]['bet'] - 0.001) < 1e-8 or
                abs(paras[i]['bet'] - 0.005) < 1e-8 or
                abs(paras[i]['bet'] - 0.01) < 1e-8 or
                abs(paras[i]['bet'] - 0.05) < 1e-8 or
                abs(paras[i]['bet'] - 0.1) < 1e-8 # or
                # abs(paras[i]['bet'] - 0.5) < 1e-8 # or
                #abs(paras[i]['bet'] - 1.0) < 1e-8
                ):
            continue

        # if paras[i]['eps'] < 0.00075:
        #     continue
        # # if paras[i]['bet'] < 0.001 or paras[i]['bet'] > 1.05:
        # if paras[i]['bet'] > 0.15:
        #     continue
        print(paras[i])
        cur_val_per = _get_mean_perf(perfs[i][0], repeats=repeats)
        print('Valid:', cur_val_per)
        cur_tes_per = _get_mean_perf(perfs[i][1], repeats=repeats)
        print('Test:', cur_tes_per)
        if cur_val_per['acc'] > best_val_per['acc']:
            best_val_per = cur_val_per
            best_tes_per = cur_tes_per

    print('Best valid perf:', best_val_per)
    print('Best test perf:', best_tes_per)


    # write out
    print('------------')
    for i in range(len(paras)):
        print(paras[i])
        print('Valid:', perfs[i][0])
        print('Test:', perfs[i][1])

def _check_para_order(par1, par2):
    if not par1['seq'] == par2['seq']:
        return par1['seq'] > par2['seq']
    elif not par1['unit'] == par2['unit']:
        return par1['unit'] > par2['unit']
    elif not par1['alp'] == par2['alp']:
        return par1['alp'] > par2['alp']
    elif not par1['bet'] == par2['bet']:
        return par1['bet'] > par2['bet']
    elif not par1['eps'] == par2['eps']:
        return par1['eps'] > par2['eps']
    else:
        return False

def _get_mean_perf(cur_perfs, repeats):
    mean_per = copy.copy(cur_perfs[0])
    for j in range(1, repeats):
        for metric in mean_per.keys():
            mean_per[metric] = mean_per[metric] + cur_perfs[j][metric]
    for metric in mean_per.keys():
        mean_per[metric] = mean_per[metric] / repeats
    return mean_per

def report_loss(fnames, ofname):
    tra_objs = []
    val_losses = []
    for fname in fnames:
        with open(fname) as fin:
            cur_objs = []
            cur_losses = []
            lines = fin.readlines()
            for ind, line in enumerate(lines):
                line = line.replace('\'', '"')
                if '----->>>>>' in line:
                    # value of objective function on training set
                    tra_obj = float(lines[ind + 1].split(' ')[2])
                    val_loss = float(lines[ind + 2].split('Val loss: ')[1])
                    cur_objs.append(tra_obj)
                    cur_losses.append(val_loss)
            tra_objs.append(cur_objs)
            val_losses.append(cur_losses)
    np.savetxt(ofname + 'tra_loss.csv', tra_objs, fmt='%.6f', delimiter=',')
    np.savetxt(ofname + 'val_loss.csv', val_losses, fmt='%.6f', delimiter=',')

if __name__ == '__main__':
    exp_path = '/home/ffl/nus/MM/fintech/tweet_stock/exp'
    # fnames = [
    #     os.path.join(exp_path, 'tune_att_lstm_alp-1.log'),
    #     os.path.join(exp_path, 'tune_att_lstm.log'),
    #     # os.path.join(exp_path, 'tune_att_lstm_alp-10.log')
    # ]
    # report_performance(fnames)
    #
    # fnames = [
    #     os.path.join(exp_path, 'tune_att_lstm_fix.log')
    # ]
    # report_performance(fnames, repeats=1)

    # fnames = [
    #     os.path.join(exp_path, 'tune_pure_lstm_logloss.log')
    # ]
    # report_performance(fnames, repeats=5)

    # fnames = [
    #     os.path.join(exp_path, 'tune_pure_lstm_norm_ln.log')
    # ]
    # report_performance(fnames, repeats=5)

    # fnames = [
    #     os.path.join(exp_path, 'tune_aw_lstm_l5-u8.log')
    # ]
    # report_performance(fnames, repeats=5)

    # fnames = [
    #     os.path.join(exp_path, 'tune_pure_lstm.log')
    # ]
    # report_performance(fnames, repeats=5)

    # fnames = [
    #     os.path.join(exp_path, 'tune_att_lstm_norm_ln_logloss.log')
    # ]
    # report_performance(fnames, repeats=5)

    # fnames = [
    #     os.path.join(exp_path, 'tune_att_lstm_norm_ln.log')
    # ]
    # report_performance(fnames, repeats=5)

    # fnames = [
    #     os.path.join(exp_path, 'tune_aw_lstm_l10-u16.log')
    # ]
    # report_performance(fnames, repeats=5)

    # fnames = [
    #     os.path.join(exp_path, 'tune_aw_lstm_l5u8_lr1e-2'),
    #     os.path.join(exp_path, 'tune_aw_lstm_l5u8_lr1e-2_big_bet'),
    #     os.path.join(exp_path, 'tune_aw_lstm_l5u8_lr1e-2_small_eps')
    # ]
    # report_performance(fnames, repeats=5)

    # fnames = [
    #     os.path.join(exp_path, 'tune_aw_lstm_l5u4_lr1e-2_r3')
    # ]
    # report_performance(fnames, repeats=1)

    # fnames = [
    #     os.path.join(exp_path, 'tune_aw_lstm_l5u4_lr1e-2_pred_r3')
    # ]
    # report_performance(fnames, repeats=1)

    # fnames = [
    #     os.path.join(exp_path, 'tune_rand_pred_lstm_l5u4')
    # ]
    # report_performance(fnames, repeats=1)

    # exp_path = '/Users/ffl/Research/AWS/'
    #
    # fnames = [
    #     os.path.join(exp_path, 'kddcla_tune_att_lstm.log'),
    #     os.path.join(exp_path, 'kddcla_tune_att_lstm_1e-3.log')
    # ]
    # # fnames = [
    # #     os.path.join(exp_path, 'kddcla_tune_pure_lstm.log'),
    # #     os.path.join(exp_path, 'kddcla_tune_pure_lstm_1e-3.log')
    # # ]
    # report_performance(fnames, repeats=3)

    report_loss(
        ['no_shuffle_lr0.log', 'no_shuffle_lr1e-2.log', 'no_shuffle_lr1e-1.log'],
        'no_shuffle'
    )
    # report_loss(
    #     ['shuffle_lr0.log', 'shuffle_lr1e-2.log', 'shuffle_lr1e-1.log'],
    #     'shuffle'
    # )


    exp_path = '/home/ffl/nus/MM/fintech/tweet_stock/exp/kdd/run_aug_16/cpu'
    # fnames = [
    #     # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
    #     os.path.join(exp_path, 'kdd_tune_aw_lstm_l10u8')
    # ]
    # fnames = [
    #     # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
    #     os.path.join(exp_path, 'kdd_tune_pred_lstm_l10u8')
    # ]
    # fnames = [
    #     # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
    #     os.path.join(exp_path, 'kdd_tune_rand_pred_lstm_l10u8')
    # ]

    exp_path = '/home/ffl/nus/MM/fintech/tweet_stock/exp/kdd/run_aug_16/gpu'
    # fnames = [
    #     # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
    #     os.path.join(exp_path, 'kdd_tune_aw_lstm_l10u8')
    # ]
    # fnames = [
    #     # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
    #     os.path.join(exp_path, 'kdd_tune_pred_lstm_l10u8')
    # ]
    # fnames = [
    #     # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
    #     os.path.join(exp_path, 'kdd_tune_rand_pred_lstm_l10u8')
    # ]
    # fnames = [
    #     # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
    #     os.path.join(exp_path, 'kdd_tune_rand_pred_lstm_l15u16')
    # ]
    # fnames = [
    #     # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
    #     os.path.join(exp_path, 'kdd_tune_pred_lstm_l15u16')
    # ]
    # fnames = [
    #     # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
    #     os.path.join(exp_path, 'kdd_tune_aw_lstm_l15u16')
    # ]
    # report_performance(fnames, 1)

    exp_path = '/home/ffl/nus/MM/fintech/tweet_stock/exp/kdd/run_aug_16/cpu'
    # fnames = [
    #     # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
    #     os.path.join(exp_path, 'kdd_tune_rand_pred_lstm_l15u16')
    # ]
    # fnames = [
    #     # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
    #     os.path.join(exp_path, 'kdd_tune_pred_lstm_l15u16')
    # ]
    # fnames = [
    #     # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
    #     os.path.join(exp_path, 'kdd_tune_aw_lstm_l15u16')
    # ]

    exp_path = '/home/ffl/nus/MM/fintech/tweet_stock/exp/fea_lstm'
    fnames = [
        # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
        os.path.join(exp_path, 'log_r5')
    ]

    # exp_path = '/home/ffl/nus/MM/fintech/tweet_stock/exp/kdd/fea_lstm'
    # fnames = [
    #     # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
    #     os.path.join(exp_path, 'fea_lstm_r1')
    # ]

    exp_path = '/home/ffl/nus/MM/fintech/tweet_stock/exp/fea_lstm_r1'
    exp_path = '/home/ffl/nus/MM/fintech/tweet_stock/exp/m_lstm'
    fnames = [
        # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
        os.path.join(exp_path, 'log_mr')
    ]

    exp_path = '/home/ffl/nus/MM/fintech/tweet_stock/exp/kdd/fea_lstm'
    fnames = [
        # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
        os.path.join(exp_path, 'fea_lstm_mr1e-3'),
        os.path.join(exp_path, 'fea_lstm_mr5e-3'),
        os.path.join(exp_path, 'fea_lstm_mr1e-2'),
        os.path.join(exp_path, 'fea_lstm_mr5e-2'),
        os.path.join(exp_path, 'fea_lstm_mr1e-1'),
    ]
    exp_path = '/home/ffl/nus/MM/fintech/tweet_stock/exp/kdd/m_lstm'
    fnames = [
        # os.path.join(exp_path, 'kddour_tune_pure_lstm.log'),
        os.path.join(exp_path, 'm_lstm_mr1e-3'),
        os.path.join(exp_path, 'm_lstm_mr5e-3'),
        os.path.join(exp_path, 'm_lstm_mr1e-2'),
        os.path.join(exp_path, 'm_lstm_mr5e-2'),
        os.path.join(exp_path, 'm_lstm_mr1e-1'),
    ]

    # report_performance(fnames, 1)