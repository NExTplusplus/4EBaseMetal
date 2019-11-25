from copy import copy
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def _read_spot_price(fname, steps):
    X = pd.read_csv(fname, index_col=0)
    X = X.to_numpy().squeeze()
    print(np.max(X), np.min(X), X.shape)
    label = np.ones([len(X) - steps], dtype=int)
    for i in range(steps, len(X)):
        label[i - steps] = 1 if X[i] > X[i - steps] else -1
    return X[:len(X) - steps], label


def _get_single_cm_feature(price_slice, slice_id, lag):
    price_slice = copy(price_slice)
    oc_line = copy(price_slice)
    step = (price_slice[-1] - price_slice[0]) / (lag - 1)
    for i in range(lag):
        oc_line[i] = price_slice[0] + step * i

    # check special day
    special_ind = []
    # for i in range(1, lag):
    #     if abs(oc_line[i] - price_slice[i]) < 0.01 and \
    #             abs(oc_line[i - 1] - price_slice[i - 1]) < 0.01:
    #         print('two continues days with equal values', i)
    #         special_ind.append(i)
    for i in range(1, lag - 1):
        if abs(oc_line[i] - price_slice[i]) < 0.01:
            # print(slice_id, 'middle days with equal values', i)
            special_ind.append(i)
    if len(special_ind) > 0:
        oc_line = np.delete(oc_line, special_ind)
        price_slice = np.delete(price_slice, special_ind)
        lag = len(oc_line)
        # print(len(oc_line), len(price_slice), lag)

    # the first day
    cross_type = []
    cross_ind = []
    for i in range(1, lag):
        if abs(oc_line[i] - price_slice[i]) >= 0.01:
            cross_ind.append(i)
            if oc_line[i] < price_slice[i]:
                cross_type.append(1)
            else:
                cross_type.append(-1)
            break

    for i in range(cross_ind[-1], lag):
        if oc_line[i - 1] - price_slice[i - 1] < -0.01:
            if oc_line[i] - price_slice[i] >= -1e-9:
                cross_ind.append(i)
                cross_type.append(-1)

        if oc_line[i - 1] - price_slice[i - 1] > 0.01:
            if oc_line[i] - price_slice[i] <= 1e-9:
                cross_ind.append(i)
                cross_type.append(1)

    # check continues cross points
    for i in range(1, len(cross_type)):
        if cross_type[i - 1] == cross_type[i]:
            print(slice_id, 'something wrong:', cross_ind[i - 1], cross_ind[i])
            print(price_slice[cross_ind[i - 1] - 1], price_slice[cross_ind[i - 1]])
            print(oc_line[cross_ind[i - 1] - 1], oc_line[cross_ind[i - 1]])
            print(price_slice[cross_ind[i] - 1], price_slice[cross_ind[i]])
            print(oc_line[cross_ind[i] - 1], oc_line[cross_ind[i]])

    # get peaks
    peak_ind = []
    peak_type = []
    for i in range(1, len(cross_type)):
        # minimum
        if cross_type[i - 1] == -1 and cross_type[i] == 1:
            peak_ind.append(cross_ind[i - 1] +
                            np.argmin(price_slice[cross_ind[i - 1]: cross_ind[i]]))
            peak_type.append(-1)

        # maximum
        if cross_type[i - 1] == 1 and cross_type[i] == -1:
            peak_ind.append(cross_ind[i - 1] +
                            np.argmax(price_slice[cross_ind[i - 1]: cross_ind[i]]))
            peak_type.append(1)
    return {'cross_type': cross_type, 'cross_ind': cross_ind,
            'peak_type': peak_type, 'peak_ind': peak_ind, 'step': step}


def _get_cross_momentum_features(X, lag):
    cm_feas = []
    for i in range(lag, len(X) + 1):
        # print(i)
        cm_feas.append(_get_single_cm_feature(X[i - lag: i], i, lag))
    return cm_feas


def evaluate_prediction(label, predictions, granularity=125, rollings=10):
    for pred in predictions:
        assert len(pred) == len(label), 'length mismatch'
    performance = np.zeros([rollings, len(predictions)], dtype=float)
    for i in range(rollings):
        star_ind = granularity * (i + 1) * -1
        end_ind = granularity * i * -1
        # print('round:', i, star_ind, end_ind)
        if i == 0:
            label_rol = label[star_ind:]
        else:
            label_rol = label[star_ind:end_ind]

        for j, pred in enumerate(predictions):
            if i == 0:
                pred_rol = pred[star_ind:]
            else:
                pred_rol = pred[star_ind:end_ind]
            # accs.append(accuracy_score(label_rol, pred_rol))
            performance[i][j] = accuracy_score(label_rol, pred_rol)
    print(np.sum(performance, axis=0) / rollings)


def evaluate_prediction_all(label, predictions):
    for pred in predictions:
        assert len(pred) == len(label), 'length mismatch'
    performance = np.zeros([len(predictions)], dtype=float)
    for i, pred in enumerate(predictions):
        performance[i] = accuracy_score(label, pred)
    print(performance)


def get_prediction(ground_truth, steps, lag):
    X, label = _read_spot_price(ground_truth, steps)
    cm_feas = _get_cross_momentum_features(X, lag)
    label = label[lag - 1:]
    assert len(cm_feas) == len(label), 'length mismatch'

    # get prediction
    pred_trend = copy(label)
    pred_trend_ratio = copy(label)
    step_length = np.zeros(pred_trend.shape, dtype=float)
    cross_num = np.zeros(pred_trend.shape, dtype=int)
    for i, cm_fea in enumerate(cm_feas):
        # simplest rule, just the latest trend
        cross_num[i] = len(cm_fea['cross_type'])
        step_length[i] = cm_fea['step']
        last_cross_type = cm_fea['cross_type'][-1]
        last_period = cm_fea['cross_ind'][-1] - cm_fea['cross_ind'][-2]
        peak_cross_dis = cm_fea['cross_ind'][-1] - cm_fea['peak_ind'][-1]
        peak_period_ratio = peak_cross_dis / last_period
        pred_trend[i] = last_cross_type
        if peak_period_ratio > 0.5:
            pred_trend_ratio[i] = -1 * last_cross_type
        else:
            pred_trend_ratio[i] = last_cross_type

    return label, [pred_trend, pred_trend_ratio], [step_length, cross_num]


def filter_cross_num(label, predictions, cross_num, thres):
    select_inds = []
    for i, cross in enumerate(cross_num):
        if cross < thres:
            continue
        else:
            select_inds.append(i)
    label = label[select_inds]
    for i in range(len(predictions)):
        predictions[i] = predictions[i][select_inds]
    return label, predictions


def test_metal(args):
    label, predictions, filters = get_prediction(args.ground_truth,
                                                 args.steps, args.lag)

    # # observe cross num
    # counts = np.zeros(31, dtype=int)
    # for cross_num in filters[-1]:
    #     if cross_num < 30:
    #         counts[cross_num] = counts[cross_num] + 1
    #     else:
    #         counts[30] = counts[30] + 1
    # for i in range(31):
    #     print(i, counts[i])
    # print('before filtering:', len(label))
    # evaluate_prediction(label, predictions)
    #
    # label_fil, predictions_fil = filter_cross_num(copy(label), copy(predictions),
    #                                               filters[-1], args.lag / 10)
    # print('after filtering {}: {}'.format(args.lag / 10, len(label_fil)))
    # evaluate_prediction(label_fil, predictions_fil)
    #
    # label_fil, predictions_fil = filter_cross_num(copy(label),
    #                                               copy(predictions),
    #                                               filters[-1], args.lag / 20)
    # print('after filtering {}: {}'.format(args.lag / 20, len(label_fil)))
    # evaluate_prediction(label_fil, predictions_fil)

    print('before filtering:', len(label))
    evaluate_prediction_all(label, predictions)

    label_fil, predictions_fil = filter_cross_num(copy(label),
                                                  copy(predictions),
                                                  filters[-1],
                                                  args.lag / 10)
    print('after filtering {}: {}'.format(args.lag / 10, len(label_fil)))
    evaluate_prediction_all(label_fil, predictions_fil)

    label_fil, predictions_fil = filter_cross_num(copy(label),
                                                  copy(predictions),
                                                  filters[-1],
                                                  args.lag / 20)
    print('after filtering {}: {}'.format(args.lag / 20, len(label_fil)))
    evaluate_prediction_all(label_fil, predictions_fil)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyze month-K and year-K cross point')
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                        type=str, default="./data/Financial Data/LME/LMCADY.csv")
    parser.add_argument('-s', '--steps', type=int, default=5,
                        help='steps in the future to be predicted')
    parser.add_argument(
        '-sou', '--source', help='source of data', type=str, default="NExT",
        choices=["NExT", "4E"],
    )
    parser.add_argument(
        '-l', '--lag', type=int, default=250, help='lag'
    )
    parser.add_argument('-o', '--action', type=str, default='sep',
                        choices=["sep", "all"], help='sep, all')
    args = parser.parse_args()
    # DEBUG
    # args.ground_truth = '/home/ffl/nus/MM/fintech/4e_base_metal/4EBaseMetal/data/Financial Data/LME/LMCADY.csv'

    gts = [
        './data/Financial Data/LME/LMCADY.csv',
        './data/Financial Data/LME/LMZSDY.csv',
        './data/Financial Data/LME/LMSNDY.csv',
        './data/Financial Data/LME/LMPBDY.csv',
        './data/Financial Data/LME/LMNIDY.csv',
        './data/Financial Data/LME/LMAHDY.csv'

    ]
    if args.action == 'all':
        for gt in gts:
            args.ground_truth = gt
            print(args)
            test_metal(args)
    elif args.action == 'sep':
        test_metal(args)