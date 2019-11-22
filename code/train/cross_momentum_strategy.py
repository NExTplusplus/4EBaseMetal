from copy import copy
import argparse
import numpy as np
import pandas as pd


def _read_spot_price(fname, steps):
    X = pd.read_csv(fname, index_col=0)
    X = X.to_numpy().squeeze()
    print(np.max(X), np.min(X), X.shape)
    label = np.ones([len(X) - steps], dtype=int)
    for i in range(steps, len(X)):
        label[i - steps] = 1 if X[i] > X[i - steps] else -1
    return X[:len(X) - steps], label


def _get_single_cm_feature(price_slice, lag):
    price_slice = copy(price_slice)
    oc_line = copy(price_slice)
    step = (price_slice[-1] - price_slice[0]) / (lag - 1)
    for i in range(lag):
        oc_line[i] = price_slice[0] + step * i

    # check special day
    special_ind = []
    for i in range(1, lag):
        if abs(oc_line[i] - price_slice[i]) < 0.01 and \
                abs(oc_line[i - 1] - price_slice[i - 1]) < 0.01:
            print('two continues days with equal values', i)
            special_ind.append(i)
    if len(special_ind) > 0:
        oc_line = np.delete(oc_line, special_ind)
        price_slice = np.delete(price_slice, special_ind)

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
            print('something wrong:', cross_ind[i - 1], cross_ind[i])

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
            'peak_type': peak_type, 'peak_ind': peak_ind}


def _get_cross_momentum_features(X, lag):
    cm_feas = []
    for i in range(lag, len(X) + 1):
        print(i)
        cm_feas.append(_get_single_cm_feature(X[i - lag: i], lag))
    return cm_feas


def get_prediction(args):
    X, label = _read_spot_price(args.ground_truth, args.steps)
    cm_feas = _get_cross_momentum_features(X, args.lag)
    label = label[args.lag - 1:]
    assert len(cm_feas) == len(label), 'length mismatch'

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
        '-l','--lag', type=int, default=250, help='lag'
    )
    args = parser.parse_args()
    # DEBUG
    args.ground_truth = '/Users/ffl/Research/4EBaseMetal/data/Financial Data/LME/LMCADY.csv'
    print(args)
    get_prediction(args)