from copy import copy
import numpy as np

'''
parameters:
X (2d numpy array): the data to be normalized

returns:
X_norm (2d numpy array): note that the dimension of X_norm is different from
    that of X since it less one row (cannot calculate return for the 1st day).
'''
def log_1d_return(X):
    # assert type(X) == np.ndarray, 'only 2d numpy array is accepted'
    if type(X) == np.ndarray:
        return np.log(np.true_divide(X[1:, :], X[:-1, :]))
    else:
        X.values[1:, :] = np.log(np.true_divide(X.values[1:, :],
                                                X.values[:-1, :]))
    return X