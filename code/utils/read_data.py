import pandas as pd


'''
parameters:
fname (str): the file going to be read. 
sel_col_names [str]: the columns to be returned in the exactly same order

returns: 
X (a pandas DataFrame): the data in the input file
'''
def read_single_csv(fname, sel_col_names):
    X = pd.read_csv(fname, index_col=0)
    # for col_name in sel_col_names:
    #     # select the column with name of col_name
    #     pass
    # return X
    return X[sel_col_names]


'''

'''
def merge_data_frame(X, Y):
    return pd.concat([X, Y], axis=1, sort=True)


'''

'''
def process_missing_value(X):
    sta_ind = 0
    for i in range(X.shape[0]):
        if X.iloc[i].isnull().values.any():
            sta_ind = i + 1
    return X[sta_ind:], sta_ind