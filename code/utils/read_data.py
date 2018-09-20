import pandas as pd


'''
parameters:
fname (str): the file going to be read. 
sel_col_names [str]: the columns to be returned in the exactly same order

returns: 
X (a pandas DataFrame): the data in the input file
'''
def read_sigle_csv(fname, sel_col_names):
    X = pd.read_csv(fname)
    for col_name in sel_col_names:
        # select the column with name of col_name
        pass
    return X