import pandas as pd

'''
parameters:
fname (str): the file going to be read. 
sel_col_names [str]: the columns to be returned in the exactly same order

returns: 
X (a pandas DataFrame): the data in the input file
'''
def read_single_csv(fname, sel_col_names = None):
    ans = pd.DataFrame()
    X = pd.read_csv(fname, index_col=0)
    exchange = identify_exchange(fname)
    metal = identify_metal(fname)
    for col in sel_col_names:
        
        if col[0:2] == "LM" and col[4:6] == "DY":
            col_name = str.strip('_'.join((exchange,metal,"Spot"))).strip("_")
        else:
            col_name = str.strip('_'.join((exchange,metal,identify_col(col)))).strip("_")
        ans[col_name] = X[col]
    # if sel_col_names == "All":
    #     return X
    # else:
    #     available_col = X.columns
    #     choosen_col =[]
    #     missing_col = []
    #     for col_name in sel_col_names:
    #         if col_name in available_col:
    #             choosen_col.append(col_name)
    #         else:
    #             missing_col.append(col_name)
    #     if len(missing_col)!=0:
    #         print("Available columns are following: "+str(list(available_col)))
    #         print("The following columns are missing: " + str(missing_col))
    # return X[choosen_col]
    print(col_name)
    return ans
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

def process_missing_value_v2(X):
    return X.dropna()

# See "Deal with NA value" in google drive/ data cleaning file for more explanations
# "X" is the dataframe we want to process and "cons_data" is number of consecutive complede data we need to have 
def process_missing_value_v3(X,cons_data):
    count = 0
    sta_ind = 0
    for i in range(X.shape[0]):
        if not X.iloc[i].isnull().values.any():
            count= count +1
            if sta_ind!=0:
                sta_ind = i
        else:
            count = 0
            sta_ind =0
        if count == cons_data:
            break
        
    return X[sta_ind:].dropna()

def identify_col(col_name):
    col_name = str.strip(col_name)
    if col_name in ["Open","Open.Price"]:
        return "Open"
    elif col_name in ["High","High.Price"]:
        return "High"
    elif col_name in ["Low","Low.Price"]:
        return "Low"
    elif col_name in ["Close","Close.Price"]:
        return "Close"
    elif col_name in ["Open.Interest","Open Interest"] or col_name[6:] == "03":
        print(col_name)
        return "OI"
    else:
        return col_name

def identify_exchange(fpath):
    folders = fpath.split("/")
    if folders[-1] == "CNYUSD Curncy.csv":
        return ""
    for f in folders:
        if f in ["LME","DCE","SHFE","COMEX"]:
            return f
    return ""

def identify_metal(fpath):
    folders = fpath.split("/")
    f = folders[-1].strip(".csv")
    if f[0:3] == "LME":
        return f[3:5]
    if f[0:2] == "LM":
        f = f[2:4]
    if f in ["AA","AH"]:
        return "Al"
    elif f in ["HG","CU","CA"]:
        return "Co"
    elif f in ["XII","NI"]:
        return "Ni"
    elif f in ["ZNA","ZS"]:
        return "Zi"
    elif f in ["XOO","SN"]:
        return "Ti"
    elif f in ["PBL","PB"]:
        return "Le"
    elif " Index" in f or " Curncy" in f:
        return "" 
    else:
        return f



