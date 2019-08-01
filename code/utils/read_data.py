import pandas as pd
from copy import copy
import numpy as np
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


def process_missing_value_v3(X,cons_data=1):
    count = 0
    sta_ind = 0
    for i in range(X.shape[0]):
        if not X.iloc[i].isnull().values.any():
            count= count + 1
            if sta_ind==0:
                sta_ind = i
        else:
            count = 0
            sta_ind = 0
        if count == cons_data:
            break
    return X[sta_ind:].dropna()


def identify_col(col_name):
    '''
        Identify the feature on the level of OHLCV,OI (not including exchange and metal) and standardize the name of said feature across all exchange and metals
        Input
        col_name(str)	: the name of the column
        Output
        col_name(str)	: the name of the column that is standardized across all exchange and metals
    '''
    col_name = str.strip(col_name)
    if col_name in ["Open","Open.Price"]:
        # column represent Open Price
        return "Open"
    elif col_name in ["High","High.Price"]:
        # column represent High Price
        return "High"
    elif col_name in ["Low","Low.Price"]:
        # column represent Low Price
        return "Low"
    elif col_name in ["Close","Close.Price"]:
        # column represent Close Price
        return "Close"
    elif col_name in ["Open.Interest","Open Interest"] or col_name[6:] == "03":
        # column represent Open Interest
        return "OI"
    else:
        return col_name


def identify_exchange(fpath):
    '''
        Identify the exchange on which the asset is being traded
        Input
        fpath(str)		: filepath of csv file
        Output
        exchange(str)	: Name of exchange on which asset is being traded
    '''
    folders = fpath.split("/")
    if folders[-1] == "CNYUSD Curncy.csv":
        return ""
    for f in folders:
        if f in ["LME","DCE","SHFE","COMEX"]:
            return f
    return ""


def identify_metal(fpath):
    '''
        Identify the metal which is being referred to for the 6 metals.
        Input
        fpath(str)		: filepath of csv file
        Output
        metal(str)		: returns a short form of each of the metals
                          Copper => Co
                          Aluminium => Al
                          Nickel => Ni
                          Zinc => Zi
                          Tin => Ti
                          Lead = Le
    '''
    folders = fpath.split("/")
    f = folders[-1].strip(".csv")
    # consider special case of LME
    if f[0:3] == "LME":
        return f[3:5]
    if f[0:2] == "LM":
        f = f[2:4]
    # Aluminium case
    if f in ["AA","AH"]:
        return "Al"
    # Copper
    elif f in ["HG_lag1","CU","CA"]:
        return "Co"
    # Nickel
    elif f in ["XII","NI"]:
        return "Ni"
    #Zinc
    elif f in ["ZNA","ZS"]:
        return "Zi"
    #Tin
    elif f in ["XOO","SN"]:
        return "Ti"
    #Lead
    elif f in ["PBL","PB"]:
        return "Le"
    elif " Index" in f or " Curncy" in f:
        return "" 
    else:
        return f


def m2ar(matrix,lag = False):
    '''
        convert from rmatrix to pandas DataFrame (4E server only)
        Input
        matrix(rmatrix)		: rmatrix that holds data with index of date
        lag(bool)			: Boolean to decide whether lagging is required 
        Output
        time_series(df)		: Pandas DataFrame similar to output of read_single_csv
    '''
    from rpy2.robjects.packages import importr
    rbase = importr('base')
    rzoo = importr('zoo')
    arr = np.array(matrix)
    '''Get index'''
    idx = rbase.as_character(rzoo.index(matrix))
    '''Convert to pandas dataframe'''
    if not lag:
        time_series = pd.DataFrame(arr,index=idx)
    else:
        time_series = pd.DataFrame(arr[:-1],index = idx[1:])
    '''Assign proper column names'''
    time_series.columns = matrix.colnames
    return time_series


def read_data_NExT(config,start_date):
    '''
        Method to read data from csv files to pandas DataFrame
        Input
        config(dict)		: Dictionary with (fpath of csv, columns to read from csv) as key value pair
        start_date(str)		: Date that we start considering data
        Output
        data(df)			: A single Pandas Dataframe that holds all listed columns from their respective exchanges and for their respective metals
        LME_dates(list)		: list of dates on which LME has trading operations
    '''

    data = []
    LME_dates = None
    dates = []
    for fname in config:
        df = read_single_csv(fname,sel_col_names = config[fname])
        df = df.loc[start_date:]
        temp = copy(df.loc["2004-11-12":])
        data.append(df)
        # put in dates all dates that LME has operations (even if only there are metals that are not traded)
        if "LME" in fname:
            dates.append(temp.index)
    for date in dates:
        if LME_dates is None:
            LME_dates = date
        else:
            # Union of all LME dates
            LME_dates = LME_dates.union(date)
    return data, LME_dates.tolist()


def read_data_v5_4E(start_date):
    '''
        Method to read data from 4E database
        Input
        start_date(str)		: Date that we start considering data
        Output
        data(df)			: A single Pandas Dataframe that holds all listed columns from their respective exchanges and for their respective metals
        dates(list)			: list of dates on which LME has trading operations
    '''
    import rpy2.robjects as robjects
    robjects.r('.sourceAlfunction()')
    LME = robjects.r('''merge(getSecurity("LMCADY Comdty", start = "'''+start_date+'''"), getSecurity("LMAHDY Comdty", start = "'''+start_date+'''"),
                            getSecurity("LMPBDY Comdty", start = "'''+start_date+'''"), getSecurity("LMZSDY Comdty", start = "'''+start_date+'''"), 
                            getSecurity("LMNIDY Comdty", start = "'''+start_date+'''"), getSecurity("LMSNDY Comdty", start = "'''+start_date+'''"), 
                            getSecurityOHLCV("LMCADS03 Comdty", start = "'''+start_date+'''"), 
                            getSecurityOHLCV("LMPBDS03 Comdty", start = "'''+start_date+'''"), 
                            getSecurityOHLCV("LMNIDS03 Comdty", start = "'''+start_date+'''"), 
                            getSecurityOHLCV("LMSNDS03 Comdty", start = "'''+start_date+'''"), 
                            getSecurityOHLCV("LMZSDS03 Comdty", start = "'''+start_date+'''"), 
                            getSecurityOHLCV("LMAHDS03 Comdty", start = "'''+start_date+'''"))
                        ''')
    LME.colnames = robjects.vectors.StrVector(["LME_Co_Spot","LME_Al_Spot","LME_Le_Spot","LME_Zi_Spot","LME_Ni_Spot","LME_Ti_Spot"
                    ,"LME_Co_Open","LME_Co_High","LME_Co_Low","LME_Co_Close","LME_Co_Volume","LME_Co_OI"
                    ,"LME_Le_Open","LME_Le_High","LME_Le_Low","LME_Le_Close","LME_Le_Volume","LME_Le_OI"
                    ,"LME_Ni_Open","LME_Ni_High","LME_Ni_Low","LME_Ni_Close","LME_Ni_Volume","LME_Ni_OI"
                    ,"LME_Ti_Open","LME_Ti_High","LME_Ti_Low","LME_Ti_Close","LME_Ti_Volume","LME_Ti_OI"
                    ,"LME_Zi_Open","LME_Zi_High","LME_Zi_Low","LME_Zi_Close","LME_Zi_Volume","LME_Zi_OI"
                    ,"LME_Al_Open","LME_Al_High","LME_Al_Low","LME_Al_Close","LME_Al_Volume","LME_Al_OI"])
    COMEX_HG = robjects.r('''getGenOHLCV("HG", start = "'''+start_date+'''")''')
    COMEX_PA = robjects.r('''getGen("PA1S",zoom="'''+start_date+'''::")''')
    COMEX_PL = robjects.r('''getGenOHLCV("PL", start = "'''+start_date+'''")[,4]''')
    COMEX_GC = robjects.r('''getGenOHLCV("GC",start = "'''+start_date+'''")''')
    COMEX_SI = robjects.r('''getGenOHLCV("SI", start = "'''+start_date+'''")[,4:6]''')

    COMEX_HG.colnames = robjects.vectors.StrVector(["COMEX_Co_Open","COMEX_Co_High","COMEX_Co_Low","COMEX_Co_Close","COMEX_Co_Volume", "COMEX_Co_OI"])
    COMEX_PA.colnames = robjects.vectors.StrVector(["COMEX_PA_lag1_Close"])
    COMEX_PL.colnames = robjects.vectors.StrVector(["COMEX_PL_lag1_Close"])
    COMEX_GC.colnames = robjects.vectors.StrVector(["COMEX_GC_lag1_Open","COMEX_GC_lag1_High","COMEX_GC_lag1_Low","COMEX_GC_lag1_Close","COMEX_GC_lag1_Volume", "COMEX_GC_lag1_OI"])
    COMEX_SI.colnames = robjects.vectors.StrVector(["COMEX_SI_lag1_Close","COMEX_SI_lag1_Volume","COMEX_SI_lag1_OI"])


    DCE = robjects.r('''merge(getGenOHLCV("AKcl", start = "'''+start_date+'''"),getGenOHLCV("AEcl", start = "'''+start_date+'''"),
                        getGenOHLCV("ACcl", start = "'''+start_date+'''"))
                    ''')
    DCE.colnames = robjects.vectors.StrVector(["DCE_AK_Open","DCE_AK_High","DCE_AK_Low","DCE_AK_Close","DCE_AK_Volume","DCE_AK_OI",
                                            "DCE_AE_Open","DCE_AE_High","DCE_AE_Low","DCE_AE_Close","DCE_AE_Volume","DCE_AE_OI",
                                            "DCE_AC_Open","DCE_AC_High","DCE_AC_Low","DCE_AC_Close","DCE_AC_Volume","DCE_AC_OI"
                                            ])

    SHFE = robjects.r('''merge(getGenOHLCV("AAcl", start = "'''+start_date+'''"), getGenOHLCV("CUcl",start = "'''+start_date+'''")[,1:3],
                getGenOHLCV("CUcl",start = "'''+start_date+'''")[,5:6],getGenOHLCV("RTcl", start = "'''+start_date+'''")[,1:5],
            getDataAl("CNYUSD Curncy", start = "'''+start_date+'''"))
                        ''')

    SHFE.colnames = robjects.vectors.StrVector(["SHFE_Al_Open","SHFE_Al_High","SHFE_Al_Low","SHFE_Al_Close","SHFE_Al_Volume","SHFE_Al_OI",
                                            "SHFE_Co_Open","SHFE_Co_High","SHFE_Co_Low","SHFE_Co_Volume","SHFE_Co_OI",
                                                "SHFE_RT_Open","SHFE_RT_High","SHFE_RT_Low","SHFE_RT_Close","SHFE_RT_Volume", "CNYUSD"                                                    
                                            ]) 

    DXY = robjects.r('''getSecurity("DXY Curncy", start = "'''+start_date+'''")''')
    SX5E = robjects.r('''getSecurity("SX5E Index", start = "'''+start_date+'''")''')
    UKX = robjects.r('''getSecurity("UKX Index", start = "'''+start_date+'''")''')
    SPX = robjects.r('''getSecurity("SPX Index", start = "'''+start_date+'''")''')
    VIX = robjects.r('''getSecurity("VIX Index", start = "'''+start_date+'''")''')
    DXY.colnames = robjects.vectors.StrVector(["DXY"])
    SX5E.colnames = robjects.vectors.StrVector(["SX5E"])
    UKX.colnames = robjects.vectors.StrVector(["UKX"])
    SPX.colnames = robjects.vectors.StrVector(["SPX"])
    VIX.colnames = robjects.vectors.StrVector(["VIX"])

    index = robjects.r('''merge(getSecurity("HSI Index", start = "'''+start_date+'''"),getSecurity("NKY Index", start = "'''+start_date+'''"),
                                getSecurity("SHCOMP Index", start = "'''+start_date+'''"), getSecurity("SHSZ300 Index", start = "'''+start_date+'''")
                        )''')
    index.colnames = robjects.vectors.StrVector(["HSI","NKY","SHCOMP","SHSZ300"])

    LME = m2ar(LME)
    COMEX_PA = m2ar(COMEX_PA, lag = True)
    COMEX_HG = m2ar(COMEX_HG, lag = True)
    COMEX_GC = m2ar(COMEX_GC, lag = True)
    COMEX_PL = m2ar(COMEX_PL, lag = True)
    COMEX_SI = m2ar(COMEX_SI, lag = True)
    DCE = m2ar(DCE)
    SHFE = m2ar(SHFE)
    DXY = m2ar(DXY,lag = True)
    SX5E = m2ar(SX5E,lag = True)
    UKX = m2ar(UKX,lag = True)
    SPX = m2ar(SPX,lag = True)
    VIX = m2ar(VIX,lag = True)
    index = m2ar(index)
    LME_temp = copy(LME.loc['2004-11-12':])
    dates = LME_temp.index

    data = LME.join([DCE,SHFE,index,COMEX_HG,COMEX_GC,COMEX_SI,COMEX_PA,COMEX_PL,DXY,SX5E,UKX,SPX,VIX], how = "outer")
    return data, dates

