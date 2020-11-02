import pandas as pd
from copy import copy
import numpy as np


def read_single_csv(fname, sel_col_names = None):
    '''
    parameters:
    fname (str): the file going to be read. 
    sel_col_names [str]: the columns to be returned in the exactly same order

    returns: 
    X (a pandas DataFrame): the data in the input file
    '''

    ans = pd.DataFrame()
    time_series = pd.read_csv(fname, index_col=0)
    ans = time_series[sel_col_names]
    return ans


def read_data_NExT(config,start_date):
    '''
        Method to read data from csv files to pandas DataFrame
        Input:  config(dict)		: Dictionary with (fpath of csv, columns to read from csv) as key value pair
                start_date(str)		: Date that we start considering data
        Output: data(df)			: A single Pandas Dataframe that holds all listed columns from their respective exchanges and for their respective metals
                LME_dates(list)		: list of dates on which LME has trading operations
        
    '''

    time_series = []
    LME_dates = None
    dates = []

    #generate the data column names (here they should be similarly sorted)
    colnames = generate_required_columns(config)

    for fname in config:
        df = read_single_csv(fname,sel_col_names = config[fname])
        df = df.loc[start_date:]
        temp = copy(df.loc[start_date:])
        time_series.append(df)
        # put in dates all dates that LME has operations (even if only there are metals that are not traded)
        if "LME" in fname and "1GQ" not in config[fname][0]:
            dates.append(temp.index)
    for date in dates:
        if LME_dates is None:
            LME_dates = date
        else:
            # Union of all LME dates
            LME_dates = LME_dates.union(date)
    time_series = pd.concat(time_series, axis = 1, sort = True)
    time_series.columns = colnames
    time_series.to_csv("ts.csv")
    return time_series, LME_dates.tolist()

#formatting of names of data column to be unified as (exchange)_(metal)_(feature)
def generate_required_columns(config):
    """
        input   :   config   (dic)   :dictionary from data configuration file
        output  :   names   (list)   :list of columns names that should be loaded
    """
    colnames = []
    for fname in config:
        #identify exchange from filepath
        exchange = identify_exchange(fname)

        #identify metal from filepath
        metal = identify_metal(fname)

        #identify feature
        for col in config[fname]:
            if col[0:2] == "LM" and col[4:6] == "DY":
                col_name = str.strip('_'.join((exchange,metal,"Spot"))).strip("_")
            else:
                col_name = str.strip('_'.join((exchange,metal,identify_feature(col)))).strip("_")
            colnames.append(col_name)
    
    return colnames

def identify_feature(feature_name):
    '''
        Identify the feature on the level of OHLCV,OI (not including exchange and metal) and standardize the name of said feature across all exchange and metals
        Input
        feature_name(str)	: the name of the column
        Output
        feature_name(str)	: the name of the column that is standardized across all exchange and metals
    '''
    feature_name = str.strip(feature_name)
    if feature_name in ["Open","Open.Price"]:
        # featureumn represent Open Price
        return "Open"
    elif feature_name in ["High","High.Price"]:
        # featureumn represent High Price
        return "High"
    elif feature_name in ["Low","Low.Price"]:
        # featureumn represent Low Price
        return "Low"
    elif feature_name in ["Close","Close.Price"]:
        # featureumn represent Close Price
        return "Close"
    elif feature_name in ["Open.Interest","Open Interest"] or feature_name[6:] == "03":
        # featureumn represent Open Interest
        return "OI"
    elif "LE" in feature_name:
        return "Demand"
    elif "LS" in feature_name:
        return "Supply"
    else:
        return feature_name


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
    if f in ["AA","AH","LSAH","LEAH"]:
        return "Al"
    # Copper
    elif f in ["HG_lag1","CU","CA","LECA","LSCA"]:
        return "Co"
    # Nickel
    elif f in ["XII","NI","LENI","LSNI"]:
        return "Ni"
    #Zinc
    elif f in ["ZNA","ZS","LEZS","LSZS"]:
        return "Zi"
    #Tin
    elif f in ["XOO","SN","LESN","LSSN"]:
        return "Ti"
    #Lead
    elif f in ["PBL","PB","LEPB","LSPB"]:
        return "Le"
    elif " Index" in f or " Curncy" in f:
        return "" 
    elif "METF" in f:
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
    #Get index
    idx = rbase.as_character(rzoo.index(matrix))
    #Convert to pandas dataframe
    if not lag:
        time_series = pd.DataFrame(arr,index=idx)
    else:
        time_series = pd.DataFrame(arr[:-1],index = idx[1:])
    #Assign proper column names
    time_series.columns = matrix.colnames
    return time_series


def read_data_4E(config,start_date):
    '''
        Method to read all data as provided by 4E on the 4E server (for online testing)
        Input
        config(dic)         : data configuration dictionary
        start_date(str)		: Date that we start considering data
        Output
        data(df)			: A single Pandas Dataframe that holds all listed columns from their respective exchanges and for their respective metals
        dates(list)			: list of dates on which LME has trading operations
    '''

    #generate the data column names to be extracted after complete reading of data from 4E server
    colnames = generate_required_columns(config)

    import rpy2.robjects as robjects

    #load 4E code base to extract data
    robjects.r('.sourceQlib()')

    #load LME data and generate column names
    LME = robjects.r('''merge(getSecurity(c("LMCADY Comdty","LMAHDY Comdty","LMPBDY Comdty","LMZSDY Comdty","LMNIDY Comdty","LMSNDY Comdty"), start = "'''+start_date+'''"), 
                            getSecurityOHLCV(c("LMCADS03 Comdty","LMPBDS03 Comdty","LMNIDS03 Comdty","LMSNDS03 Comdty","LMZSDS03 Comdty","LMAHDS03 Comdty"), start = "'''+start_date+'''")
                            )
                        ''')
    LME.colnames = robjects.vectors.StrVector(["LME_Co_Spot","LME_Al_Spot","LME_Le_Spot","LME_Zi_Spot","LME_Ni_Spot","LME_Ti_Spot"
                    ,"LME_Co_Open","LME_Co_High","LME_Co_Low","LME_Co_Close","LME_Co_Volume","LME_Co_OI"
                    ,"LME_Le_Open","LME_Le_High","LME_Le_Low","LME_Le_Close","LME_Le_Volume","LME_Le_OI"
                    ,"LME_Ni_Open","LME_Ni_High","LME_Ni_Low","LME_Ni_Close","LME_Ni_Volume","LME_Ni_OI"
                    ,"LME_Ti_Open","LME_Ti_High","LME_Ti_Low","LME_Ti_Close","LME_Ti_Volume","LME_Ti_OI"
                    ,"LME_Zi_Open","LME_Zi_High","LME_Zi_Low","LME_Zi_Close","LME_Zi_Volume","LME_Zi_OI"
                    ,"LME_Al_Open","LME_Al_High","LME_Al_Low","LME_Al_Close","LME_Al_Volume","LME_Al_OI"
                    ])
    
    #load LME Supply and Demand Data and generate column names
    LME_SD = robjects.r('''getTickersBaseMetalsMacro(c("LEAH","LECA","LENI","LEPB","LESN","LEZS","LSAH","LSCA","LSNI","LSPB","LSSN","LSZS"), asPrice = TRUE, zoom = "'''+start_date+'''::")
                            ''')
    LME_SD.colnames = robjects.vectors.StrVector([
                    "LME_Al_Demand","LME_Co_Demand","LME_Ni_Demand","LME_Le_Demand","LME_Ti_Demand","LME_Zi_Demand"
                    ,"LME_Al_Supply","LME_Co_Supply","LME_Ni_Supply","LME_Le_Supply","LME_Ti_Supply","LME_Zi_Supply"])


    #load COMEX data and generate column names
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


    #load DCE data and generate column names
    DCE = robjects.r('''merge(getGenOHLCV("AKcl", start = "'''+start_date+'''"),getGenOHLCV("AEcl", start = "'''+start_date+'''"),
                        getGenOHLCV("ACcl", start = "'''+start_date+'''"))
                    ''')
    DCE.colnames = robjects.vectors.StrVector(["DCE_AK_Open","DCE_AK_High","DCE_AK_Low","DCE_AK_Close","DCE_AK_Volume","DCE_AK_OI",
                                            "DCE_AE_Open","DCE_AE_High","DCE_AE_Low","DCE_AE_Close","DCE_AE_Volume","DCE_AE_OI",
                                            "DCE_AC_Open","DCE_AC_High","DCE_AC_Low","DCE_AC_Close","DCE_AC_Volume","DCE_AC_OI"
                                            ])

    #load SHFE data and generate column names
    SHFE = robjects.r('''merge(getGenOHLCV("AAcl", start = "'''+start_date+'''"), getGenOHLCV("CUcl",start = "'''+start_date+'''")[,1:3],
                getGenOHLCV("CUcl",start = "'''+start_date+'''")[,5:6],getGenOHLCV("RTcl", start = "'''+start_date+'''")[,1:5],
            getDataAl("CNYUSD Curncy", start = "'''+start_date+'''"))
                        ''')

    SHFE.colnames = robjects.vectors.StrVector(["SHFE_Al_Open","SHFE_Al_High","SHFE_Al_Low","SHFE_Al_Close","SHFE_Al_Volume","SHFE_Al_OI",
                                            "SHFE_Co_Open","SHFE_Co_High","SHFE_Co_Low","SHFE_Co_Volume","SHFE_Co_OI",
                                                "SHFE_RT_Open","SHFE_RT_High","SHFE_RT_Low","SHFE_RT_Close","SHFE_RT_Volume", "CNYUSD"                                                    
                                            ]) 

    #load index data and generate column names
    DXY = robjects.r('''getSecurity("DXY Curncy", start = "'''+start_date+'''")''')
    SX5E = robjects.r('''getSecurity("SX5E Index", start = "'''+start_date+'''")''')
    UKX = robjects.r('''getSecurity("UKX Index", start = "'''+start_date+'''")''')
    SPX = robjects.r('''getSecurity("SPX Index", start = "'''+start_date+'''")''')
    VIX = robjects.r('''getSecurity("VIX Index", start = "'''+start_date+'''")''')
    index = robjects.r('''getSecurity(c("HSI Index","NKY Index","SHCOMP Index","SHSZ300 Index"), start = "'''+start_date+'''")
                        ''')
    DXY.colnames = robjects.vectors.StrVector(["DXY"])
    SX5E.colnames = robjects.vectors.StrVector(["SX5E"])
    UKX.colnames = robjects.vectors.StrVector(["UKX"])
    SPX.colnames = robjects.vectors.StrVector(["SPX"])
    VIX.colnames = robjects.vectors.StrVector(["VIX"])
    index.colnames = robjects.vectors.StrVector(["HSI","NKY","SHCOMP","SHSZ300"])

    #load Third Party data and generate column names
    Third_Party = robjects.r('''merge(getTickersBaseMetalsForecast(c("METFA3 1GQ","METFC3 1GQ","METFN3 1GQ","METFL3 1GQ","METFT3 1GQ","METFZ3 1GQ"), asPrice = TRUE, zoom = "'''+start_date+'''::")
                                        )'''
                                        )
    Third_Party.colnames = robjects.vectors.StrVector(["METFA3 1GQ","METFC3 1GQ","METFN3 1GQ","METFL3 1GQ","METFT3 1GQ","METFZ3 1GQ"])


    #convert all R Dataframes to pandas Dataframes with corresponding lag
    LME = m2ar(LME)
    LME_SD = m2ar(LME_SD)
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
    Third_Party = m2ar(Third_Party)

    #generate dates that LME has trading operation
    LME_temp = copy(LME.loc['2004-11-12':])
    dates = LME_temp.index.values.tolist()

    #merge data
    time_series = LME.join([LME_SD,DCE,SHFE,index,COMEX_HG,COMEX_GC,COMEX_SI,COMEX_PA,COMEX_PL,DXY,SX5E,UKX,SPX,VIX,Third_Party], how = "outer")
    time_series = time_series[colnames]
    return time_series, dates
