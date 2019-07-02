import os
import sys
import pandas as pd
from copy import copy
import numpy as np
import argparse

sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],"..","..")))

def count_consecutive(df,leftout=[]):
  dc = {'count':[1]}
  mx = 1
  for col in df.columns:
    if col in leftout:
      continue
    ser = copy(df[col])
    for i in range(len(ser)):
      if np.isnan(ser[i]):
        if ser[i-1] >= 0:
          ser[i] = -1
        elif ser[i-1] < 0:
          ser[i] = ser[i-1]-1
    j = 1
    while True:
      if j == 1:
        dc[col] = [sum(ser == -j)]
      else:
        dc[col].append(sum(ser == -j))
      if j > mx:
        mx = j
      j+=1
      if sum(ser== -j) == 0:
        break
  dc['count'] = list(range(1,mx+1))
  for key in dc.keys():
    if key == 'count':
      continue
    while (len(dc[key])< mx):
      dc[key].append(0)
    for i in range(mx-1):
      dc[key][i] = dc[key][i] - dc[key][i+1]
  return pd.DataFrame(dc)

if __name__ == '__main__':
  desc = 'Data Loss'
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument(
    '--action', '-o', type=str,
    help='action that should be taken',
    default='exchange'
  )

  args = parser.parse_args()
  if args.action == "exchange":
    with open(os.path.join(sys.path[0],"..","..","i1.csv")) as f:
      df = pd.read_csv(f,header = 0, index_col = 0)
      COMEX = None
      LME = None
      SHFE = None
      DCE = None
      columns = copy(df.columns.values.tolist())
      for col in df.columns:
        hit = False
        if "COMEX" in col:
          hit = True
          if COMEX is None:
            COMEX = pd.DataFrame(df[col], index = df.index)
          else:
            COMEX = COMEX.join(df[col], how = "outer")
        elif "LME" in col:
          hit = True
          if LME is None:
            LME = pd.DataFrame(df[col], index = df.index)
          else:
            LME = LME.join(df[col], how = "outer")
        elif "SHFE" in col:
          hit = True
          if SHFE is None:
            SHFE = pd.DataFrame(df[col], index = df.index)
          else:
            SHFE = SHFE.join(df[col], how = "outer")
        elif "DCE" in col:
          hit = True
          if DCE is None:
            DCE = pd.DataFrame(df[col], index = df.index)
          else:
            DCE = DCE.join(df[col], how = "outer")
        if hit:
          columns.remove(col)
      columns.remove("CNYUSD")
      others =pd.DataFrame(df[["DXY","HSI","NKY","SHCOMP","SHSZ300","SPX","SX5E","UKX","VIX"]])
      all_exchange = None
      with open("../../data_loss_1.csv","w") as out:
        to_be_compared = None
        for exchange in [LME,COMEX,SHFE,DCE,others]:
          ex = exchange.columns.values.tolist()[0].split("_")[0]
          out.write(exchange.columns.values.tolist()[0].split("_")[0]+",All\n")
          exchange = exchange[sorted(exchange.columns)]
          exchange = exchange[exchange.index.get_loc("2003-11-12"):]
          any_nan = exchange.transpose(copy = True).isnull().any()
          all_nan = exchange.transpose(copy=True).isnull().all()
          exchange = exchange.applymap(np.isnan)
          exchange[ex +" Any"] = any_nan
          exchange[ex +" All"] = all_nan
          temp = copy(exchange)
          prev = 0
          for i in temp.index:
            if not exchange.loc[i,ex +' Any']:
              exchange = exchange.drop(i, axis = 0)
            else:
              if to_be_compared is not None:
                if i in to_be_compared:
                  curr = to_be_compared.get_loc(i)
                  for j in range(curr - prev-1):
                    out.write(",False\n")
                  out.write(i+","+str(exchange.loc[i,ex +' All'])+"\n")
                  prev = curr

          if to_be_compared is None:
            to_be_compared = exchange.index
            for j in to_be_compared:
              out.write(j+","+str(exchange.loc[j,ex +" All"])+"\n")

              

          exchange.to_csv(os.path.join(sys.path[0],"..","..",str(exchange.columns[0].split("_")[0])+".csv"),na_rep = "NA")
          if all_exchange is None:
            all_exchange = pd.DataFrame(exchange[[ex+" Any",ex+" All"]])
          else:
            all_exchange = all_exchange.join(pd.DataFrame(exchange[[ex+" Any",ex+" All"]]),how = "outer")
        all_exchange.to_csv (os.path.join(sys.path[0],"..","..","All_Exchange.csv"), na_rep = "FALSE")
  if args.action == "consecutive":
    COMEX = os.path.join(sys.path[0],"data","Financial Data","COMEX","Generic","Lagged")
    DCE = os.path.join(sys.path[0],"data","Financial Data","DCE","Generic")
    LME = os.path.join(sys.path[0],"data","Financial Data","LME")
    SHFE = os.path.join(sys.path[0],"data","Financial Data","SHFE", "Generic")
    lag_indices = os.path.join(sys.path[0],"data","Financial Data","Indices","Lagged")
    indices = os.path.join(sys.path[0],"data","Financial Data","Indices")

    for exchange in [COMEX,DCE,LME,SHFE,lag_indices,indices]:
      leftout = []
      if exchange == SHFE:
        leftout = ["PBL.csv","ZNA.csv","XII.csv","XOO.csv"]
      
      if exchange == LME:
        leftout = ["LMAHDS.csv","LMCADS.csv","LMPBDS.csv","LMZSDS.csv","LMNIDS.csv","LMSNDS.csv"]

      if exchange == indices:
        leftout = ["DXY Curncy.csv","SPX Index.csv","SX5E Index.csv","UKX Index.csv","VIX Index.csv"]
      
      for fl in os.listdir(exchange):
        if ".csv" not in fl:
          continue
        col_leftout = []
        if fl in leftout:
          continue
        if fl == "PA_lag1.csv" or fl == "PL_lag1.csv":
          col_leftout =["Open","High","Low","Volume","Open.Interest"]
        
        df = pd.read_csv(os.path.join(exchange,fl),index_col = 0, header = 0)
        
        
        df = df.loc["2004-11-12":,:]
        df = count_consecutive(df,col_leftout)
        df = df.set_index('count')
        df.to_csv(os.path.join(sys.path[0],"..","..",fl))




        
          
          
            