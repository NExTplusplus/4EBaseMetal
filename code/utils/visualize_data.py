import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],"..","..")))

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

with open(os.path.join(sys.path[0],"..","..","i6.csv")) as f:
  df = pd.read_csv(f,header = 0, index_col = 0)
  start_loc = df.index.get_loc("2011-06-01")
  end_loc = df.index.get_loc("2016-06-01")
  df = df.dropna(axis =1)
  df = df.iloc[start_loc:end_loc,:]
  for col in df.columns:
    plots = sns.distplot(df[col],kde = False)
    plt.savefig(os.path.join(sys.path[0],"..","..","mingwei","NExT","Plots","Normalized_"+col))
    plt.close()
    