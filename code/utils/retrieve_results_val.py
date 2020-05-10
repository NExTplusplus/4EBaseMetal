import pandas as pd
import os

for version in ["v5","v7","v9","v10","v12"]:
# ground_truth = ("all","all")
  for h in [1,3,5]:
    total = pd.DataFrame(index = [1,5,10,20,30],columns = ["all","near","far","same","reverse"])
    for lag in [1,5,10,20,30]:
      all_mean_results = pd.DataFrame()
      near_mean_results = pd.DataFrame()
      far_mean_results = pd.DataFrame()
      same_mean_results = pd.DataFrame()
      reverse_mean_results = pd.DataFrame()
      for ground_truth in [("Co","LME_Co_Spot"),("Al","LME_Al_Spot"),("Ni","LME_Ni_Spot"),("Ti","LME_Ti_Spot"),("Le","LME_Le_Spot"),("Zi","LME_Zi_Spot")]:
        if version == "v10" or version == "v12":
          ground_truth=("all","all")
        f = pd.read_csv('_'.join([ground_truth[0],version,str(lag),"h"+str(h)])+".csv")
        all_mean_results = pd.concat([all_mean_results,f.iloc[0:1,:]],axis = 0)
        near_mean_results = pd.concat([near_mean_results,f.iloc[5:6,:]],axis = 0)
        far_mean_results = pd.concat([far_mean_results,f.iloc[10:11,:]],axis = 0)
        same_mean_results = pd.concat([same_mean_results,f.iloc[15:16,:]],axis = 0)
        reverse_mean_results = pd.concat([reverse_mean_results,f.iloc[20:21,:]],axis = 0)
        if version == "v10" or version == 'v12':
          break
      all_mean_results = all_mean_results.mean(axis = 0)['result']
      near_mean_results = near_mean_results.mean(axis = 0)['result']
      far_mean_results = far_mean_results.mean(axis = 0)['result']
      same_mean_results = same_mean_results.mean(axis = 0)['result']
      reverse_mean_results = reverse_mean_results.mean(axis = 0)['result']

      total.loc[lag,"all"] = all_mean_results
      total.loc[lag,"near"] = near_mean_results
      total.loc[lag,"far"] = far_mean_results
      total.loc[lag,"same"] = same_mean_results
      total.loc[lag,"reverse"] = reverse_mean_results
    total.to_csv(version+"_"+str(h)+"_1.csv")




