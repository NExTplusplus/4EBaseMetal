import pandas as pd

ground_truth = ("all","all")
version = "v10"
with open(ground_truth[0]+" "+version+".txt","w") as out:
  for lag in [5,10,20,30]:
    for h in [1,3,5]:
      f = pd.read_csv('_'.join([ground_truth[0],version,str(lag),"h"+str(h)])+".csv")
      out.write(" ".join(["python code/train/train_xgboost_online_v10.py",
                          # "-gt",ground_truth[1],
                          "-l",str(lag),"-s",str(h),"-xgb 0","-v",version,"-c exp/LMCADY_v5.conf",
                          "-sou NExT","-max_depth",str(int(f.iloc[0,1])),"-learning_rate",str(f.iloc[0,2]),
                          "-gamma",str(f.iloc[0,3]),"-min_child",str(int(f.iloc[0,4])),"-subsample",str(f.iloc[0,5]),"-voting all",
                          ">","_".join([ground_truth[0],"l"+str(lag),"h"+str(h),version,"1718.txt"]),"2>&1 &"])+"\n")
