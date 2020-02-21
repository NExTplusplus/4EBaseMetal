Al = ""
Le = ""
Ni = ""
Zi = ""
Ti = ""

with open("commands/xgboost/Co_xgb_v23.sh") as f:
  lines = f.read()
  Al = lines
  Al = Al.replace("Co_","Al_")
  Le = lines
  Le = Le.replace("Co_","Le_")
  Ni = lines
  Ni = Ni.replace("Co_","Ni_")
  Zi = lines
  Zi = Zi.replace("Co_","Zi_")
  Ti = lines
  Ti = Ti.replace("Co_","Ti_")

with open("commands/xgboost/Al_xgb_v23.sh","w") as f:
  f.write(Al)
  f.close()
    
with open("commands/xgboost/Le_xgb_v23.sh","w") as f:
  f.write(Le)
  f.close()
    
with open("commands/xgboost/Ni_xgb_v23.sh","w") as f:
  f.write(Ni)
  f.close()
    
with open("commands/xgboost/Zi_xgb_v23.sh","w") as f:
  f.write(Zi)
  f.close()
    
with open("commands/xgboost/Ti_xgb_v23.sh","w") as f:
  f.write(Ti)
  f.close()
    