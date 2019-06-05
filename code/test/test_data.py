import json
import os

def check_init_process_missing_val(f1,f2):
  with open(f1) as f:
    lines_1 = f.readlines()
  with open(f2) as f:
    lines_2 = f.readlines()
  lines_1 = [line.strip("\n").split(",") for line in lines_1]
  lines_2 = [line.strip("\n").split(",") for line in lines_2]
  sta_ind = 0
  count = 0
  for ind in range(len(lines_1)):
    if "nan" in lines_1[ind]:
      count = 0
      sta_ind = 0
    else:
      if count == (10 - 1):
        count = -1
        break
      elif count != -1:
        count += 1
        if sta_ind == 0:
          sta_ind = ind
  tweak = 0
  for ind in range(len(lines_2)):
    if "nan" in lines_1[sta_ind + ind]:
      tweak += 1
    else:
      print(lines_1[sta_ind + ind])
      print(lines_2[ind-tweak])
      assert lines_1[sta_ind + ind] == lines_2[ind - tweak]

for f in os.listdir(os.path.join(os.curdir,"..","i1")):
  check_init_process_missing_val(os.path.join(os.curdir,"..","i1",f),os.path.join(os.curdir,"..","i2","i2"+f.lstrip("i1")))





  



