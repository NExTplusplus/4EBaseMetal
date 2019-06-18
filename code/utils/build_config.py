import argparse
import json
import os
import sys
import itertools
import copy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

def read_combination(comb):
  ans = dict()
  for j in comb:
    temp = j.split(":")
    key = temp[0].strip("\'")
    val = temp[1].strip("\'")
    if key in ans.keys():
      ans[key] = ans[key][0:-1]+","+val[1:len(val)]
    else:
      ans[key] = val
  res = list()
  for key in ans.keys():
    res.append(key+":"+ans[key])
  return res

if __name__ == "__main__":
  desc = "Build the config"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument(
    '--data_configure_file', '-c', type=str,
    help='configure file of the features to be read',
    default='../../exp/config.conf'
  )
  parser.add_argument(
    '-gt','--ground_truth', type = str, default = "LMAHDY",help ="ground truth to consider"
  )
  parser.add_argument(
    '-s','--steps',type = int, help='step ahead to predict', default = 1
  )  
  parser.add_argument(
    '-n','--var_count',type = int,default =1,
    help = 'the number of variables to include in each model'
  )
  args = parser.parse_args()

  with open(args.data_configure_file) as fin:
    lines = fin.readlines()
    lines = lines[1:len(lines)-1]
    lines = lines[1:len(lines)-1:3]
    lines = [line.rstrip(',\n') for line in lines]
    lines = [line.strip() for line in lines]
    combinations = itertools.combinations(lines,args.var_count)
    
  with open("4EBaseMetal/exp/ground_truth.conf") as gt:
    ground_truth = json.load(gt)

  for g in ground_truth:
    if ground_truth[g][0] != args.ground_truth:
      continue
    with open("4EBaseMetal/exp/"+args.ground_truth+"_h"+str(args.steps)+"_n"+str(args.var_count)+"_config.conf","w") as out:
      temp = copy.copy(combinations)
      out.write("[\n\t{\n\t\t\""+g+"\":[\""+args.ground_truth+"\"]\n\t},\n\t")
      count = len(list(combinations))
      for j in temp:
        count -= 1
        done = False
        for i in j:
          if i[-8:-2] == args.ground_truth:
            done = True
            break
        if done:
          continue
        out.write('{\n\t\t\"'+g+"\":[\""+args.ground_truth+'\"],\n\t\t') 
        for k in read_combination(j):
          if k[-8:-2] == args.ground_truth:
            continue       
          if k != read_combination(j)[-1]:
            out.write(k+',\n\t\t')
          else:
            out.write(k)
        out.write('\n\t}')
        if count == 0:
          out.write('\n')
        else:
          out.write(',\n\t')
        
      out.write("]")
      out.close()

