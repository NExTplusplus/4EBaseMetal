import os
import sys
import itertools
import copy
import argparse
import json
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

if __name__ == "__main__":
  desc = "Build the config"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument(
    '--data_configure_file', '-c', type=str,
    help='configure file of the features to be read',
    default='../../exp/config.conf'
  )
  parser.add_argument(
    '-m', '--metal', type=str,
    help='metal to be chosen',
    default='../../exp/al_1.csv'
  )
  parser.add_argument(
    '-s','--steps',type = int, help='step ahead to predict', default = 1
  )
  parser.add_argument(
    '-out','--output',type = str, help='output file', default ="al"
  )
  args = parser.parse_args()

  with open(args.data_configure_file) as f:
    fname_column =json.load(f)

  with open(args.metal) as m:
    results = m.readlines()
    val_results = [result.split(",")[5] for result in results]
    # tes_results = [result.split(",")[7] for result in results]
    
  max_val = 1
  max_tes = 1

  with open("4EBaseMetal/exp/"+args.output+"_h"+str(args.steps)+".conf","w") as out:
    out.write("[\n\t")
    for ind in range(len(val_results)):
      if ind == 0 or ind == 1:
        continue
      else:
        if val_results[ind] > val_results[max_val]:
          max_val = ind
        # if tes_results[ind] > tes_results[max_tes]:
          # max_tes = ind
    print(max_val)
    out.write(str(fname_column[max_val-1]).replace("\'","\""))
    # out.write(",\n\t")
    # out.write(str(fname_column[max_tes-1]).replace("\'","\""))


    out.write("\n]")

