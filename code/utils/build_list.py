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
    '-m', '-metal', type=str,
    help='metal to be chosen',
    default='../../exp/al_1.csv'
  )
  # parser.add_argument(
  #   '-h','--horizon',type = int, help='step ahead to prenum_appt', default = 1
  # )
  parser.add_argument(
    '-n','--varcount',type = int, help='number of variables', default =1 
  )
  parser.add_argument(
    '-out','--output',type = str, help='output file', default ="al"
  )
  args = parser.parse_args()

  with open(args.data_configure_file) as f:
    fname_column =json.load(f)

  with open(args.metal) as m:
    results = m.readlines()
    results = [result.split(",")[7] for result in results]

  with open("../../exp/"+args.output+"_h1"+"_n"+str(args.varcount)+".conf","w") as out:
    out.write("[\n\t")
    col_app = dict()
    num = -1
    for f in fname_column:
      num += 1
      if results[num].strip() != "TRUE":
        continue
      for fname in f:
        if f[fname][0] in col_app.get(fname,[]):
          continue
        if fname not in col_app.keys():
          col_app[fname] = [f[fname][0]]
        else:
          col_app[fname].append(f[fname][0])
        print(fname)
        print(f[fname][0])
        out.write("{\n\t\t\""+fname+"\":[\""+f[fname][0]+"\"]\n"+"\t},\n\t")

    out.write("\n]")

