import argparse
import json
import os
import sys
import itertools
import copy
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
    '-gt','--ground_truth', type = str, default = "LMAHDY",help ="ground truth to consider"
  )
  parser.add_argument(
    '-n','--var_count',type = int,default =1,
    help = 'the number of variables to include in each model'
  )
  parser.add_argument(
    '-out','--output',type = str, help='output file', default =".conf"
  )
  args = parser.parse_args()

  with open(args.data_configure_file) as fin:
    lines = fin.readlines()
    lines = lines[1:len(lines)-1]
    lines = lines[1:len(lines)-1:3]
    print(lines)
    lines = [line.rstrip(',\n') for line in lines]
    lines = [line.strip() for line in lines]
    combinations = itertools.combinations(lines,args.var_count)
    
  with open("../../exp/ground_truth.conf") as gt:
    ground_truth = json.load(gt)

  for g in ground_truth:
    if ground_truth[g][0] != args.ground_truth:
      continue
    with open("../../exp/"+args.ground_truth+"_n"+str(args.var_count)+args.output,"w") as out:
      temp = copy.copy(combinations)
      out.write("[\n\t{\n\t\t\""+g+"\":[\""+args.ground_truth+"\"]\n\t},\n\t")
      for j in temp:
        num = 0
        out.write('{\n\t\t\"'+g+"\":[\""+args.ground_truth+'\"],\n\t\t') 
        for k in j:
          num += 1
          if k[-8:-2] == args.ground_truth:
            continue       
          
          if num <args.var_count:
            out.write(k+',\n\t\t')
          else:
            out.write(k)
        out.write('\n\t},\n\t')
      out.write("\n]")
      out.close()

#remember to remove the last comma manually and the csv1,2,3,4,5,6
    






# alum = ["LME/LMAHDY.csv","LME/LMAHDS03_OI.csv","LME/LMEAluminium3M.csv","SHFE/Generic/AA.csv"]
# copp = ["LME/LMCADY.csv","LME/LMCADS03_OI.csv","LME/LMECopper3M_longer.csv","SHFE/Generic/CU.csv","COMEX/Generic/HG.csv"]
# tin = ["LME/LMSNDY.csv","LME/LMSNDS03_OI.csv","LME/LMETin3M.csv","SHFE/Generic/XOO.csv"]
# zinc = ["LME/LMZSDY.csv","LME/LMZSDS03_OI.csv","LME/LMEZinc3M.csv","SHFE/Generic/ZNA.csv"]
# lead = ["LME/LMPBDY.csv","LME/LMPBDS03_OI.csv","LME/LMELead3M.csv","SHFE/Generic/PBL.csv"]
# nick = ["LME/LMNIDY.csv","LME/LMNIDS03_OI.csv","LME/LMENickel3M.csv","SHFE/Generic/XII.csv"]
# univ = ["Indices/SHSZ300 Index.csv","Indices/HSI Index.csv","Indices/SHCOMP Index.csv",
#         "Indices/Lagged/DXY Curncy_lag1.csv","Indices/Lagged/SPX Index_lag1.csv","Indices/Lagged/VIX Index_lag1.csv",
#         "COMEX/Generic/GC.csv","COMEX/Generic/PL.csv","COMEX/Generic/PA.csv","SHFE/Generic/RT.csv",
#         "DCE/Generic/AE.csv","DCE/Generic/AK.csv","DCE/Generic/AC.csv"]
# associated = [alum,copp,tin,zinc,lead,nick]
# metals = ["al_","cu_","tn_","zn_","pb_","nc_"]

# for m in range(len(metals)):
#     with open(os.path.join(os.curdir,"..","..","exp",metals[m] + "config.conf"),mode = "w") as out:
#         out.write("[\n")
#         associated[m].extend(univ)
#         curr_assoc = associated[m]
#         gt = os.path.join(data_folder_path,curr_assoc[0])
#         with open(gt, mode = "r") as fl:
#             header = fl.readline()
#             header = header.replace("\"Index\",","")
#             gt = "\t{\n\t\""+gt+"\": ["+str.strip(header)+"]"
#         for flp in curr_assoc:
            
#             if flp == curr_assoc[0]:
#                 out.write(gt+"\n\t},\n")
#                 continue
#             curr_path = os.path.join(data_folder_path,flp)
#             with open(curr_path, mode = "r") as fl:
#                 header = fl.readline()
#                 header = header.replace("\"Index\",","")
#                 header = header.split(",")
#                 for h in header:
#                     out.write(gt+",")
#                     out.write("\n\t\""+curr_path+"\": ["+str.strip(h)+"]\n\t},\n")
#         out.write("]")
#         out.close()

