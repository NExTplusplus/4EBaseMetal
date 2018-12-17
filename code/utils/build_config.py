import argparse
import json
import os
import sys
import itertools
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
    '-n','--var_count',type = int,default =1,
    help = 'the number of variables to include in each model'
  )
  parser.add_argument(
    '-out','--output',type = str, help='output file', default ="config.conf"
  )
  args = parser.parse_args()

  with open(args.data_configure_file) as fin:
    fname_columns = json.load(fin)
  
  with open("../../exp/ground_truth.conf") as gt:
    gt = json.load(gt)
  for g in gt:
    print(gt[g][0])
    with open("../../../"+gt[g][0]+"_"+args.output,"w") as out:
      out.write("[\n\t")
      for j in fname_columns:
        out.write('{\n\t\t\"'+g+'\":[\"'+gt[g][0]+'\"],')
        out.write('\n\t\t\"'+j+'\":[\"'+fname_columns[j][1]+'\"]\n\t},\n\t')
      out.write("\n]")

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

