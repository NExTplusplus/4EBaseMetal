import os 
import sys
import argparse
from controller import run,analyze_zoom
import time
import datetime

def live(date,method,return_df = False,recent = 1):
    command = ""
    if method == "train":
        command += "python code/controller.py -s 1,3,5,10,20,60 -v v3,v5,v7,v9,v23 -o train -m logistic -gt Al,Co,Le,Ni,Ti,Zi -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 4h; sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -v v10,v12,v24 -o train -m logistic -gt All -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -v v3,v5,v7,v9,v23 -o train -m xgboost -gt Al,Co,Le,Ni,Ti,Zi -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 4h; sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -v v10,v12,v24,v28,v30 -o train -m xgboost -gt All -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 1h\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v16_best_loss -o train -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v16_best_acc -o train -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v16_ave_loss -o train -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v16_ave_acc -o train -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v26_best_loss -o train -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v26_best_acc -o train -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v26_ave_loss -o train -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v26_ave_acc -o train -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v r2_best_loss -o train -m alstmr -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 1h\n"

    elif method == "test":
        command += "python code/controller.py -s 1,3,5,10,20,60 -v v3,v5,v7,v9,v23,v10,v12,v24 -o test -m logistic -gt Al,Co,Le,Ni,Ti,Zi -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 5h;\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -v v3,v5,v7,v9,v23,v10,v12,v24,v28,v30 -o test -m xgboost -gt Al,Co,Le,Ni,Ti,Zi -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 6h;\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v16_best_loss -o test -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v16_best_acc -o test -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v16_ave_loss -o test -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v16_ave_acc -o test -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v26_best_loss -o test -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v26_best_acc -o test -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v26_ave_loss -o test -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v v26_ave_acc -o test -m alstm -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 30m\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt All -v r2_best_loss -o test -m alstmr -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 1h\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt Al,Co,Le,Ni,Ti,Zi -o test -m ensemble -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 1h\n"
        command += "python code/controller.py -s 1,3,5,10,20,60 -gt Al,Co,Le,Ni,Ti,Zi -v r2_best_loss,ensemble -o test -m Filter -sou 4E -z "+date+" > /dev/null 2>&1 &\n"
        command += "sleep 1h\n"
    os.system(command)
        
def return_dates(zoom):

    res = ""
    start = zoom.split("::")[0]
    end = zoom.split("::")[1]
    curr = start
    while curr <= end:
        curr_year = curr.split('-')[0]
        curr_month = curr.split('-')[1]
        curr_day = curr.split('-')[2]
        res += curr+","
        if curr_month == "12" and curr_day == "31":
            curr = str(int(curr_year)+1)+"-01-01"
        elif int(curr_year)%4 == 0 and curr_month == "02" and curr_day == "28":
            curr = curr_year+"02-29"
        elif (curr_day == "30" and curr_month in ["02","04","06","09","11"]) or (curr_day == "31" and curr_month in ["01","03","05","07","08","10"]):
            curr_month = int(curr_month) + 1
            curr_month = "0"+str(curr_month) if len(str(curr_month)) == 1 else str(curr_month)
            curr = curr_year+"-"+curr_month+"-01"
    return res





if __name__ ==  '__main__':        
    desc = 'controller for financial models'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-m','--method', help='action', type = str, default = "test"
    )
    parser.add_argument('-z','--zoom',type = str, help = "period of dates", default = "2014-07-01::2018-12-31")

    args = parser.parse_args()
    dates = return_dates(args.zoom)
    live(args.dates,args.method)
    
