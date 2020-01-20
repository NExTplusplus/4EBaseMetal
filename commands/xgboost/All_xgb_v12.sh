#!/bin/bash
cd ../../
python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 1 -l 1 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l1_h1_v12.txt 2>&1 &
python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 1 -l 5 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l5_h1_v12.txt 2>&1 &
python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 1 -l 10 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l10_h1_v12.txt 2>&1 &

python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 3 -l 1 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l1_h3_v12.txt 2>&1 &
python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 3 -l 5 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l5_h3_v12.txt 2>&1 &
python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 3 -l 10 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l10_h3_v12.txt 2>&1 &

python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 5 -l 1 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l1_h5_v12.txt 2>&1 &
python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 5 -l 5 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l5_h5_v12.txt 2>&1 &
python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 5 -l 10 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l10_h5_v12.txt 2>&1 &

python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 10 -l 1 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l1_h10_v12.txt 2>&1 &
python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 10 -l 5 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l5_h10_v12.txt 2>&1 &
python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 10 -l 10 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l10_h10_v12.txt 2>&1 &

python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 20 -l 1 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l1_h20_v12.txt 2>&1 &
python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 20 -l 5 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l5_h20_v12.txt 2>&1 &
python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 20 -l 10 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l10_h20_v12.txt 2>&1 &

python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 60 -l 1 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l1_h60_v12.txt 2>&1 &
python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 60 -l 5 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l5_h60_v12.txt 2>&1 &
python code/train_data_xgboost.py -gt All -d 2017-06-30 -s 60 -l 10 -c exp/online_v10.conf -v v12 -o tune > result/validation/xgboost/All_xgboost_l10_h60_v12.txt 2>&1 &
cd commands/xgboost