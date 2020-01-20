#!/bin/bash
cd ../../
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 1 -l 1 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 1 -l 5 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 1 -l 10 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 1 -l 20 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &

python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 3 -l 1 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 3 -l 5 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 3 -l 10 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 3 -l 20 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &

python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 5 -l 1 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 5 -l 5 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 5 -l 10 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 5 -l 20 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &

python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 10 -l 1 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 10 -l 5 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 10 -l 10 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 10 -l 20 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &

python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 20 -l 1 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 20 -l 5 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 20 -l 10 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 20 -l 20 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &

python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 60 -l 1 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 60 -l 5 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 60 -l 10 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
python code/train_data_lr.py -gt LME_Ni_Spot -d 2017-06-30 -s 60 -l 20 -c exp/3d/Co/logistic_regression/v3/LMCADY_v3.conf -v v3 -o tune > /dev/null 2>&1 &
cd commands/lr