python code/train_data_xgboost.py -gt LME_Al_Spot -d 2017-06-30 -s 1 -l 1 -c exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf -v v7 -o tune > result/validation/xgboost/Al_xgboost_l1_h1_v7.txt 2>&1 &
python code/train_data_xgboost.py -gt LME_Al_Spot -d 2017-06-30 -s 1 -l 5 -c exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf -v v7 -o tune > result/validation/xgboost/Al_xgboost_l5_h1_v7.txt 2>&1 &
python code/train_data_xgboost.py -gt LME_Al_Spot -d 2017-06-30 -s 1 -l 10 -c exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf -v v7 -o tune > result/validation/xgboost/Al_xgboost_l10_h1_v7.txt 2>&1 &
