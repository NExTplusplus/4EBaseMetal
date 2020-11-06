This folder contains the live code for tuning each model best parameter, training each machine learning model, and test each model in a live environment.

Index:
ALSTM_live.py               : ALSTM for classification
ALSTMR_live.py              : ALSTM for regression
ensemble_live.py            : Ensemble for all classification models
Logistic_live.py            : Logistic regression
linear_live.py              : Linear regression
post_process_live.py        : Post Process for Substitution and Filter
xgboost_live.py             : xgboost_live

Each file implements the methods (tune,train,test) that are required by the models.
For instance, post process does not have a train method because it does not require model training.

These files are used by train_data_(model).py under the code folder.