# Code Organization

The code is organized as below:

## Folders

data: functions for run preprocessing of data.

model: class objects which details the computation within the model.

live: implementation of live deployment for each model.

feature_pytest: implementation of backward feature version compatibility.

train: scripts to expedite ALSTM tuning.

utils: utility functions for live implementation (general functions, data preprocess version control, data preprocess functions)
       and scripts that each model may need to generate commands.

new_Analyst_Report_Chinese: implementation of Analyst Report Indicator.

## Model-Specific

train_data_logistic.py : Pipeline to use live deployment of logistic regression.

train_data_xgboost.py : Pipeline to use live deployment of xgboost.

train_data_ALSTM.py : Pipeline to use live deployment of ALSTM classification.

train_data_ALSTMR.py : Pipeline to use live deployment of ALSTM regression.

train_data_ensemble.py : Pipeline to use live deployment of ensemble.

train_data_pp.py : Pipeline to use live deployment of post process (includes Substitution and Filter)

## Main Controller

controller.py : Allows access to each individual model for tuning, training and testing.

live_testing.py : Live implementation.

# Usage

Here we detail the methods to generate predictions for each model.

```
python code/controller.py -s *horizon -gt *metal -v *version -z *start_date::*end_date -sou NExT -m *model -o *action
```

* horizon: 1,3,5,10,20,60 
* metal: Al,Cu,Pb,Ni,Xi,Zn
* version: feature version (if it is ALSTM then remember to include method unless tuning)
* model: logistic/xgboost/alstm/alstmr/ensemble/pp_filter/pp_sub
* action: tune/train/test
