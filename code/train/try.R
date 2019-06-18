library(reticulate)
use_python("C:/Users/mingx/AppData/Local/Programs/Python/Python36/python.exe",required= TRUE)
py_run_string("from joblib import load 
load('model.joblib')")