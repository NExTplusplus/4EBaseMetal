This folder contains the code for each machine learning model, each model per file. Note that the different version of one model (e.g., LSTM and AttLSTM) could be in different files.

Files:
base_predictor.py       This is the base class for a prediction model that implement the basic inferface of a predictor. 

logistic_regression.py  The implementation of LR, documentation: https://wiki.alphien.com/ALwiki/Logistic_Regression_Documentation

model_embedding.py      The implementation of the ALSTM model with an extension of metal id.

model_embedding_mc.py   The implementation of the ALSTM model with an extension of metal id and monte carlo.

ensemble.py             The implementation of ensemble methods

post_process.py         The implementation of post process methods

model_id_embedding.py   An exact copy of model_embedding.py
