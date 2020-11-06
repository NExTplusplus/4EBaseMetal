This folder contains the code for constructing the data and control the version of the different versions feature. Note that the different version of different feature version for different models (e.g., ALSTM, xgboost) could be in different files.

Used Files
analyze_alstm_tune.py                   analyze alstm tuning logs to identify hyperparameter combinations for training and testing operations
analyze_predictions.py                  Script to analyze the prediction results for all models.
data_preprocess_functions.py            List of functions that apply the preprocess calculations to given data.
data_preprocess_version_control.py      A version controller for the data preprocess functions for different feature versions.
fracdiff.py                             List of functions that are related to fractional differencing
general_functions.py                    List of functions that are used across all models, mainly for data
generate_labels.py                      Script used to generate labels.
log_reg_script.py                       Script related to logistic regression operations.
normalize_feature.py                    List of functions for normalization
post_process.py                         List of functions for post processing.
post_process_script.py                  Script related to post process operations.
process_strategy.py                     List of functions for strategy.
read_data.py                            list of functions for reading data from csv or from 4E
seasonal_decomp.py                      list of functions for seasonal trend decomposition calculations.
supply_and_demand.py                    list of functions for supply and demand indicator calculations.
Technical_indicator.py                  List of functions that calculate the values of the technical indicators.
xgboost_script.py                       Script related to xgboost operations


