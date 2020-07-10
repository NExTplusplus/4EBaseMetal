This folder contains the code for constructing the data and control the version of the different versions feature. Note that the different version of different feature version for different models (e.g., ALSTM, xgboost) could be in different files.

Used Files
analyze_alstm_tune.py                   analyze alstm tuning logs to identify results for hyperparameter combinations
analyze_predictions.py                  Script to analyze the prediction results for all models.
data_preprocess_functions.py            List of functions that apply the preprocess calculations to given data.
data_preprocess_version_control.py      A version controller for the data preprocess functions for different feature versions.
ensemble_script.py                      script realted to ensemble operations
fracdiff.py                             List of functions that are related to fractional differencing
general_functions.py                    List of functions that are used across all models, mainly for data
generate_labels.py                      Script used to generate labels.
log_reg_script.py                       Script related to logistic regression results.
normalize_feature.py                    List of functions for normalization
post_process.py                         List of functions for post processing.
process_strategy.py                     List of functions for strategy.
read_data.py                            list of functions for reading data from csv or from 4E
seasonal_decomp.py                      list of functions for seasonal trend decomposition calculations.
supply_and_demand.py                    list of functions for supply and demand indicator calculations.
Technical_indicator.py                  List of functions that calculate the values of the technical indicators.
xgboost_script.py                       Script related to xgboost operations


Legacy Documents
analyze_alstm_results.py
analyze_similarity.py
analyze.py
build_config.py
build_list.py
data_loss.py
dataset_id_embedding.py
deal_with_minor_value.py
dealt_with_OI_max_six.py
deal_wuth_OI_max.py
evaluator.py
find_max.py
peasonr.py
probability.py
regression_visuals.py
retrieve_commands.py
retrieve_data_ensemble_reversal.py
retrieve_data_top_5.py
retrieve_data_top.py
retrieve_data_V10.py
retrieve_data.py
retrieve_ensemble_new_data.py
retrieve_results_val.py
retrieve_results.py
retrive_the_data.py
set_validation.py
strategy_test.py
transform_data.py
version_control_functions_torch.py
version_control_three_classifier_fucntions.py
visualize_data.py
voting.py
get_coef.py                                     Document for logistic regression coefficients.
log_reg_functions.py                            List of functions to calculate error values for logistic regression.
strategy_test.py                                (to be filled by wanying)  
evaluator.py                                    Document.
find_Max.py                                     Document for logistic regression.
peasonr.py                                      Document.
set_validation.py                               Documents.                       
probability.py                                  Document of logistic regression probability.
transform_data.py                               list of general functions.
version_control_functions_torch.py              list of functions for version control for alstm.     
data_loss.py                                    Document for verification of data loss.       
version_control_three_classifier_functions.py   list of functions for version control for 3 classifier.
visualize_data.py                               Script for plotting data.                        
voting.py                                       
build_config.py                                 Document for building config.
build_list.py                                   Document for building list.
dataset_id_embedding.py                         Document from weiyi.           
deal_with_OI_max_six.py                         Document for dealing with large values.
deal_with_minor_value.py                        Document for dealing with minor values.
deal_wuth_OI_max.py                             Document for dealing with large values.
retrieve_commands.py                            Script for retrieving
retrieve_data_V10.py                            
retrieve_data_top.py                            
retrieve_data_top_5.py                          
retrieve_ensemble_new_data.py                   
retrieve_results.py                             
retrive_the_data.py                             