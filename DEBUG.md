# DEBUG

## MLF Live Deployment

The MLF Live Deployment has three steps, train, test and return. Train and test both have a separate folder for logging you can find under the **log** folder.

Under their own folders, we have a folder for standard output **out** and for standard error **err**.

* Should there be an error, there will be an error log for each model (logistic,alstm,xgboost,etc) in the main folder **train** or **test** which tells you which commands did it fail on. The takeaway from these commands are generally the metal that has failed, the step that has failed, and the version that has failed. Then you can search for the error log under **err** of the precise model, metal, and version to find what error was reported.


