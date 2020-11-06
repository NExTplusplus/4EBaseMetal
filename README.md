## Objective

This is the repository for the project of Base Metal Price Prediction, a collaboration between 4 Elements Capital (4E) and NExT++. Here we aim to design a usable framework for 4E to deploy on their servers to predict the prices of base metals.


## Requirements

Python packages
```
statsmodels
pandas
numpy
scipy
...to be completed...
```

## File Organization

### 1st level

We mainly have four folders: **code, data, exp, and result**. New 1st level folders could be created after a discussion. Please follow the following description to put your code:

| Folder | Description |
| :-----: | :-----: |
| code | all python/R codes in script file, e.g., the code to load financial data. |
| data | data files used to train and test model so that 4E can re-produce the model training on their server. |
| exp | auxillary files (e.g., configure file) that hold data configuration, selected hyperparameters and scope of application. |
| result | folder that stores our outputs from tuning (outputs tuning logs), training (outputs model instances) and testing (outputs predictions) |


## Deployment

We have designed a live deployment framework by connecting to the 4E database. The live deployment is strictly defined on selection of models and feature versions. To operate the live deployment, please refer to Operation.md.
