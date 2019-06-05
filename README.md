## Naming and Placing Rules



## File Organization

### 1st level

We mainly have three folders: code, exp, and notebooks. New 1st level folders could be created after a discussion. Please follow the following description to put your code:

| Folder | Description |
| :-----: | :-----: |
| code | all python/R codes in script file, e.g., the code to load financial data. |
| exp | experiment logs, e.g., the log (loss, accuracy, etc.) to tune a LR model |
| notebooks | we prepare notebooks for two reasons: efficient development or deployment (to 4E) |

### 2nd level

Under each first level folder, we roughly organize code by model, e.g., LR_Single_Model. For each model, we can create folders for different versions. Note that a folder named Online_Version is compulsary if we would do online testing for the model.
