## Synchronization

* For everything under this repository, we need to keep them synchronized across our personal computer, SoC cluster, and 4E jupyter notebook server. We use git pull and push to do the synchronize. **NO** scp and upload!
* Send a message to our Wechat group when **push**.

## Naming and Placing Rules

* Keep this repository with the **newest** code.
* **Pull** before editing code on personal computer. **Push** within one day after editing on personal computer.
* Try our best to organize the files according to the following description of the File Organization.
* For files larger than **100MB** (constraint of GitHub), we only synchronize a soft link in this repository. For example, on the 4E server, we manually copy the file (e.g., recommend.csv) to the public folder "NEXT/Large_Files" and create a soft link with the linux command "ln -s".
* Create a **readme** file in each folder. In the readme file, we give an overall description of the contents in the corresponding folder. Furthermore, we maintain a list to introduce the files under the folder. The format should be one line per file, each line is "filename\tdescription". Here is an example.

## File Organization

### 1st level

We mainly have four folders: **code, data, exp, and notebooks**. New 1st level folders could be created after a discussion. Please follow the following description to put your code:

| Folder | Description |
| :-----: | :-----: |
| code | all python/R codes in script file, e.g., the code to load financial data. |
| data | data files used to train and test model. |
| exp | experiment logs, e.g., the log (loss, accuracy, etc.) to tune a LR model |
| notebooks | we prepare notebooks for two reasons: efficient development or deployment (to 4E) |

### 2nd level

Under each first level folder, we roughly **organize code by model**, e.g., LR_Single_Model. For each model, we can create subfolders for different versions. Note that a subfolder named Online_Version is compulsary if we would do online testing for the model. Here is an example: 4EBaseMetal/notebooks/Sentiment_Indicator_Chinese_Analyst_Report/

## Deployment

In addition to offline development, we also need to deploy the online testing version of each model to the 4E notebook server. From now on, we keep all notebooks for online testing under the notebooks/MODEL_NAME/Online_Version, and **push back** any updates we would make on the 4E notebook server.
