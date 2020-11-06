# Operation

This readme explains the live deployment of the Base Metal Price Prediction. It consists of two primary components, the Machine Learning Framework (MLF) and the Analyst Report Indicator. It also contains scripts to analyze the performance of the predictions.

# Usage

The above two components are operated separately but the Analyst Report Indicator should be completed before the MLF for a day.

## Analyst Report Indicator

Please refer to code/new_Analyst_Report_Chinese/readme.txt


## MLF

* Prior to operation please ensure that data is provided in a similar format to **data** folder. For further details please check the readme.md in **data** folder.
* Following that, a feature version must be developed. Please refer to fv_development.md.
* To operate the live deployment directly, please run the following command with the working directory as the current directory (the directory this readme is in)

```
    python code/live_testing.py -o live -m train -gt *metal -s *horizon -z *start_date::*end_date -r *recent
```

* The above line of code will generate the model instances in *result/model* folder

```
    python code/live_testing.py -o live -m test -gt *metal -s *horizon -z *start_date::*end_date -r *recent
```

* The above line of code will generate the predictions (in csv format) in *result/prediction* folder

```
    python code/live_testing.py -o return -m test -gt *metal -s *horizon -z *start_date::*end_date -r *recent

```

* The above line of code extracts the predictions from the csv and returns them as a dataframe (prints as standard output)

```
    python code/controller.py -s *step -gt *metal -z *start_date::*end_date -m pp_filter
```

* To analyze results for performance please run the above command. The results will be in post_process_Filter.csv

* The parameters denoted with \* should be replaced with values such as:

  * metal : Al,Cu,Pb,Ni,Xi,Zn (metals are comma-separated, you can choose a subset of metals)
  * horizon : 1,3,5,10,20,60 (horizon are comma-separated, you can choose a subset of horizons)
  * start_date : Ex. 2014-07-01 （format is as YYYY-mm-dd)
  * end_date : Ex. 2020-06-30 （format is as YYYY-mm-dd)
  * recent : Ex. 1 (integer to denote how many recent days do you want to predict)
 
