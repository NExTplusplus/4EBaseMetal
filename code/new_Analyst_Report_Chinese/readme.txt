############descriptions for this code############
1) This code is mainly for integrating the function of the step 1 to step 4.

############code instructions############
There are totally 6 parameters:
1. bool, whether to do the step 1 to step 3, the values should be "True" or "False", please pay attention to the capital letter
2. bool, whether to do the step 4, the values should be "True" or "False", please pay attention to the capital letter
3. time zoom, the date we need to predict, the values should be "2018:", ":2018", "2018:2019", "::", "None". The permitted time period is 2008 until now. For example "2018:"->"2018:2020", ":2018"->"2008:2018", "::"->"2008:2020". And if the value is "None", then we will ignore this parameter and consider the parameter 4.
4. int, how many recent days we need to predict
5. str, the certain metal we need to predict, the values should be "Copper", "Nickel", "Zinc", "Lead", "Tin", "Aluminum", "all", when the value is "all", then will run for all the metal.
6. str, the mode for prediction, the possible values are "run" and "reproduction"
7. bool, whether to get the output dataframe, the values should be "True" or "False", please pay attention to the capital letter
(P.S. if we get the time zoom, then we will ignore the )

#run the code
1)open the terminal, and change the directory to step4_sentiment_analysis
2)python main_controller.py True True 2018: 2 Copper run True