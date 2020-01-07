############Tree for this folder############

############descriptions for this code############
1) this code is used to predict the price movement based on some date, for example, when we input '2019-01-01'
   then it will predict the price movement of the next 1, 3, 5, 10, 20, 60 transaction day compared with '2019-01-1'

2) For your convenience, I add some comment in the code and some intermediate file is also stroed('./accur_score_intermediate/', 'adjustment_intermediate/', 'discrete_param/')

3) For every update, I will write down the update in the update_history.txt(placed in './step4_sentiment_analysis/update_history.txt'), you can check it if you need.

############code instructions############
#Before you run the code, hope you can change some config for your own in the file config.ini(placed in './step4_sentiment_analysis/step4_data'). Because the account and the password need to be changed by yourself. Port and the database name, if it is the same with your own database, then you don't need to change it. Specially, this step, you also need to revise the baidu config, actually, the default config is also ok.

Here the date follows the rules: For one day: you need to input 2019-01-01 2019-01-01, For a period:you need to input 2019-01-01 2019-01-02. 
P.S. The two date you input will be include in the predict period.
#run the code
1)open the terminal, and change the directory to step4_sentiment_analysis
2)python main_function.py 2019-01-01 2019-01-02 run

#reproduce the result of 2017 and 2018
P.S. If you want to reproduce the result in your server, you need to do the following operations for 4 half, that is to say, you need to run 4 times.
1)open the terminal, and change the directory to step4_sentiment_analysis
2)python main_function.py 2017-01-01 2017-06-30 reproduction
############problems############
As the Zn is generated from the 4E server, so it may have some difference compared with the local one.