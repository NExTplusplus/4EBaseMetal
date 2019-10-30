############Tree for this folder############
├── call_aip.py
├── how_to_create_baidu_application.txt
├── readme.txt
├── step3_data
│   ├── config.ini
│   └── error_recommend.json
└── update_history.txt

############descriptions for this code############
1) this code is used to get the sentiment result from the aip, then it will insert into the table for certain metal, like Cu_sentiment, etc.

2) For your convenience , I add I add some exception control in the code, after you run the code, it will tell you whether it has some problem, it will store in the same error_json file, place in './step3_data/error_recommend.json'.

3) For each update, I will write down the update in the update_history.txt(placed in './step3_extract_recommendation/'), you can check it if you need.

############code instructions############
#Before you run the code, hope you can change some config for your own in the file config.ini(placed in './step3_extract_recommendation/step3_data'). Because the account and the password need to be changed by yourself. Port and the database name, if it is the same with your own database, then you don't need to change it. Specially, this step, you also need to revise the baidu config, actually, the default config is also ok.

#run the code
1)open the terminal, and change the directory to step3_extract_recommendation
2)python call_aip.py run 

#check the error
1)open the terminal, and change the directory to step3_extract_recommendation
2)python call_aip.py check

############problems############


