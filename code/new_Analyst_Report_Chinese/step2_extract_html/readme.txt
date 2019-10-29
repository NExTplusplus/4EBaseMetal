############Tree for this folder############
├── function_date.py
├── html_extracter.py
├── __pycache__
│   ├── html_extracter.cpython-36.pyc
│   ├── html_extracter.cpython-37.pyc
│   ├── recommend_extracter.cpython-36.pyc
│   └── recommend_extracter.cpython-37.pyc
├── recommend_extracter.py
├── step2_data
│   ├── config.ini
│   ├── html
│   │   ├── error_log_2019-10-09_10-42-41.json
│   │   └── error_log_2019-10-17_14-34-06.json
│   └── recommend
│       └── error_log_2019-10-09_11-10-56.json
├── step2_main_contraoller.py
└── update_history.txt

############descriptions for this code############
1) This code is used to extract the content from the html, then it will insert into the `content` table and `recommend` table step by step
2) For your convenience, I add some exception control in the code, after you run the code, it will tell you whether it has some problem, then you can check the path which the programme print in the terminal, it is categorized by the time when you run the programme.
3) For each update, I will write down the update in the update_history.txt(placed in './step2_extract_html/'), you can check it if you need.

############code instructions############
#Before you run the code, hope you can change some config for your own in the file config.ini(placed in './step2_extract_html/step2_data'). Because the account and the password need to be changed by yourself. Port and the database name, if it is the same with your own database, then you don't need to change it.

#run the code
1)open the terminal, and change the directory to step2_extract_html
2)python step2_main_contraoller.py

############problems############
1) Html of some companies may change, so it is better to check the error and do some update for the extract code.
