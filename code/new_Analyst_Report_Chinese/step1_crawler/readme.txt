############Tree for this folder############
├── crawler.py
├── readme.txt
├── step1_data
│   ├── chromedriver
│   ├── chromedriver.exe
│   ├── config.ini
│   └── error_link.json
└── update_history.txt

############descriptions for this code############
1)This code is used to crawl the url from the given website, then it will insert into the html table.
2)For your convernience, I add some exception control in the code, after you run the code, if the error_link.json(placed in './step1_crawler/step1_data') have some url, that is to say, it meet some problem, then you can use the different command to retry, which i state follow.
3)For each update, I will state the things I update in the update_history.txt(placed in '/step1_crawler/'), you can check it if you need it.

############code instructions############
#Before you run the code, hope you can change some config for your own in the file config.ini(placed in './step1_crawler/step1_data'). Because the account and the password need to be changed by yourself. Port and the database name, if it is the same with your own database, then you don't need to change it.

#if you want to crawl the data for something like daily update
1)open the terminal, and change the directory to step1_crawler
2)python crawler.py run

#if you want to deal with the error links lie in the error_link.json
1)open the terminal, and change the directory to step1_crawler
2)python crawler.py check

############problems############
1)Afraid that you may meet some problems with the version of the chromedriver(placed in './step1_crawler/step1_data' here). Then you need to check the version of the chrome, then find the corresponding chromedriver from this website(http://chromedriver.storage.googleapis.com/index.html).


