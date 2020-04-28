import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from sklearn import metrics
import math
import argparse
import json
from copy import copy
from multiprocessing import Pool as pl
from itertools import combinations,product
        
class Post_process():
        """
        horizon: the time horizon of the predict target
        gt: the ground_truth metal name
        date: the last date of the prediction
        window: size for the single model
        """
        def __init__(self,
                    applied_comb,
                    dates,
                    version= ''):
                self.applied_comb = applied_comb
                self.dates = dates
                self.version = version
                self.full_comb = [list(x) for x in list(product(['LME_Al_Spot',"LME_Co_Spot",'LME_Le_Spot','LME_Ni_Spot','LME_Ti_Spot','LME_Zi_Spot'],[1,3,5,10,20,60]))]
        
        @staticmethod
        def read_ensemble(ground_truth, horizon, date, sm_methods, ens_method, hier, versions):
            return pd.read_csv(os.path.join('result','prediction','ensemble','_'.join([ground_truth,date,str(horizon),versions,sm_methods,ens_method,'hier',str(hier)+".csv"])),index_col = 0)

        def predict(self,sm_methods = 'vote:vote:vote',ens_method = 'vote',hier = False,versions = "v1:v1:v1"):
            for date in self.dates.split(','):
                for comb in self.applied_comb:
                    prediction = read_ensemble(comb[0],str(comb[1]),date,sm_methods,ens_method,hier,versions)
                    prediction.to_csv(os.path.join('result','prediction','post_process',"_".join([comb[0],date,str(comb[1])+".csv"])))
                            

class Post_process_substitution(Post_process):
    def __init__(self,applied_comb,dates,version):
        super(Post_process_substitution, self).__init__(self,applied_comb,dates,version)
    
    @staticmethod
    def get_ind_metal(string):
        if 'Al' in string:
            return 'Al'
        elif 'Co' in string:
            return 'Cu'
        elif 'Ni' in string:
            return 'Ni'
        elif 'Le' in string:
            return 'Pb'
        elif 'Ti' in string:
            return 'Xi'
        elif 'Zi' in string:
            return 'Zn'

    def predict(self, sm_methods, ens_method, hier, versions):
        validation_dates = [date.split("-")[0]+"-01-01" if date[5:7] <= "06" else date.split("-")[0]+"-07-01" for date in self.dates.split(',')]
        for date in self.dates.split(','):
            for comb in self.full_comb:
                if comb not in self.applied_comb:
                    super(Post_process_substitution,self).predict(sm_methods,ens_method,hier,versions)
                else:
                    new_date = "".join(date.split("-"))
                    prediction = super(Post_process_substitution,self).read_ensemble(comb[0],str(comb[1]),date,sm_methods,ens_method,hier,versions)
                    if self.version == 'analyst':
                        ground_truth = get_ind_metal(comb[0])
                        f =list(filter(lambda x: new_date in x and len(x.split('_')) >= 6 , os.listdir(os.path.join('code','new_Analyst_Report_Chinese','step4_sentiment_analysis',"predict_result",ground_truth,str(comb[1])))))[0]
                        indicator = pd.read_csv(os.path.join('code','new_Analyst_Report_Chinese','step4_sentiment_analysis',"predict_result",ground_truth,str(comb[1]),f))
                        sub_prediction = indicator[['date','discrete_score']][indicator['discrete_score']!=0.0]
                        sub_prediction.set_index('date',inplace = True)
                    for d in prediction.index:
                        if d in sub_prediction.index:
                            prediction.loc[d,'result'] = 1 if sub_prediction.loc[d,'discrete_score'] == 1 else 0
                    prediction.to_csv(os.path.join('result','prediction','post_process',"_".join([comb[0],date,str(comb[1]),self.version+".csv"])))
    