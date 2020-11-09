import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

'''
    This file contains the post processing methods
'''
        
class Post_process():
    def __init__(self):
        pass

    def predict(self, parameters):
        return parameters["Prediction"]
                            

#post process which substitutes predictions with a analyst report predictions
class Post_process_substitution(Post_process):
    def __init__(self):
        super(Post_process_substitution, self).__init__()
    
    def predict(self, parameters):
        X = parameters["Prediction"]
        X_sub = parameters["Substitute"]
        X_unc = parameters["Uncertainty"]
        #loop through all dates
        for date in X.index:
            #if date is in analyst report predictions
            if date in X_sub.index:
                print(date)
                X.loc[date,"result"] = 1 if X_sub.loc[date,"discrete_score"] == 1 else 0
                X_unc.loc[date,"uncertainty"] = 0.5
        return X,X_unc


#post process which filters predictions for low confidence in regression and conflicting results between classificaiton and regression
class Post_process_filter(Post_process):
    def __init__(self):
        super(Post_process_filter, self).__init__()

    def predict(self, parameters):
        X = parameters["Prediction"]
        X_filter = parameters["Filter"]
        ans = {"date":[],"Prediction":[]}
        #loop through all dates
        for date in X.index:
            #only change values if the final signal is triggered
            if X_filter.loc[date,"Filter"] == 1:
                ans['date'].append(date)
                ans['Prediction'].append(X.loc[date,"Prediction"])
        ans = pd.DataFrame(ans)
        ans.set_index("date",inplace = True)
        return ans
