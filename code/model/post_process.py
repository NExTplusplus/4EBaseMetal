import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
        
class Post_process():
    def __init__(self):
        pass

    def predict(self, parameters):
        return parameters["Prediction"]
                            

class Post_process_substitution(Post_process):
    def __init__(self):
        super(Post_process_substitution, self).__init__()
    
    def predict(self, parameters):
        X = parameters["Prediction"]
        X_sub = parameters["Substitute"]
        for date in X.index:
            if date in X_sub.index:
                X.loc[date,:] = 1 if X_sub.loc[date,:] == 1 else 0
        return X
        
class Post_process_filter(Post_process):
    def __init__(self):
        super(Post_process_filter, self).__init__()

    def predict(self, parameters):
        X = parameters["Prediction"]
        X_filter = parameters["Filter"]
        ans = {"date":[],"Prediction":[]}
        for date in X.index:
            if X_filter.loc[date,"Filter"] == 1:
                ans['date'].append(date)
                ans['Prediction'].append(X.loc[date,"Prediction"])
        ans = pd.DataFrame(ans)
        ans.set_index("date",inplace = True)
        return ans
