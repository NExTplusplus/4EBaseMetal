from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from model.base_predictor import BasePredictor
from sklearn.metrics import log_loss
import numpy as np
 
class LogReg(BasePredictor):
  def __init__(self,parameters):
    BasePredictor.__init__(self,parameters)

  def train(self, X_tr, Y_tr, parameters):  
    model = LogisticRegression( C=parameters['C'],penalty=parameters['penalty'],tol = parameters['tol'],solver = parameters["solver"],
                                max_iter = parameters['max_iter'], verbose = parameters["verbose"], warm_start = True)
    model.fit(X_tr, Y_tr)

    self.model = model

  def test(self, X_tes, Y_tes):
    pred = self.model.predict(X_tes)
    return accuracy_score(Y_tes, pred)

  def predict(self,X):
    return self.model.predict(X)

  def coef(self):
    return np.transpose(self.model.coef_)
  


    

  
