from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from model.base_predictor import BasePredictor
from joblib import dump
from sklearn.externals import joblib
import numpy as np
 
class LogReg(BasePredictor):
  def __init__(self,parameters):
    BasePredictor.__init__(self,parameters)
    self.model = None

  def train(self, X_tr, Y_tr, parameters):  

    if self.model == None:
      self.model = LogisticRegression(  C=parameters['C'],penalty=parameters['penalty'],tol = parameters['tol'],solver = parameters["solver"],
                                        max_iter = parameters['max_iter'], verbose = parameters["verbose"], warm_start = parameters["warm_start"])
    else:
      self.model.set_params(**parameters)
    self.model.fit(X_tr, Y_tr)

  def save(self, version, gt, horizon, lag):
    joblib.dump(self.model,version+"_"+gt+"_"+str(horizon)+"_"+str(lag)+"_lr_"+'.pkl')

    # return None
    
  def load(self, version, gt, horizon, lag):
    model = joblib.load(version+"_"+gt+"_"+str(horizon)+"_"+str(lag)+"_lr_"+'.pkl')
    return model
  def log_loss(self,X,y_true):
    return log_loss(y_true,self.model.predict_proba(X))


  def test(self, X_tes, Y_tes):
    pred = self.model.predict(X_tes)
    return accuracy_score(Y_tes, pred)

  def predict(self,X):
    return self.model.predict(X)

  def predict_proba(self,X):
    return self.model.predict_proba(X)

  def coef(self):
    return self.model.coef_

  def intercept_(self):
    return self.model.intercept_

  def set_params(self,params):
    self.model.set_params(params)
  
  def n_iter(self):
    return self.model.n_iter_[0]
  
  

  


    

  
