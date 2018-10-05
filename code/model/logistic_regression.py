from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from model.base_predictor import BasePredictor

class LogReg(BasePredictor):
  def __init__(self,parameters):
    BasePredictor.__init__(self,parameters)

  def train(self, X_tr, Y_tr):  
    model = LogisticRegression(parameters)
    model.fit(X_tr, Y_tr)

    self.model = model

  def test(self, X_tes, Y_tes):
    pred = self.model.predict(X_tes)
    return accuracy_score(Y_tes, pred)
  


    

  
