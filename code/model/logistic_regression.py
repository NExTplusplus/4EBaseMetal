from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from model.base_predictor import BasePredictor
import matplotlib.pyplot as plt
import tkinter
from sklearn.model_selection import learning_curve, validation_curve
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

  def log_loss(self,X,y_true):
    return log_loss(y_true,self.model.predict_proba(X))


  def test(self, X_tes, Y_tes):
    pred = self.model.predict(X_tes)
    return accuracy_score(Y_tes, pred)

  def predict(self,X):
    return self.model.predict(X)

  def coef(self):
    return self.model.coef_

  def intercept_(self):
    return self.model.intercept_

  def set_params(self,params):
    self.model.set_params(params)
  
  def plot(self,X,y,ylim= None):
    plt.figure()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(self.model,X,y,scoring = "neg_log_loss",train_sizes=np.linspace(.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
  
  # def validation_curve(self,X,y):
  #   validation_curve(self.model,X,y,scoring = "log_loss")

  


    

  
