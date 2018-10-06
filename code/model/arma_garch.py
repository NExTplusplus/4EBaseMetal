from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMAResults
from model.base_predictor import BasePredictor
from arch import arch_model

class ARMA_GARCH(BasePredictor):
  def __init__(self,parameters):
    BasePredictor.__init__(self,parameters)

  def train(self, ts_train):  
    chosen_order = (0,0,0)
    min_aic = None
    for p in (5, 10, 20, 40):
      for q in range(1,6):
        arma_model = ARIMA(ts_train, order = (p,0,q))
        aic = ARMAResults.aic(arma_model)
        if min_aic is None:
          chosen_order = (p,0,q)
          min_aic = aic
        
        if aic < min_aic:
          chosen_order = (p,0,q)
          min_aic = aic

      final_model = arch_model(ts_train,p=p,q=q)

    self.model = final_model
  



