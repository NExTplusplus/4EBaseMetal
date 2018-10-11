import numpy as np



def objective_function(estimator,X,y):
    t_coef = np.transpose(estimator.coef())
    intercept = estimator.intercept_()
    estimation = np.matmul(X,t_coef) + intercept
    rgl = np.dot(estimator.coef(),t_coef)
    cost = sum(np.log(np.exp(-np.multiply(y,estimation))+1))
    ans = cost+rgl
    return ans


