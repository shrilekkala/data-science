import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# import training set
test_data = pd.read_csv('./regression_test.csv', header=None)
train_data = pd.read_csv('./regression_train.csv', header=None)

#print(test_data.tail())
#print(train_data.tail())

# (NB column of ones for the intercept is already in the data)

X_train = np.array(train_data.iloc[:,:-1])
Y_train = np.array(train_data.iloc()[:,-1])

X_test = np.array(test_data.iloc[:,:-1])
Y_test = np.array(test_data.iloc()[:,-1])


def max_lik_estimate(X, y):
    
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # returns: maximum likelihood parameters (D x 1)
    
    N, D = X.shape
    

    beta_ml = np.linalg.solve(X.T @ X, X.T @ y)
    return beta_ml

def predict_with_estimate(X_test, beta):
    # X_test: K x D matrix of test inputs
    # beta: D x 1 vector of parameters
    # returns: prediction of f(X_test); K x 1 vector
    
    prediction = X_test @ beta
    
    return prediction 

# parameters of the model
beta_ml = max_lik_estimate(X_train,Y_train)

# in sample MSE
train_preds = predict_with_estimate(X_train, beta_ml)
MSE_train = np.mean((Y_train - train_preds) ** 2)

# out of sample MSE
test_preds = predict_with_estimate(X_test, beta_ml)
MSE_test = np.mean((Y_test - test_preds) ** 2)

print("In sample error    : " + str(MSE_train))
print("Out of sample error: " + str(MSE_test))

# possible outliers in training set causing MSE to be higher
# further, we haven't done any K-fold cross validation, 
# this could solve problems where the training set is not representative