import numpy as np
import matplotlib.pyplot as plt
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

def ridge_estimate(X, y, penalty):
    
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # returns: maximum likelihood parameters (D x 1)
    
    N, D = X.shape
    I = np.identity(D)
    beta_ridge = np.linalg.solve(X.T @ X + penalty * I, X.T @ y)
    return beta_ridge

lambda_penalty = 0

# parameters of the model
beta_ridge = ridge_estimate(X_train, Y_train, penalty=lambda_penalty)

# in sample MSE
train_preds_ridge = predict_with_estimate(X_train, beta_ridge)
MSE_train_ridge = np.mean((Y_train - train_preds_ridge) ** 2)

# out of sample MSE
test_preds_ridge = predict_with_estimate(X_test, beta_ridge)
MSE_test_ridge = np.mean((Y_test - test_preds_ridge) ** 2)

print("In sample error    : " + str(MSE_train_ridge))
print("Out of sample error: " + str(MSE_test_ridge))

def cross_val_split(data, num_folds):
  fold_size = int(len(data) / num_folds)
  data_perm = np.random.permutation(data)
  folds = []
  for k in range(num_folds):
    folds.append(data_perm[k*fold_size:(k+1)*fold_size, :])

  return folds

# Aggregate the X and Y data into one array to be used for cross validation
train = np.hstack((X_train, Y_train[:, np.newaxis]))
test = np.hstack((X_test, Y_test[:, np.newaxis]))

def cross_val_evaluate(data, num_folds, lambda_vec):
  
    folds = cross_val_split(data, num_folds)
    
    train_MSE = {1:[], 2:[], 3:[], 4:[], 5:[]}
    val_MSE = {1:[], 2:[], 3:[], 4:[], 5:[]}

    for i in range(len(folds)):
      
        print('Fold', i+1)
        # define the training set (i.e. selecting all folds and deleting the one used for validation)
        train_set = np.delete(np.asarray(folds).reshape(len(folds), folds[0].shape[0], folds[0].shape[1]), i, axis=0)
        train_folds = train_set.reshape(len(train_set)*train_set[0].shape[0], train_set[0].shape[1])
        X_train = train_folds[:,:-1]
        y_train = train_folds[:, -1]
        
        # define the validation set
        val_fold = folds[i]
        X_val = val_fold[:,:-1]
        y_val = val_fold[:, -1]
    
        # train the model and obtain the parameters for each lambda
        for pen in lambda_vec:
            # print(pen)
            
            beta_ridge = ridge_estimate(X_train, y_train, penalty=pen)
            
            # evaluate
            # in sample MSE
            train_preds_ridge = predict_with_estimate(X_train, beta_ridge)
            MSE_train_ridge = np.mean((y_train - train_preds_ridge) ** 2)
            
            # out of sample MSE
            test_preds_ridge = predict_with_estimate(X_val, beta_ridge)
            MSE_test_ridge = np.mean((y_val - test_preds_ridge) ** 2)
            
            # store these in the appropriate dictionaries
            train_MSE[i+1].append(MSE_train_ridge)
            val_MSE[i+1].append(MSE_test_ridge)
    
   
    print("Training finished.")

    return train_MSE, val_MSE

lambda_vec = np.linspace(0, 100, 200)

train_MSE, val_MSE = cross_val_evaluate(train, 5, lambda_vec)

"""
Consider fold 1, scan penalty parameter
"""

train_errors_fold1 = train_MSE[1]
val_errors_fold1 = val_MSE[1]

for i in range(5):
    plt.title("Plot of MSE errors for over different penalty terms for Ridge Regression [Fold 1]" + str(i))
    plt.plot(lambda_vec, train_MSE[i+1], label = "Training Errors")
    plt.plot(lambda_vec, val_MSE[i+1], label = "Validation Errors")
    plt.legend()
    plt.grid()
    plt.xlabel("Penalty Term: $\lambda$")
    plt.ylabel("MSE")
    plt.show()





