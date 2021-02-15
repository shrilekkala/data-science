import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy

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

# print("In sample error    : " + str(MSE_train_ridge))
# print("Out of sample error: " + str(MSE_test_ridge))

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
    
    # create dictionaries
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
            # training data MSE
            train_preds_ridge = predict_with_estimate(X_train, beta_ridge)
            MSE_train_ridge = np.mean((y_train - train_preds_ridge) ** 2)
            
            # validation data MSE
            test_preds_ridge = predict_with_estimate(X_val, beta_ridge)
            MSE_val_ridge = np.mean((y_val - test_preds_ridge) ** 2)
            
            # store these in the appropriate dictionaries
            train_MSE[i+1].append(MSE_train_ridge)
            val_MSE[i+1].append(MSE_val_ridge)
    
   
    print("Training finished.")

    return train_MSE, val_MSE

lambda_vec = np.linspace(0, 100, num=1001)

train_MSE, val_MSE = cross_val_evaluate(train, 5, lambda_vec)

"""
Consider fold 1, scan penalty parameter
Note to self, choose one that looks nice by setting random seed
"""

plt.title("Plot of MSE errors for over different penalty terms for Ridge Regression [Fold 1]")
plt.plot(lambda_vec, train_MSE[1], label = "Training Errors")
plt.plot(lambda_vec, val_MSE[1], label = "Validation Errors")
plt.legend()
#plt.grid()
plt.xlabel("$\lambda$")
plt.ylabel("MSE")
plt.show()
# print("Optimal λ for Fold 1 is " + str(lambda_vec[np.argmin(val_MSE[1])]))


# The optimal lambdas for for each fold obtained using argmin
for i in range(0,5):
    print("Optimal λ for Fold " + str(i+1) + " is " + str(lambda_vec[np.argmin(val_MSE[i+1])]))

"""
1.2.2
"""
# Compute the average validation MSE over the folds, to get average for each penalty term
average_val_MSE = np.mean([val_MSE[fold] for fold in range(1, 6)], axis = 0)
optimal_lambda = lambda_vec[np.argmin(average_val_MSE)]

"""
??
# So the average in-sample MSE is:
print("Average in-sample MSE:" + str(average_val_MSE[np.argmin(average_val_MSE)]))
"""

# train the model over the whole data set using the optimal_lambda
beta_ridge = ridge_estimate(X_train, Y_train, penalty=optimal_lambda)

# in sample MSE
train_preds_ridge = predict_with_estimate(X_train, beta_ridge)
MSE_train_ridge = np.mean((Y_train - train_preds_ridge) ** 2)

# out of sample MSE
test_preds_ridge = predict_with_estimate(X_test, beta_ridge)
MSE_test_ridge = np.mean((Y_test - test_preds_ridge) ** 2)

print("In sample error    : " + str(MSE_train_ridge))
print("Out of sample error: " + str(MSE_test_ridge))

"""
Discussion
"""

# Differences observed in parameters
abs_diff_beta = np.abs(beta_ml - beta_ridge)

df_beta_parameters = pd.DataFrame(columns=['Linear Regression Parameters', 
                                           'Ridge Regression Parameters',
                                           'Absolute Difference in Parameters'],
                                  index=[['β_%s' %i for i in range(18)]],
                                  data = np.array([beta_ml, beta_ridge, abs_diff_beta]).T)

df_beta_parameters

# Percentage difference in predicted value
p_diff_pred_ml = (Y_test - test_preds)*100/Y_test
p_diff_pred_ridge = (Y_test - test_preds_ridge)*100/Y_test
np.mean((np.abs(p_diff_pred_ridge - p_diff_pred_ml)))
np.max(np.abs(p_diff_pred_ridge - p_diff_pred_ml))

# Further Plots (NB could just use scale-location instead of ridge)
### Residual Plots
res_ml = Y_test - test_preds
res_ridge = Y_test - test_preds_ridge

fig1, ax1 = plt.subplots(ncols = 2, figsize=(10,5))

sns.scatterplot(Y_test, res_ml, ax=ax1[0])
sns.scatterplot(Y_test, res_ridge, ax=ax1[1])

ax1[0].set_xlabel("Y values")
ax1[1].set_xlabel("Y values")
ax1[0].set_ylabel("Residuals")
ax1[1].set_ylabel("Residuals")
ax1[0].set_title('Residual plot for Linear Regression')
ax1[1].set_title('Residual plot for Ridge Regression')


plt.show()

### Scale - Location Plots
fig2, ax2 = plt.subplots(ncols = 2, figsize=(10,5))

#standardise residuals
std_res_ml = res_ml / np.std(res_ml)
std_res_ridge = res_ridge / np.std(res_ridge)

sns.scatterplot(test_preds, std_res_ml, ax=ax2[0])
sns.scatterplot(test_preds_ridge, std_res_ridge, ax=ax2[1])
ax2[0].set_xlabel("Predicted Values")
ax2[1].set_xlabel("Predicted Values")
ax2[0].set_ylabel("Standardised Residuals")
ax2[1].set_ylabel("Standardised Residuals")
ax2[0].set_title('Scale-Location plot for Linear Regression')
ax2[1].set_title('Scale-Location plot for Ridge Regression')

#QQplots??
#NB most QQ plots need statsmodels which we can't use...

fig3, ax3 = plt.subplots(ncols = 2, figsize=(10,5))

scipy.stats.probplot(std_res_ml, dist="norm", plot=ax3[0])
scipy.stats.probplot(std_res_ridge, dist="norm", plot=ax3[1])
ax3[0].get_lines()[0].set_markersize(2)
ax3[1].get_lines()[0].set_markersize(2)

ax3[0].set_title("Normal Probability Plot for Linear Regression")
ax3[1].set_title("Normal Probability Plot for Ridge Regression")














