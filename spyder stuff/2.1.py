import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import training set
test_data = pd.read_csv('./classification_test.csv', header=None)
train_data = pd.read_csv('./classification_train.csv', header=None)

df_X_train = train_data[train_data.columns[:-1]]
df_Y_train = train_data[train_data.columns[-1]]

df_X_test = test_data[test_data.columns[:-1]]
df_Y_test = test_data[test_data.columns[-1]]

X_train = np.array(train_data.iloc[:,:-1])
Y_train = np.array(train_data.iloc()[:,-1])

X_test = np.array(test_data.iloc[:,:-1])
Y_test = np.array(test_data.iloc()[:,-1])

# transpose for use in logistic regression
X_train_logistic = X_train.T
X_test_logistic = X_test.T

def logistic(x):
    return 1 / (1 + np.exp(-x))

# function to compute y lof
def predict_log(X, beta, beta_0):
  y_log = logistic(beta.T @ X + beta_0)
  return y_log

# intialise beta and beta_0
def initialise(d):
    beta = np.zeros(shape=(d, 1), dtype=np.float32)
    beta_0 = 0

    return beta, beta_0

# forward pass of cost function and its derivatives
def propagate(X, y, beta, beta_0):
  """
  Arguments:
  X: data of size (d, n)
  y: true label vector of size (1, n)
  beta: parameters, a numpy array of size (d, 1)
  beta_0: offset, a scalar

  Returns:
  cost: negative log-likelihood cost for logistic regression
  dbeta: gradient of the loss with respect to beta
  dbeta_0: gradient of the loss with respect to beta_0
  """
  n = X.shape[1]
  y_log = predict_log(X, beta, beta_0)

  # cost function
  cost = (-1) * np.mean(np.multiply(y, np.log(y_log)) + np.multiply((1-y), np.log(1-y_log)), axis = 1) 

  # derivatives
  dbeta = (1/n) * X @ (y_log - y).T
  dbeta_0 =  np.mean(y_log - y, axis = 1)

  assert(dbeta.shape==beta.shape)
  assert(dbeta_0.dtype==float)
  cost = np.squeeze(cost)
  assert(cost.shape==())
  
  # store gradients in a dictionary
  grads = {"dbeta": dbeta, "dbeta_0": dbeta_0}
  
  return grads, cost

# function that performs the optimisation of the cost function as required
def optimise(X, y, beta, beta_0, num_iterations=1000, learning_rate=0.005, print_cost=False):
  """
  Arguments:
  X: data of size (d, n)
  y: true label vector of size (1, n)
  beta: parameters, a numpy array of size (d, 1)
  beta_0: offset, a scalar
  num_iterations: number of iterations gradient descent shall update the parameters
  learning_rate: step size in updating procedure
  print_cost: whether to print the cost every 100 iterations or not

  Returns:
  params: dictionary containing the parameters beta and offset beta_0
  grads: dictionary containing the gradients
  costs: list of all the costs computed during the optimisation (can be used to plot the learning curve).
  """
  costs = []
    
  for i in range(num_iterations):

      # calculate cost and gradients
      grads, cost = propagate(X, y, beta, beta_0)
      
      # retrieve derivatives from grads
      dbeta = grads["dbeta"]
      dbeta_0 = grads["dbeta_0"]
      
      # updating procedure
      beta = beta - learning_rate * dbeta
      beta_0 = beta_0 - learning_rate * dbeta_0
      
      # record the costs
      if i % 100 == 0:
          costs.append(cost)
      
      # print the cost every 100 iterations
      if print_cost and i % 100 == 0:
          print ("cost after iteration %i: %f" %(i, cost))
  
  # save parameters and gradients in dictionary
  params = {"beta": beta, "beta_0": beta_0}
  grads = {"dbeta": dbeta, "dbeta_0": dbeta_0}
  
  return params, grads, costs

# function that predicts the labels for a given data given the beta parameters
def predict(X_test, beta, beta_0, decision_threshold):
    """
    Arguments:
    X_test: test data of size (d, n)
    beta: parameters, a numpy array of size (d, 1)
    beta_0: offset, a scalar
    decision_threshold: a scalar in [0,1]
    
    Returns:
    y_pred: vector containing all binary predictions (0/1) for the examples in X_test
    """
    n = X_test.shape[1]
    y_pred = np.zeros((1,n))
    beta = beta.reshape(X_test.shape[0], 1)
    
    # compute vector y_log predicting the probabilities
    y_log = predict_log(X_test, beta, beta_0)
    
    for i in range(y_log.shape[1]):
        
        # convert probabilities y_log to actual predictions y_pred
        if y_log[0, i] > decision_threshold:
            y_pred[0, i] = 1  ## <-- EDIT THIS LINE - DONE
        else:
            y_pred[0, i] = 0  ## <-- EDIT THIS LINE - DONE
    
    assert(y_pred.shape==(1, n))
    
    return y_pred

decision_threshold = 0.5

# One function that has all hyperparameters as arguments
def model(X_train, y_train, X_test, y_test, decision_threshold, num_iterations=5000, learning_rate=0.5, print_cost=False):
    # initialize parameters with zeros
    beta, beta_0 = initialise(X_train.shape[0])
    
    # gradient descent
    parameters, grads, costs = optimise(X_train, y_train, beta, beta_0, num_iterations, learning_rate, print_cost=print_cost)
    
    # retrieve parameters beta and beta_0 from dictionary "parameters"
    beta = parameters["beta"]
    beta_0 = parameters["beta_0"]
    
    # predict test and train set examples
    y_pred_test = predict(X_test, beta, beta_0, decision_threshold)
    y_pred_train = predict(X_train, beta, beta_0, decision_threshold)
    
    # obtain train/test Errors
    train_accuracy = 100 - np.mean(np.abs(y_pred_train - y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(y_pred_test - y_test)) * 100
    
    # saving all information
    d = {"costs": costs, 
         "y_pred_test": y_pred_test, 
         "y_pred_train": y_pred_train, 
         "beta": beta, "beta_0": beta_0, 
         "learning_rate": learning_rate, 
         "num_iterations": num_iterations,
         "train_acc": train_accuracy,
         "test_acc": test_accuracy}
    
    return d

d = model(X_train_logistic, Y_train, X_test_logistic, Y_test, decision_threshold=0.5, num_iterations=5000, learning_rate=0.1)
print("train accuracy: {} %".format(d["train_acc"]))
print("test accuracy: {} %".format(d["test_acc"]))

costs = np.squeeze(d['costs'])
plt.figure(figsize=(12,8))
plt.ylabel('Cost', size=20)
plt.xlabel('Iterations (in hundreds)', size=20)
plt.title("Learning rate = " + str(d["learning_rate"]), size=20)
plt.plot(costs)

# a function to create a grid given two vectors
def make_grid(v1,v2):
    grid = np.array(np.meshgrid(v1, v2))
    return grid.T.reshape(-1,2)

def cross_val_split(data, num_folds):
  fold_size = int(len(data) / num_folds)
  data_perm = np.random.permutation(data)
  folds = []
  for k in range(num_folds):
    folds.append(data_perm[k*fold_size:(k+1)*fold_size, :])

  return folds

def cross_val_evaluate_logistic(folds, grid):
    # create dictionaries
    train_acc = {1:[], 2:[], 3:[], 4:[], 5:[]}
    val_acc = {1:[], 2:[], 3:[], 4:[], 5:[]}

    for i in range(len(folds)):
      
        print('Fold', i+1)
        # define the training set (i.e. selecting all folds and deleting the one used for validation)
        train_set = np.delete(np.asarray(folds).reshape(len(folds), folds[0].shape[0], folds[0].shape[1]), i, axis=0)
        train_folds = train_set.reshape(len(train_set)*train_set[0].shape[0], train_set[0].shape[1])
        X_train = train_folds[:,:-1].T
        y_train = train_folds[:, -1]
        
        # define the validation set
        val_fold = folds[i]
        X_val = val_fold[:,:-1].T
        y_val = val_fold[:, -1]
    
        # train the model and obtain the parameters for each lambda
        # GRID SEARCH
        for parameters in grid:
            # print(parameters)
            
            # train the model
            model_dict = model(X_train, y_train, X_val, y_val, decision_threshold=parameters[1], num_iterations=5000, learning_rate=parameters[0])
            
            # obtain the accuracies and store these in the appropriate dictionaries
            train_acc[i+1].append(model_dict["train_acc"])
            val_acc[i+1].append(model_dict["test_acc"])
    
    print("Training finished.")
    return train_acc, val_acc

# Aggregate the X and Y data into one array to be used for cross validation
train = np.hstack((X_train, Y_train[:, np.newaxis]))
test = np.hstack((X_test, Y_test[:, np.newaxis]))

# Generate the folds
folds = cross_val_split(train, 5)



# Create the grid for grid search
learning_rate_vec = np.arange(1,11) / 10
decision_threshold_vec = np.arange(1,11) / 10
grid = make_grid(learning_rate_vec, decision_threshold_vec)

#NB takes a long time
train_acc, val_acc = cross_val_evaluate_logistic(folds, grid)

# Compute the average validation accuracy over the folds, to get average for each penalty term
average_val_acc = np.mean([val_acc[fold] for fold in range(1, 6)], axis = 0)
optimal_parameters = grid[np.argmax(average_val_acc)]

print("Optimal Decision Threshold: " + str(optimal_parameters[1]))
print("Optimal Learning Rate    : " + str(optimal_parameters[0]))

# Mean Accuracies
# Retrain the model using the optimal paramaters
d = model(X_train_logistic, Y_train, X_test_logistic, Y_test, decision_threshold=optimal_parameters[1],
          num_iterations=5000, learning_rate=optimal_parameters[0])
print("train accuracy: {} %".format(d["train_acc"]))
print("test accuracy: {} %".format(d["test_acc"]))

