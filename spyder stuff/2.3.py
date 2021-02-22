import numpy as np
# import matplotlib.pyplot as plt
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

# convert the 0s to -1s so the labels are 1 and -1
diag_map = {1.0: 1.0, 0.0: -1.0}
df_Y_train_svm = df_Y_train.map(diag_map)
df_Y_test_svm = df_Y_test.map(diag_map)

Y_train_svm = np.array(df_Y_train_svm.iloc()[:])
Y_test_svm = np.array(df_Y_test_svm.iloc()[:])

def standardise(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_std = (X - mu) / sigma
    return X_std

# NB it appears that the data is already standardised:
np.min(X_train, axis = 0)
np.max(X_train, axis = 0)

"""
# Standardise the data (NB in the notebook, they standardised both together)
df_X_train_std = standardise(df_X_train)
X_train_std = np.array(df_X_train_std)

df_X_test_std = standardise(df_X_test)
X_test_std = np.array(df_X_test_std)
"""

"""
Standardise both together 
df_full_data = pd.concat([train_data, test_data])
full_X = np.array(full_data.iloc[:,:-1])
full_X_std = standardise(full_X)
X_train_full_std = full_X_std[:800, :]
X_test_full_std = full_X_std[800:, :]

# train the model
W_full = sgd(X_train_full_std, Y_train_svm, max_iterations=2000, stop_criterion=0.01, learning_rate=1e-3, regul_strength=1e3, print_outcome=True)
print("Training finished.")

print("Accuracy on train set: {}".format(score(W_full, X_train_full_std, Y_train_svm)))
print("Accuracy on test set: {}".format(score(W_full, X_test_full_std, Y_test_svm)))
"""

# insert 1 in every row for the intercept b
df_X_train_std.insert(loc=len(df_X_train_std.columns), column='intercept', value=1)
df_X_test_std.insert(loc=len(df_X_test_std.columns), column='intercept', value=1)

# Hinge loss function
def compute_cost(W, X, y, regul_strength=1e5):
    n = X.shape[0]
    distances = 1 - y * np.dot(X, W)
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge = regul_strength * (np.sum(distances) / n)
    
    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge
    return cost

# calculate gradient of cost
def calculate_cost_gradient(W, X_batch, y_batch, regul_strength=1e5):
    # if only one example is passed
    if type(y_batch) == np.float64:
        y_batch = np.asarray([y_batch])
        X_batch = np.asarray([X_batch])  # gives multidimensional array
    
    distance = 1 - (y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))
    
    for ind, d in enumerate(distance):
        if max(0, d)==0:
            di = W
        else:
            di = W - (regul_strength * y_batch[ind] * X_batch[ind])
        dw += di
    
    dw = dw/len(y_batch)  # average
    return dw

def sgd(X, y, max_iterations=2000, stop_criterion=0.01, learning_rate=1e-5, regul_strength=1e5, print_outcome=False):
    # initialise zero weights
    weights = np.zeros(X.shape[1])
    nth = 0
    # initialise starting cost as infinity
    prev_cost = np.inf
    
    # stochastic gradient descent
    for iteration in range(1, max_iterations):
        # shuffle to prevent repeating update cycles
        np.random.shuffle([X, y])
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, y[ind], regul_strength) ## <-- EDIT THIS LINE - DONE
            weights = weights - (learning_rate * ascent)
    
        # convergence check on 2^n'th iteration
        if iteration==2**nth or iteration==max_iterations-1:
            # compute cost
            cost = compute_cost(weights, X, y, regul_strength)  ## <-- EDIT THIS LINE - DONE
            if print_outcome:
              print("Iteration is: {}, Cost is: {}".format(iteration, cost))
            # stop criterion
            if abs(prev_cost - cost) < stop_criterion * prev_cost:
                return weights
            
            prev_cost = cost
            nth += 1
    
    return weights

# train the model
W = sgd(X_train, Y_train_svm, max_iterations=2000, stop_criterion=0.01, learning_rate=1e-3, regul_strength=1e3, print_outcome=True)
print("Training finished.")

# function to evaluate the mean accuracy
def score(W, X, y):
    y_preds = np.array([])
    for i in range(X.shape[0]):
        y_pred = np.sign(np.dot(X[i], W))
        y_preds = np.append(y_preds, y_pred)
    
    return np.float(sum(y_preds == y)) / float(len(y))

print("Accuracy on train set: {}".format(score(W, X_train, Y_train_svm)))
print("Accuracy on test set: {}".format(score(W, X_test, Y_test_svm)))


