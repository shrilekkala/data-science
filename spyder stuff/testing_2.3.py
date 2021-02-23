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
df_full_data = pd.concat([df_X_train, df_X_test])
full_X = np.array(df_full_data.iloc[:,:])
full_X_std = standardise(full_X)
X_train_full_std = full_X_std[:800, :]
X_test_full_std = full_X_std[800:, :]

# add a column of 1s
X_train_full_std_svm = np.hstack((X_train_full_std, np.ones(800)[:, np.newaxis]))
X_test_full_std_svm = np.hstack((X_test_full_std, np.ones(200)[:, np.newaxis]))

# train the model
W_full = sgd(X_train_full_std_svm, Y_train_svm, max_iterations=2000, stop_criterion=0.01, learning_rate=1e-3, regul_strength=1e3, print_outcome=True)
print("Training finished.")

print("Accuracy on train set: {}".format(score(W_full, X_train_full_std_svm, Y_train_svm)))
print("Accuracy on test set: {}".format(score(W_full, X_test_full_std_svm, Y_test_svm)))

# insert 1 in every row for the intercept b
df_X_train_std.insert(loc=len(df_X_train_std.columns), column='intercept', value=1)
df_X_test_std.insert(loc=len(df_X_test_std.columns), column='intercept', value=1)
"""
# insert 1 in every row for the intercept b
df_X_train_svm = df_X_train.copy()
df_X_test_svm = df_X_test.copy()
df_X_train_svm.insert(loc=len(df_X_train.columns), column='intercept', value=1)
df_X_test_svm.insert(loc=len(df_X_test.columns), column='intercept', value=1)

X_train_svm = np.array(df_X_train_svm)
X_test_svm = np.array(df_X_test_svm)

# Hinge loss function
def compute_cost(W, X, y, regul_strength=1e5, rbf_kernel=False, sigma=None):
    n = X.shape[0]
    
    b_vec = np.dot(X[:, -1], W[-1])
    
    if rbf_kernel:
        # Separate b
        distances = 1 - np.multiply(y, (rbf(W[:-1], X[:, :-1], sigma) + b_vec))
    else:
        distances = 1 - y * np.dot(X, W)
    
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge = regul_strength * (np.sum(distances) / n)
    
    # calculate cost
    if rbf_kernel:
        cost = 1 / 2 * rbf(W[:-1], W[:-1], sigma) + hinge
    else:
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

def sgd(X, y, max_iterations=2000, stop_criterion=0.01, learning_rate=1e-5, regul_strength=1e5, print_outcome=False, rbf_kernel=False, sigma=None):
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
            ascent = calculate_cost_gradient(weights, x, y[ind], regul_strength)
            weights = weights - (learning_rate * ascent)
    
        # convergence check on 2^n'th iteration
        if iteration==2**nth or iteration==max_iterations-1:
            # compute cost
            cost = compute_cost(weights, X, y, regul_strength=1e5, rbf_kernel=rbf_kernel, sigma=sigma)
            if print_outcome:
                print("Iteration is: {}, Cost is: {}".format(iteration, cost))
            # stop criterion
            if abs(prev_cost - cost) < stop_criterion * prev_cost:
                return weights
            
            prev_cost = cost
            nth += 1
    
    return weights

# train the model
# NB using a v large C here
## W = sgd(X_train_svm, Y_train_svm, max_iterations=2000, stop_criterion=0.01, learning_rate=1e-3, regul_strength=1e20, print_outcome=True)
print("Training finished.")

def sign(n):
    if n == 0:
        sgn = 1
    else:
        sgn = np.sign(n)
    return sgn

# function to evaluate the mean accuracy
def score(W, X, y, model_train_data, rbf_kernel = False, sigma = None, return_preds = False):
    model_X_train = model_train_data
    
    y_preds = np.array([])
    
    b_vec = np.dot(model_X_train[:, -1], W[-1])
    
    for i in range(X.shape[0]):
        if rbf_kernel:
            # Separate b   
            y_pred = sign(rbf(X[i][:-1], W[:-1], sigma) + b_vec[i])
        else:
            y_pred = sign(np.dot(X[i], W))
        
        y_preds = np.append(y_preds, y_pred)
        
    if return_preds:
        return np.float(sum(y_preds == y)) / float(len(y)), y_preds
    else:
        return np.float(sum(y_preds == y)) / float(len(y))
    
    
## print("Accuracy on train set: {}".format(score(W, X_train_svm, Y_train_svm)))
## print("Accuracy on test set: {}".format(score(W, X_test_svm, Y_test_svm)))

# definte the radial basis function
def rbf(x, y, sigma):
    return np.exp(-(np.linalg.norm(x - y)**2)/(sigma))

sigma_param = 0.01

# train the model
# NB using a v large C here
W_rbf = sgd(X_train_svm, Y_train_svm, max_iterations=2000,
                   stop_criterion=0.01, learning_rate=1e-3, regul_strength=1e20,
                   print_outcome=True, rbf_kernel=True, sigma=sigma_param)
print("Training finished.")


print("RBF Accuracy on train set: {}".format(score(W_rbf, X_train_svm, Y_train_svm, X_train_svm, rbf_kernel=True, sigma=0.001)))
print("RBF Accuracy on test set: {}".format(score(W_rbf, X_test_svm, Y_test_svm, X_train_svm, rbf_kernel=True, sigma=0.001)))

_, y_preds_train_rbf = score(W_rbf, X_train_svm, Y_train_svm, X_train_svm, rbf_kernel=True, sigma=sigma_param, return_preds = True)
_, y_preds_test_rbf = score(W_rbf, X_train_svm, Y_train_svm, X_train_svm, rbf_kernel=True, sigma=sigma_param, return_preds = True)