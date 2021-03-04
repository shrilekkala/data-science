import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(1024)

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
def compute_cost(W, X, y, regul_strength=1e5, rbf_kernel=False, sigma=None, b=None):
    n = X.shape[0]
    
    # b_vec = np.dot(X[:, -1], W[-1])
    
    if rbf_kernel:
        # Separate b
        distances = 1 - y * (rbf(W, X, sigma) + b)
    else:
        distances = 1 - y * np.dot(X, W)
    
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge = regul_strength * (np.sum(distances) / n)
    
    # calculate cost
    if rbf_kernel:
        cost = 1 / 2 * np.dot(W, W) + hinge
    else:
        cost = 1 / 2 * np.dot(W, W) + hinge
    return cost

# calculate gradient of cost
def calculate_cost_gradient(W, X_batch, y_batch, regul_strength=1e5, rbf_kernel=False, sigma=None, b=None):
    # if only one example is passed
    if type(y_batch) == np.float64:
        y_batch = np.asarray([y_batch])
        X_batch = np.asarray([X_batch])  # gives multidimensional array
    
    if rbf_kernel:
        distance = 1 - (y_batch * (rbf(X_batch, W, sigma) + b))
    else:
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

def sgd(X, y, max_iterations=2000, stop_criterion=0.01, learning_rate=1e-5, regul_strength=1e5,
        print_outcome=False, rbf_kernel=False, sigma=None, b=None):
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
            ascent = calculate_cost_gradient(weights, x, y[ind], regul_strength, rbf_kernel, sigma=sigma, b=b)
            weights = weights - (learning_rate * ascent)
    
        # convergence check on 2^n'th iteration
        if iteration==2**nth or iteration==max_iterations-1:
            # compute cost
            cost = compute_cost(weights, X, y, regul_strength=1e5, rbf_kernel=rbf_kernel, sigma=sigma, b=b)
            if print_outcome:
                print("Iteration is: {}, Cost is: {}".format(iteration, cost))
            # stop criterion
            if abs(prev_cost - cost) < stop_criterion * prev_cost:
                return weights
            
            prev_cost = cost
            nth += 1
    
    return weights

def sign(n):
    if n == 0:
        sgn = 1
    else:
        sgn = np.sign(n)
    return sgn

# function to evaluate the mean accuracy
def score(W, X, y, model_train_data, rbf_kernel = False, sigma = None, return_preds = False, b=None):
    model_X_train = model_train_data
    
    y_preds = np.array([])
    
    b_vec = np.dot(model_X_train[:, -1], W[-1])
    
    for i in range(X.shape[0]):
        if rbf_kernel:
            # Separate b   
            y_pred = sign((rbf(X[i], W, sigma) + b))
            # print((rbf(X[i], W, sigma) + b))
        else:
            y_pred = sign(np.dot(X[i], W))
        
        y_preds = np.append(y_preds, y_pred)
        
    if return_preds:
        return np.float(sum(y_preds == y)) / float(len(y)), y_preds
    else:
        return np.float(sum(y_preds == y)) / float(len(y))

"""
Linear
"""

# train the model
# NB using a v large C here
W = sgd(X_train_svm, Y_train_svm, max_iterations=2000, stop_criterion=0.01, learning_rate=1e-3, regul_strength=1e20, print_outcome=True)
print("Training finished.")

accuracy_train, y_preds_train = score(W, X_train_svm, Y_train_svm, X_train_svm, return_preds = True)
accuracy_test, y_preds_test = score(W, X_test_svm, Y_test_svm, X_train_svm, return_preds = True)

print("Accuracy on train set: {}".format(accuracy_train))
print("Accuracy on test set: {}".format(accuracy_test))

"""
rbf
"""

# definte the radial basis function
def rbf(x, y, sigma):
    return np.exp(-(np.linalg.norm(x - y)**2)/(sigma))

# 0.1 to 1.1
sigma_param = 0.3
b_param = -0.1

# train the model
# NB using a v large C here
W_rbf = sgd(X_train, Y_train, max_iterations=500,
                   stop_criterion=0.01, learning_rate=1e-8, regul_strength=1e5,
                   print_outcome=True, rbf_kernel=True, sigma=sigma_param, b=b_param)
print("Training finished.")

print(W_rbf)

accuracy_train_rbf, y_preds_train_rbf = score(W_rbf, X_train, Y_train_svm, X_train,
                                              rbf_kernel=True, sigma=sigma_param, return_preds = True, b=b_param)
accuracy_test_rbf, y_preds_test_rbf = score(W_rbf, X_test, Y_test_svm, X_train,
                                            rbf_kernel=True, sigma=sigma_param, return_preds = True, b=b_param)

print("RBF Accuracy on train set: {}".format(accuracy_train_rbf))
print("RBF Accuracy on test set: {}".format(accuracy_test_rbf))

print(np.unique(y_preds_train_rbf))
print(np.unique(y_preds_test_rbf))


"""
grid search - 5 fold cross validation
"""


# Generate the folds
def cross_val_split(data, num_folds):
  fold_size = int(len(data) / num_folds)
  data_perm = np.random.permutation(data)
  folds = []
  for k in range(num_folds):
    folds.append(data_perm[k*fold_size:(k+1)*fold_size, :])

  return folds

# Aggregate the X and Y data into one array to be used for cross validation
train = np.hstack((X_train, Y_train[:, np.newaxis]))

folds = cross_val_split(train, 5)


def cross_val_evaluate_svm(folds, sigma_param):
    # create dictionaries
    train_f1 = {1:[], 2:[], 3:[], 4:[], 5:[]}
    val_f1 = {1:[], 2:[], 3:[], 4:[], 5:[]}
    
    for i in range(len(folds)):
        
        # print('Fold', i+1)
        # define the training set (i.e. selecting all folds and deleting the one used for validation)
        train_set = np.delete(np.asarray(folds).reshape(len(folds), folds[0].shape[0], folds[0].shape[1]), i, axis=0)
        train_folds = train_set.reshape(len(train_set)*train_set[0].shape[0], train_set[0].shape[1])
        X_train = train_folds[:,:-1]
        y_train = train_folds[:, -1]
        
        # define the validation set
        val_fold = folds[i]
        X_val = val_fold[:,:-1]
        y_val = val_fold[:, -1]
        
        # convert the 0s to -1s so the labels are 1 and -1
        y_train_svm = y_train.copy()
        y_train_svm[y_train_svm == 0] = -1
        y_val_svm = y_val.copy()
        y_val_svm[y_val_svm == 0] = -1
        
        # train the svm model
        W_rbf = sgd(X_train, Y_train, max_iterations=500,
                   stop_criterion=0.01, learning_rate=1e-8, regul_strength=1e5,
                   print_outcome=False, rbf_kernel=True, sigma=sigma_param, b=-0.1)
        
        # obtain the accuracies and predictions and store in the appropriate variables
        accuracy_train_rbf, y_preds_train_rbf = score(W_rbf, X_train, y_train_svm, X_train,
                                              rbf_kernel=True, sigma=sigma_param, return_preds = True, b=-0.1)
        
        accuracy_val_rbf, y_preds_val_rbf = score(W_rbf, X_val, y_val_svm, X_train,
                                            rbf_kernel=True, sigma=sigma_param, return_preds = True, b=-0.1)
        
        
        # Stack the actual and the prediction vectors into one matrix
        training_y_comparison_vec = np.vstack((y_train_svm, y_preds_train_rbf)).T
        val_y_comparison__vec = np.vstack((y_val_svm, y_preds_val_rbf)).T
        
        # Calculate the F1 Scores
        f1score_train = calculate_f1_score(training_y_comparison_vec)
        f1score_val = calculate_f1_score(val_y_comparison__vec)

        # Add them to the appropriate dictionaries
        train_f1[i+1].append(f1score_train)
        val_f1[i+1].append(f1score_val)
        
    return train_f1, val_f1

def calculate_f1_score(v, get_ROC_data = False):
    TP = sum(1 for val in (v == [1,1]).sum(axis=1) if val==2)
    FN = sum(1 for val in (v == [1,-1]).sum(axis=1) if val==2)
    TN = sum(1 for val in (v == [-1,-1]).sum(axis=1) if val==2)
    FP = sum(1 for val in (v == [-1,1]).sum(axis=1) if val==2)
    
    #print(FN)
    
    if get_ROC_data:
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        return TPR, FPR
    
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
    
        f1score = 2*(precision * recall) / (precision+recall)
    return f1score

# vector to scan the sigmas
sigma_vec = np.arange(3,21)/10


# Create matrices to store f1 scores
train_f1_mat = np.zeros(len(sigma_vec))
val_f1_mat = np.zeros(len(sigma_vec))


# Grid Search over max_features and max_depth keeping N_trees fixed at 5
for i, sigma_param in enumerate(sigma_vec):
    if (i+1)%2 == 0:
        print("Loop: " + str(i+1)+ "/" + str(len(sigma_vec)))
    train_f1, val_f1 = cross_val_evaluate_svm(folds, sigma_param)
    print(val_f1)
    mean_train_f1 = np.mean([train_f1[k][0] for k in range(1,6)])
    mean_val_f1 = np.mean([val_f1[k][0] for k in range(1,6)])
        
    train_f1_mat[i] = mean_train_f1
    val_f1_mat[i] = mean_val_f1
    
optimal_index = np.where(val_f1_mat == val_f1_mat.max())
optimal_sigma = sigma_vec[optimal_index[0][0]]

# Retrain the model using the optimal sigma
b_param = -0.1

W_rbf = sgd(X_train, Y_train, max_iterations=500,
                   stop_criterion=0.01, learning_rate=1e-8, regul_strength=1e5,
                   print_outcome=True, rbf_kernel=True, sigma=optimal_sigma, b=b_param)
print("Training finished.")

print(W_rbf)

accuracy_train_rbf, y_preds_train_rbf = score(W_rbf, X_train, Y_train_svm, X_train,
                                              rbf_kernel=True, sigma=optimal_sigma, return_preds = True, b=b_param)
accuracy_test_rbf, y_preds_test_rbf = score(W_rbf, X_test, Y_test_svm, X_train,
                                            rbf_kernel=True, sigma=optimal_sigma, return_preds = True, b=b_param)

print("RBF Accuracy on train set: {}".format(accuracy_train_rbf))
print("RBF Accuracy on test set: {}".format(accuracy_test_rbf))

print(np.unique(y_preds_train_rbf))
print(np.unique(y_preds_test_rbf))

# Compare Accuracies
svm_accuracies = np.array([[accuracy_train, accuracy_test],[accuracy_train_rbf, accuracy_test_rbf]])
svm_accuracies_df = pd.DataFrame(svm_accuracies, columns = ["Training Accuracy", "Test Accuracy"], index = ["Standard Linear", "RBF kernel"])

"""
2.3.2
"""
# use the same sigma_vec as above

# Create matrices to store the TPR and the TNR
TPR_mat = np.zeros(len(sigma_vec))
FPR_mat = np.zeros(len(sigma_vec))

for i, sigma_param in enumerate(sigma_vec):
    if (i+1)%2 == 0:
        print("Loop: " + str(i+1)+ "/" + str(len(sigma_vec)))
        
    # train the svm model
    W_rbf = sgd(X_test, Y_test_svm, max_iterations=500,
               stop_criterion=0.01, learning_rate=1e-8, regul_strength=1e5,
               print_outcome=False, rbf_kernel=True, sigma=sigma_param, b=-0.1)
    
    accuracy_test_rbf, y_preds_test_rbf = score(W_rbf, X_test, Y_test_svm, X_train,
                                        rbf_kernel=True, sigma=sigma_param, return_preds = True, b=-0.1)
    
    # Stack the actual and the prediction vectors into one matrix
    test_y_comparison__vec = np.vstack((Y_test_svm, y_preds_test_rbf)).T
    
    # Obtain the TPR and FPR
    TPR, FPR = calculate_f1_score(test_y_comparison__vec, get_ROC_data = True)
    
    TPR_mat[i] = TPR
    FPR_mat[i] = FPR
    

plt.plot(FPR_mat, TPR_mat)
plt.plot((min(FPR_mat),max(TPR_mat)),(min(FPR_mat),max(TPR_mat)),color='red',linewidth=2,linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve")
plt.plot(FPR_mat[optimal_index], TPR_mat[optimal_index],'bo') 
arrow_properties = dict(
    facecolor="black", width=0.5,
    headwidth=4, shrink=0.1)
plt.annotate(
    "Optimal Kernel SVM", xy=(FPR_mat[optimal_index], TPR_mat[optimal_index]),
    xytext=(0.35, 0.1),
    arrowprops=arrow_properties)
plt.show()