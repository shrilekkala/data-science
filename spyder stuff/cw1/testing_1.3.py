import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# import training set
test_data = pd.read_csv('./regression_test.csv', header=None)
train_data = pd.read_csv('./regression_train.csv', header=None)

# NB our data is already standardised
full_data = pd.concat([test_data, train_data])
full_X = np.array(full_data.iloc[:,:-1])
np.mean(full_X, 0)
np.std(full_X, 0)

#print(test_data.tail())
#print(train_data.tail())

# (NB column of ones for the intercept is already in the data)

X_train = np.array(train_data.iloc[:,:-1])
Y_train = np.array(train_data.iloc()[:,-1])

X_test = np.array(test_data.iloc[:,:-1])
Y_test = np.array(test_data.iloc()[:,-1])




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

# Generate the folds
folds = cross_val_split(train, 5)

"""""""""""""""""""""""""""""""""""""""""
kNN 
"""""""""""""""""""""""""""""""""""""""""
# distrance function for kNN
def euclidian_distance(p, q):
    return np.sqrt(np.sum((q-p)**2, axis = 1))

# visual exploration - make better
fig4, ax4 = plt.subplots(nrows=6, ncols=3, figsize=(20,15))
for i, ax in enumerate(ax4.flatten()):
    ax.set_xlabel(r"Predictor X_" + str(i+1), size=10, axes = ax)
    ax.set_ylabel(r"Median value of home", size=10, axes = ax)
    ax.scatter(X_train[:,i], Y_train, c='blue', alpha=0.25)
    ax.scatter(X_test[:,i], Y_test, c='red', alpha=0.25)
   
plt.show()

#NB we notice above that X_2 (i=3) and X_3 (i=4) appear to be very close
# Also i=2 and i = 16
np.linalg.norm(X_test[:,3] - X_test[:,4])
np.linalg.norm(X_test[:,2] - X_test[:,16])
# so we could remove one of them as they are redundant
# or do that 19x19 plot thing

def k_neighbours(X_train, X_test, k=5, return_distance=False):
    dist = []
    neigh_ind = []
    
    # compute distance from each point x_text in X_test to all points in X_train 
    point_dist =  [euclidian_distance(x_test, X_train) for x_test in X_test]
    
    # determine which k training points are closest to each test point
    for row in point_dist:
        enum_neigh = enumerate(row)
        sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:k]
    
        ind_list = [tup[0] for tup in sorted_neigh]
        dist_list = [tup[1] for tup in sorted_neigh]
    
        dist.append(dist_list)
        neigh_ind.append(ind_list)
    
    # return distances together with indices of k nearest neighbouts
    if return_distance:
        return np.array(dist), np.array(neigh_ind)
    
    return np.array(neigh_ind)

def reg_predict(X_train, Y_train, X_test, k):
    # each of the k neighbours contributes equally to the classification of any data point in X_test  
    neighbours = k_neighbours(X_train, X_test, k=k)
    # compute mean over neighbours labels 
    Y_pred = np.array([np.mean(Y_train[neighbour]) for neighbour in neighbours])
    return Y_pred

k = 5
Y_pred_knn = reg_predict(X_train, Y_train, X_test, k)

# visual exploration - make better
fig5, ax5 = plt.subplots(nrows=6, ncols=3, figsize=(20,15))
for i, ax in enumerate(ax5.flatten()):
    ax.set_xlabel(r"Predictor X_" + str(i+1), size=10, axes = ax)
    ax.set_ylabel(r"Median value of home", size=10, axes = ax)
    ax.scatter(X_test[:,i], Y_test, c='red', alpha=0.25)
    ax.scatter(X_test[:,i], Y_pred_knn, c='yellow', alpha=0.25)
   
plt.show()

def r2_score(y_test, y_pred):
    numerator = np.sum((y_test - y_pred)**2)
    y_avg = np.mean(y_test)
    denominator = np.sum((y_test - y_avg)**2)
    return 1 - numerator/denominator


# in sample MSE
train_preds_kNN = reg_predict(X_train, Y_train, X_train, k=k)
MSE_train_kNN = np.mean((Y_train - train_preds_kNN) ** 2)

# out of sample MSE
test_preds_kNN = reg_predict(X_train, Y_train, X_test, k=k)
MSE_test_kNN = np.mean((Y_test - test_preds_kNN) ** 2)

print("kNN - In sample error    : " + str(MSE_train_kNN))
print("kNN - Out of sample error: " + str(MSE_test_kNN))

print('Train set mean accuracy:', r2_score(Y_train, train_preds_kNN))
print('Test set mean accuracy:', r2_score(Y_test, test_preds_kNN))



def cross_val_evaluate_kNN(folds, k_vec):
    
    # create dictionaries
    train_MSE = {1:[], 2:[], 3:[], 4:[], 5:[]}
    val_MSE = {1:[], 2:[], 3:[], 4:[], 5:[]}
    val_residuals = {1:[], 2:[], 3:[], 4:[], 5:[]}

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
        for k in k_vec:
            # print(k)
            
            train_preds_kNN = reg_predict(X_train, y_train, X_train, k)
            val_preds_kNN = reg_predict(X_train, y_train, X_val, k)
            
            # evaluate
            # training data MSE
            MSE_train_kNN = np.mean((y_train - train_preds_kNN) ** 2)
            
            # validation data MSE
            MSE_val_kNN = np.mean((y_val - val_preds_kNN) ** 2)
            
            # store these in the appropriate dictionaries
            train_MSE[i+1].append(MSE_train_kNN)
            val_MSE[i+1].append(MSE_val_kNN)
            val_residuals[i+1].append(y_val - val_preds_kNN)
    
   
    print("Training finished.")
    return train_MSE, val_MSE

k_vec = np.arange(100)+1

train_MSE_kNN, val_MSE_kNN = cross_val_evaluate_kNN(folds, k_vec)

"""
Consider fold 1, scan k parameter
Note to self, choose one that looks nice by setting random seed
"""

for i in range(1,6):
    plt.title("Plot of MSE errors for over different k values for kNN [Fold " + str(i) + "]")
    plt.plot(k_vec, train_MSE_kNN[i], label = "Training Errors")
    plt.plot(k_vec, val_MSE_kNN[i], label = "Validation Errors")
    plt.legend()
    plt.grid()
    plt.xlabel("k")
    plt.ylabel("MSE")
    plt.show()
    # print("Optimal λ for Fold 1 is " + str(lambda_vec[np.argmin(val_MSE[1])]))


# The optimal lambdas for for each fold obtained using argmin
for i in range(0,5):
    print("Optimal k for Fold " + str(i+1) + " is " + str(k_vec[np.argmin(val_MSE_kNN[i+1])]))
    
# Compute the average validation MSE over the folds, to get average for each penalty term
average_val_MSE_kNN = np.mean([val_MSE_kNN[fold] for fold in range(1, 6)], axis = 0)
optimal_k = k_vec[np.argmin(average_val_MSE_kNN)]

# retrain the model over the whole data set using the optimal k

# in sample MSE
train_preds_kNN = reg_predict(X_train, Y_train, X_train, optimal_k)
MSE_train_kNN = np.mean((Y_train - train_preds_kNN) ** 2)

# out of sample MSE
test_preds_kNN = reg_predict(X_train, Y_train, X_test, optimal_k)
MSE_test_kNN = np.mean((Y_test - test_preds_kNN) ** 2)

print("Cross-val kNN - In sample error    : " + str(MSE_train_kNN))
print("Cross-val kNN - Out of sample error: " + str(MSE_test_kNN))

print('Cross-val Train set mean accuracy:', r2_score(Y_train, train_preds_kNN))
print('Cross-val Test set mean accuracy:', r2_score(Y_test, test_preds_kNN))


"""
Distribution of errors fold 1
"""