# import packages
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from collections import Counter

# load data
data = load_iris()
# print data to see how it is structured
# print(data)
X, y, column_names = data['data'], data['target'], data['feature_names']
# combining all information in one data frame
X_y = pd.DataFrame(X, columns=column_names)
X_y['label'] = y

# stacking data X and labels y into one matrix
X_y_shuff = X_y.iloc[np.random.permutation(len(X_y))]

# we split train to test as 70:30
split_rate = 0.7
train, test = np.split(X_y_shuff, [int(split_rate*(X_y_shuff.shape[0]))])

X_train = train[train.columns[:-1]]
y_train = train[train.columns[-1]]

X_test = test[test.columns[:-1]]
y_test = test[test.columns[-1]]

y_train = y_train.astype(int)
y_test = y_test.astype(int)

def gini_impurity(y, sample_weights=None):
    """ 
    Calculate the gini impurity for labels.
    Arguments:
        y: vector of training labels, of shape (N,).
        sample_weights: weights for each samples, of shape (N,).
    Returns:
        (float): the gini impurity for y.
    """
    if sample_weights is None:
        sample_weights = np.ones(y.shape[0]) / y.shape[0]
    
    gini = 1
    num = y.shape[0]  # number of labels
    label_counts = {}  # caculate different labels in y，and store in label_counts
    for i in range(num):
        if y[i] not in label_counts.keys():
            label_counts[y[i]] = 0
        label_counts[y[i]] += sample_weights[i]
    
    for key in label_counts:
        prob = float(label_counts[key]) / float(np.sum(sample_weights))
        gini -= prob**2 ## <-- EDIT THIS LINE - DONE
    
    return gini

# NB I am just going to use entropy for now
"""
ENTROPY
"""
def cross_entropy(y, sample_weights=None):
    """ 
    Calculate the cross_entropy for labels.
    Arguments:
        y: vector of training labels, of shape (N,).
        sample_weights: weights for each samples, of shape (N,).
    Returns:
        (float): the gini impurity for y.
    """
    if sample_weights is None:
        sample_weights = np.ones(y.shape[0]) / y.shape[0]
    
    ce = 0
    num = y.shape[0]  # number of labels
    label_counts = {}  # caculate different labels in y，and store in label_counts
    for i in range(num):
        if y[i] not in label_counts.keys():
            label_counts[y[i]] = 0
        label_counts[y[i]] += sample_weights[i]
    
    for key in label_counts:
        prob = float(label_counts[key]) / float(np.sum(sample_weights))
        # NB -  instead of + as in the notes
        ce -= prob * np.log(prob)
    
    return ce

cross_entropy(y_train.to_numpy())
gini_impurity(y_train.to_numpy())

def split_dataset(X, y, column, value, sample_weights=None):
    """
    Return the split of data whose column-th feature equals value.
    Arguments:
        X: training features, of shape (N, D).
        y: vector of training labels, of shape (N,).
        column: the column of the feature for splitting.
        value: the value of the column-th feature for splitting.
        sample_weights: weights for each samples, of shape (N,).
    Returns:
        (np.array): the subset of X whose column-th feature equals value.
        (np.array): the subset of y whose column-th feature equals value.
        (np.array): the subset of sample weights whose column-th feature equals value.
    """ 
    ret = []
    featVec = X[:, column]
    # selecting all columns of X except the "column" we are splitting on
    X = X[:,[i for i in range(X.shape[1]) if i!=column]]
    
    for i in range(len(featVec)):
        if featVec[i]==value:
            ret.append(i)
    
    sub_X = X[ret,:]
    sub_y = y[ret]
    sub_sample_weights = sample_weights[ret]
    
    return sub_X, sub_y, sub_sample_weights

def cross_entropy_calculate(X, y, column, sample_weights=None):
  """
  Calculate the resulted cross entropy given a vector of features.
  Arguments:
      X: training features, of shape (N, D).
      y: vector of training labels, of shape (N,).
      column: the column of the feature for calculating. 0 <= column < D
      sample_weights: weights for each samples, of shape (N,).
  Returns:
      (float): the resulted gini impurity after splitting by this feature.
  """
  if sample_weights is None:
      sample_weights = np.ones(y.shape[0]) / y.shape[0]
  
  information_gain = 0
  old_cost = cross_entropy(y, sample_weights)
  
  unique_vals = np.unique(X[:, column])
  new_cost = 0.0
  #split the values of i-th feature and calculate the cost 
  for value in unique_vals:
      sub_X, sub_y, sub_sample_weights = split_dataset(X, y, column, value, sample_weights=sample_weights) 
      prob = np.sum(sub_sample_weights) / float(np.sum(sample_weights))
      # New cost (cross entropy multiplied by a weighted prob depending on the sample weights)
      new_cost += prob * cross_entropy(sub_y, sub_sample_weights)
  
  information_gain = old_cost - new_cost # information gain

  return information_gain

# evaluate for feature sepal width (cm)
cross_entropy_calculate(X_train.to_numpy(), y_train.to_numpy(), 3)

def choose_best_feature(X, y, sample_weights=None):
    """
    Choose the best feature to split according to criterion.
    Args:
        X: training features, of shape (N, D).
        y: vector of training labels, of shape (N,).
        sample_weights: weights for each samples, of shape (N,).
    Returns:
        (int): the column for the best feature
    """
    if sample_weights is None:
        sample_weights = np.ones(y.shape[0]) / y.shape[0]
    
    best_feature_idx = 0
    n_features = X.shape[1]    
    
    # use C4.5 algorithm
    best_gain_cost = 0.0
    for i in range(n_features):
        info_gain_cost = cross_entropy_calculate(X, y, i, sample_weights=sample_weights)
        if info_gain_cost > best_gain_cost:
            best_gain_cost = info_gain_cost
            best_feature_idx = i                
    
    return best_feature_idx

choose_best_feature(X_train.to_numpy(), y_train.to_numpy())

def majority_vote(y, sample_weights=None):
  """
  Return the label which appears the most in y.
  Args:
      y: vector of training labels, of shape (N,).
      sample_weights: weights for each samples, of shape (N,).
  Returns:
      (int): the majority label
  """
  if sample_weights is None:
      sample_weights = np.ones(y.shape[0]) / y.shape[0]
  
  majority_label = y[0]

  dict_num = {}
  for i in range(y.shape[0]):
      if y[i] not in dict_num.keys():
          dict_num[y[i]] = sample_weights[i]
      else:
          dict_num[y[i]] += sample_weights[i]
  
  majority_label = max(dict_num, key=dict_num.get)
  # end answer
  return majority_label

majority_vote(y_train.to_numpy())

# Finally, we can build the decision tree by using choose_best_feature to find the best feature to split the X, 
# and split_dataset to get sub-trees.
def build_tree(X, y, feature_names, depth, sample_weights=None, max_depth=10, min_samples_leaf=2):
  """Build the decision tree according to the data.
  Args:
      X: (np.array) training features, of shape (N, D).
      y: (np.array) vector of training labels, of shape (N,).
      feature_names (list): record the name of features in X in the original dataset.
      depth (int): current depth for this node.
      sample_weights: weights for each samples, of shape (N,).
  Returns:
      (dict): a dict denoting the decision tree. 
      Example:
          The first best feature name is 'title', and it has 5 different values: 0,1,2,3,4. 
          For 'title' == 4, the next best feature name is 'pclass', we continue split the remain data. 
          If it comes to the leaf, we use the majority_label by calling majority_vote.
          
          mytree = {
              'title': {
                  0: subtree0,
                  1: subtree1,
                  2: subtree2,
                  3: subtree3,
                  4: {
                      'pclass': {
                          1: majority_vote([1, 1, 1, 1]) # which is 1, majority_label
                          2: majority_vote([1, 0, 1, 1]) # which is 1
                          3: majority_vote([0, 0, 0]) # which is 0
                      }
                  }
              }
          }
  """
  mytree = dict()

  # include a clause for the cases where (i) no feature, (ii) all lables are the same, 
  # (iii) depth exceed, or (iv) X is too small
  if len(feature_names)==0 or len(np.unique(y))==1 or depth>=max_depth or len(X)<=min_samples_leaf: 
      return majority_vote(y, sample_weights)
  
  else:
    best_feature_idx = choose_best_feature(X, y, sample_weights=sample_weights)
    best_feature_name = feature_names[best_feature_idx]
    feature_names = feature_names[:]
    feature_names.remove(best_feature_name)
    
    mytree = {best_feature_name:{}}
    unique_vals = np.unique(X[:, best_feature_idx])
    for value in unique_vals:
        sub_X, sub_y, sub_sample_weights = split_dataset(X, y, best_feature_idx, value, sample_weights=sample_weights)  
        mytree[best_feature_name][value] = build_tree(sub_X, sub_y, feature_names, depth+1, sample_weights=sub_sample_weights) 

    return mytree

# wrapper function to call the build_tree function
def train_decision_tree(X, y, sample_weights=None):
    """
    Build the decision tree according to the training data.
    Args:
        X: (pd.Dataframe) training features, of shape (N, D). Each X[i] is a training sample.
        y: (pd.Series) vector of training labels, of shape (N,). y[i] is the label for X[i], and each y[i] is
        an integer in the range 0 <= y[i] <= C. Here C = 1.
        sample_weights: weights for each samples, of shape (N,).
    """
    if sample_weights is None:
        # if the sample weights is not provided, we assume the samples have uniform weights
        sample_weights = np.ones(X.shape[0]) / X.shape[0]
    else:
        sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    
    feature_names = X.columns.tolist()
    X = np.array(X)
    y = np.array(y)
    tree = build_tree(X, y, feature_names, depth=1, sample_weights=sample_weights)
    return tree

# fit the decision tree with training data
tree = train_decision_tree(X_train, y_train)

# use this fitted decision tree to make predictions for our test set X_test
def classify(tree, x):
    """
    Classify a single sample with the fitted decision tree.
    Args:
        x: ((pd.Dataframe) a single sample features, of shape (D,).
    Returns:
        (int): predicted testing sample label.
    """
    feature_name = list(tree.keys())[0] # first element
    second_dict = tree[feature_name]            
    key = x.loc[feature_name]
    if key not in second_dict:
        key = np.random.choice(list(second_dict.keys()))
    value_of_key = second_dict[key]
    # if value_of_key is a dictionary, recursively call the classify function again
    if isinstance(value_of_key, dict):
        label = classify(value_of_key, x)
    # if not, return the value as the label
    else:
        label = value_of_key
    return label

def predict(X):
    """
    Predict classification results for X.
    Args:
        X: (pd.Dataframe) testing sample features, of shape (N, D).
    Returns:
        (np.array): predicted testing sample labels, of shape (N,).
    """
    if len(X.shape)==1:
        return classify(tree, X)
    else:
        results=[]
        for i in range(X.shape[0]):
            results.append(classify(tree, X.iloc[i, :]))
        return np.array(results)

def score(X_test, y_test):
  y_pred = predict(X_test)
  return np.float(sum(y_pred==y_test)) / float(len(y_test))

print('Training accuracy:', score(X_train, y_train))
print('Test accuracy:', score(X_test, y_test))

# Create a bootstrapped dataset given a data frame
def bootstrap(df_data, N_trees):
    # dictionary to store each sample
    boot_data = dict()
    
    # random sampling with replacement
    for i in range(N_trees):
        boot_data[i] = df_data.sample(n = len(X_train), replace = True)
        
    return boot_data

# boot_X_train = bootstrap(df_X_train, 5)


# function that creates a random forest via bagging
# NB here I am using Hard Voting (i.e. taking the modal value instead of mean probability)
def randforest(df_X_y, N_trees, sample_weights=None):
    
    # bootstrap the data and create samples
    boot_data = bootstrap(df_X_y, N_trees)
        
    # create a dictionary to store each tree
    boot_tree = {}
    
    # for each sample of the data create a decision tree
    for i in range(N_trees):
        # split the aggregated X and y into separate arrays again
        boot_df_X_y = boot_data[i]
        boot_df_X = boot_df_X_y[boot_df_X_y.columns[:-1]]
        boot_df_y = boot_df_X_y[boot_df_X_y.columns[-1]]

        boot_tree[i] = train_decision_tree(boot_df_X, boot_df_y)
    
    return boot_tree
    
def classify_random_forest(boot_tree, x):
    """
    Classify a single sample with the fitted decision tree.
    Args:
        x: ((pd.Dataframe) a single sample features, of shape (D,).
    Returns:
        (int): predicted testing sample label.
    """
    # create a list of values predicted by the tree
    tree_preds = []
    
    for i in range(len(boot_tree)):
        tree_preds.append(classify(boot_tree[i], x))
    
    # assign the modal value as the label (if it exists)
    
    # find frequency of each value
    freq_counter = Counter(tree_preds)
    freq_dict = dict(freq_counter)
    
    max_freq = max(list(freq_counter.values()))
    modal_values = [num for num, freq in freq_dict.items() if freq == max_freq]

    # in case there are multiple modal values, randomly choose from the list
    label = modal_values[np.random.randint(0, len(modal_values))]
    
    return label
    
# fit the random forest with training data
boot_tree = randforest(train, 5, sample_weights=None)

def predict_random_forest(boot_tree, X):
    """
    Predict classification results for X.
    Args:
        X: (pd.Dataframe) testing sample features, of shape (N, D).
    Returns:
        (np.array): predicted testing sample labels, of shape (N,).
    """
    if len(X.shape)==1:
        return classify_random_forest(boot_tree, X)
    else:
        results=[]
        for i in range(X.shape[0]):
            results.append(classify_random_forest(boot_tree, X.iloc[i, :]))
        return np.array(results)
    
def score_random_forest(boot_tree, X_test, y_test):
    y_pred = predict_random_forest(boot_tree, X_test)
    return np.float(sum(y_pred==y_test)) / float(len(y_test))

print('Training accuracy:', score_random_forest(boot_tree, X_train, y_train))
print('Test accuracy:', score_random_forest(boot_tree, X_test, y_test))

def cross_val_evaluate_random_forest(folds):
    # create dictionaries
    train_acc = {1:[], 2:[], 3:[], 4:[], 5:[]}
    val_acc = {1:[], 2:[], 3:[], 4:[], 5:[]}
    
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
    
        # convert whole training set, X_train and X_val into panda data frames as required
        df_train = pd.DataFrame(train_folds)
        df_X_train = pd.DataFrame(X_train)
        df_X_val = pd.DataFrame(X_val)
        
        # train the random forest and obtain the trees
        boot_tree = randforest(df_train, 5, sample_weights=None)
        
        # obtain the accuracies and store in the appropriate dictionaries
        train_accuracy = score_random_forest(boot_tree, df_X_train, y_train)
        val_accuracy = score_random_forest(boot_tree, df_X_val, y_val)
        
        train_acc[i+1].append(train_accuracy)
        val_acc[i+1].append(val_accuracy)
        
    return train_acc, val_acc
    
train_accuracy, val_accuracy = cross_val_evaluate_random_forest(folds)
