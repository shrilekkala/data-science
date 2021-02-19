# import packages
import pandas as pd
import numpy as np
from collections import Counter


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

# Aggregate the X and Y data into one array to be used for cross validation
train = np.hstack((X_train, Y_train[:, np.newaxis]))
test = np.hstack((X_test, Y_test[:, np.newaxis]))

def cross_val_split(data, num_folds):
  fold_size = int(len(data) / num_folds)
  data_perm = np.random.permutation(data)
  folds = []
  for k in range(num_folds):
    folds.append(data_perm[k*fold_size:(k+1)*fold_size, :])

  return folds

# Generate the folds
folds = cross_val_split(train, 5)


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

def split_dataset2(X, y, column, value, sample_weights=None):
  region1 = []
  region2 = []
  featVec = X[:, column]  
  # We are removing the feature column from dataset (split), as before
  X = X[:,[i for i in range(X.shape[1]) if i!=column]]  
  for i in range(len(featVec)):
      if featVec[i] <= value:
          region1.append(i)
      else:
          region2.append(i)     
  X1 = X[region1,:]
  y1 = y[region1]
  X2 = X[region2,:]
  y2 = y[region2]
  # returns list of splits, implemented as tuples. A few ways to do this.
  if (sample_weights is None):
      return [(X1, y1), (X2, y2)]
  else:
      return [(X1, y1, sample_weights[region1]) , (X2, y2, sample_weights[region2])]

def cross_entropy_calculate(X, y, column, value, sample_weights=None):
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
    
    new_cost = 0.0
    #split the values of i-th feature and calculate the cost of the split
  
    for sub_X, sub_y, sub_sample_weights in split_dataset2(X, y, column, value, sample_weights):
        prob = np.sum(sub_sample_weights) / float(np.sum(sample_weights))
        # New cost (cross entropy multiplied by a weighted prob depending on the sample weights)
        new_cost += prob * cross_entropy(sub_y, sub_sample_weights)
  
    information_gain = old_cost - new_cost # information gain

    return information_gain


def choose_best_feature2(X, y, max_features, sample_weights=None):
      if sample_weights is None:
          sample_weights = np.ones(y.shape[0]) / y.shape[0]
      n_features = X.shape[1]
      
      if n_features > max_features:
          n_features = max_features
      best_split=None
      best_gain_cost = 0.0
      
      indices = np.arange(X.shape[1])
      np.random.shuffle(indices)
      
      
      for i in indices[:max_features]:
          unique_vals = np.unique(X[:, i])
          for v in unique_vals:
              info_gain_cost = cross_entropy_calculate(X, y, i, v, sample_weights)
              # print(i,v,info_gain_cost)
              if info_gain_cost > best_gain_cost:
                  best_gain_cost = info_gain_cost
                  best_split = (i,v)     

      return best_split



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


# Finally, we can build the decision tree by using choose_best_feature to find the best feature to split the X, 
# and spli _dataset to get sub-trees.
def build_tree2(X, y, feature_names, max_features, max_depth, current_depth, sample_weights=None, min_samples_leaf=2):
    """Build the decision tree according to the data.
    Args:
        X: (np.array) training features, of shape (N, D).
        y: (np.array) vector of training labels, of shape (N,).
        feature_names (list): record the name of features in X in the original dataset.
        depth (int): current depth for this node.
        sample_weights: weights for each samples, of shape (N,).
    Returns:
        (dict): a dict denoting the decision tree. 
        <tree> ::= node:'leaf' label:<iris-categ>
                |  node:'split' feature-name:<col-name> value:<num> leaf:<tree> right:<tree>  
    """
    mytree = dict()
    
    # include a clause for the cases where (i) no feature, (ii) all lables are the same, 
    # (iii) depth exceed, or (iv) X is too small, or (v) X consists of exactly the same features
    if len(feature_names)==0 or len(np.unique(y))==1 or current_depth>=max_depth or len(X)<=min_samples_leaf or len(np.unique(X, axis = 0)) == 1: 
        mytree = { 'node':'leaf' ,  'label': majority_vote(y, sample_weights) }
        return mytree
    
    placeholder = choose_best_feature2(X, y, max_features, sample_weights)    
    
    if placeholder == None:
        mytree = { 'node':'leaf' ,  'label': majority_vote(y, sample_weights) }
        return mytree
    else:  
        if placeholder == None:
            print("ERROR - start")
            print(X)
            print(y)
            print("ERROR - end")
        best_feature_idx, value = placeholder
        best_feature_name = feature_names[best_feature_idx]
        feature_names = feature_names[:]
        # feature_names.remove(best_feature_name)
        splits = split_dataset2(X, y, best_feature_idx, value, sample_weights)
        mytree = { 'node':'split', 'feature_name':best_feature_name, 'value':value }
        # split[i] = (subX, sub_y, sub_sample_weight)
        mytree['left'] = build_tree2(splits[0][0], splits[0][1], feature_names, max_features, max_depth, current_depth+1, splits[0][2]) 
        mytree['right'] = build_tree2(splits[1][0], splits[1][1], feature_names, max_features, max_depth, current_depth+1, splits[1][2]) 
          
        return mytree

# wrapper function to call the build_tree function
def train_decision_tree(X, y, max_features, max_depth, sample_weights=None):
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
    tree = build_tree2(X, y, feature_names, max_features, max_depth, current_depth=1, sample_weights=sample_weights)
    return tree

# fit the decision tree with training data
## tree = train_decision_tree(df_X_train, df_Y_train, max_features, max_depth)

# use this fitted decision tree to make predictions for our test set X_test
def classify2(tree, x):
  """
  Classify a single sample with the fitted decision tree.
  Args:
      x: ((pd.Dataframe) a single sample features, of shape (D,).
  Returns:
      (int): predicted testing sample label.
  """
  if tree['node'] == 'leaf':
    return tree['label']
  else:
    feature_name = tree['feature_name']
    v = x.loc[feature_name]
    if (v <= tree['value']):
        return classify2(tree['left'],x)
    else:
        return classify2(tree['right'],x)

def predict(tree, X):
    """
    Predict classification results for X.
    Args:
        X: (pd.Dataframe) testing sample features, of shape (N, D).
    Returns:
        (np.array): predicted testing sample labels, of shape (N,).
    """
    if len(X.shape)==1:
        return classify2(tree, X)
    else:
        results=[]
        for i in range(X.shape[0]):
            results.append(classify2(tree, X.iloc[i, :]))
        return np.array(results)

def score(X_test, y_test):
  y_pred = predict(X_test)
  return np.float(sum(y_pred==y_test)) / float(len(y_test))

## print('Training accuracy:', score(df_X_train, df_Y_train))
## print('Test accuracy:', score(df_X_test, df_Y_test))

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
def randforest(df_X_y, N_trees, max_features, max_depth, sample_weights=None):
    
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

        boot_tree[i] = train_decision_tree(boot_df_X, boot_df_y, max_features, max_depth)
    
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
        tree_preds.append(classify2(boot_tree[i], x))
    
    # assign the modal value as the label (if it exists)
    
    # find frequency of each value
    freq_counter = Counter(tree_preds)
    freq_dict = dict(freq_counter)
    
    max_freq = max(list(freq_counter.values()))
    modal_values = [num for num, freq in freq_dict.items() if freq == max_freq]

    # in case there are multiple modal values, randomly choose from the list
    label = modal_values[np.random.randint(0, len(modal_values))]
    
    return label
    

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

# fit the random forest with training data
## boot_tree = randforest(train_data, 5, 5, max_depth, sample_weights=None)

## print('Training accuracy:', score_random_forest(boot_tree, df_X_train, df_Y_train))
## print('Test accuracy:', score_random_forest(boot_tree, df_X_test, df_Y_test))

def cross_val_evaluate_random_forest(folds, N_trees, max_features, max_depth):
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
        boot_tree = randforest(df_train, N_trees, max_features, max_depth, sample_weights=None)
        
        # obtain the accuracies and store in the appropriate dictionaries
        train_accuracy = score_random_forest(boot_tree, df_X_train, y_train)
        val_accuracy = score_random_forest(boot_tree, df_X_val, y_val)
        
        train_acc[i+1].append(train_accuracy)
        val_acc[i+1].append(val_accuracy)
        
    return train_acc, val_acc
    
a = 5
b = 4
c = 10
train_accuracy, val_accuracy = cross_val_evaluate_random_forest(folds, a, b, c)
print(train_accuracy)
print(val_accuracy)

mean_train_accuracy = np.mean([train_accuracy[i][0] for i in range(1,6)])
mean_val_accuracy = np.mean([val_accuracy[i][0] for i in range(1,6)])

print(mean_train_accuracy)
print(mean_val_accuracy)

