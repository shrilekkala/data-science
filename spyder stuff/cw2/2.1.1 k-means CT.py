# imports
import numpy as np
import collections
import matplotlib.pyplot as plt
import csv

np.random.seed(1024)

# Load the matrices
# Feature Matrix
F = np.loadtxt(open("data/feature_matrix_karate_club.csv", "rb"), delimiter=",", skiprows=1)
individuals = F[:, 0]
feature_matrix = F[:, 1:]

A = np.loadtxt(open("data/karate_club_graph.csv", "rb"), delimiter=",", skiprows=1)
adjacency_matrix = A[:, 1:]

S = list(csv.reader(open("data/ground_truth_karate_club.csv")))
true_split = S[1:]
# Convert to numpy array with "Mr Hi" being 0 and "Officer" being 1
true_split_np = np.vstack((np.arange(34), np.zeros(34))).T
for i in range(34):
    if true_split[i][1] == "Officer":
        true_split_np[i, 1] = 1


# visualising the clusters
plt.figure(figsize=(10,8))
plt.scatter(feature_matrix[:,0], feature_matrix[:,1])
plt.show()

"""
2.1.1
"""

# parameters
k = 5
max_iter = 15

# Distance matrix
difference = feature_matrix[:, :, None] - feature_matrix[:, :, None].T
D = np.sqrt((difference**2).sum(1))

def no_empty_labels(k, new_labels):
    # Find which clusters are empty
    empty_labels = np.setdiff1d(np.arange(k),new_labels)
    
    # Randomly assign a new element from a random cluster (with size > 1) to this label
    for empty_lab in empty_labels:
        
        # Find clusters with greater than one element in them
        c = collections.Counter(new_labels)
        multiple_element_labels = [j[0] for j in c.items() if j[1] > 1]
        
        # randomly choose cluster
        rand_lab = np.random.choice(multiple_element_labels)
        
        # randomly choose element in that cluster and change the label
        idx = np.random.choice(np.where(new_labels == rand_lab)[0])
        new_labels[idx] = empty_lab
        
    return new_labels

# create an assignment matrix (N x k) given a vector of labels
def assignment_matrix(k, labels):
    return np.eye(k)[labels]

# create a function to return the average within cluster distance W(C)
def within_cluster(H, D):
    return np.trace(np.linalg.inv(H.T @ H) @ (H.T @ D @ H))/2

def k_means(X, k, max_iter, print_progress=False):
    n_samples, n_features = X.shape

    # labels: assign every sample to a cluster at random
    labels = np.random.randint(low=0, high=k, size=n_samples)
    
    # ensure there are no empty clusters
    labels = no_empty_labels(k, labels)
    
    X_labels = np.append(X, labels.reshape(-1,1), axis=1)
    
    # computing the centroids of each of the k clusters
    centroids = np.zeros((k, n_features))
    for i in range(k):
        centroids[i] = np.mean([x for x in X_labels if x[-1]==i], axis=0)[:-1]
    
    new_labels = np.zeros(len(X))
    difference = 0

    # k-means algorithm
    for i in range(max_iter):
        if print_progress:
            print('Iteration:', i)
        
        # distances: between data points and centroids
        distances = np.array([np.linalg.norm(X - c, axis=1) for c in centroids])
        # new_labels: computed by finding centroid with minimal distance
        new_labels = np.argmin(distances, axis=0)
        
        # Ensure that there are no empty labels
        new_labels = no_empty_labels(k, new_labels)
        
        # calculate the average within cluster distance
        W = within_cluster(assignment_matrix(k, labels), D)
        
        # Check if the labels are unchanged
        if (labels==new_labels).all():
            labels = new_labels
            ## print('Labels unchanged! Terminating k-means.')
            break

        else:
            # Calculate the percentage of changed labels
            difference = np.mean(labels!=new_labels)
            
            if print_progress:
                print('%4f%% labels changed' % (difference * 100))
            
            # Update labels and centroids
            labels = new_labels
            for c in range(k):
                if (labels == c).any():
                  centroids[c] = np.mean(X[labels==c], axis=0)
    
    return labels, W

def k_means_100(X, k, max_iter, print_progress=False):
    # Create lists to store the clusterings and average within-cluster distances
    label_list = []
    W_list = []
    
    for i in range(100):
        new_label, new_W = k_means(X, k, max_iter, print_progress=print_progress)
        label_list.append(new_label)
        W_list.append(new_W)
    
    return label_list, W_list

# Run k-means with 100 random initialisations for all values of k in interval [2, 10]
# Create a dictionary to store all the scores and clusterings obtained
k_means_all = {}

for k in range(2, 11):
    print("Loop k = " + str(k))
    k_means_all[k] = k_means_100(feature_matrix, k, 15)

# Find the index for the optimal clusters for each k
# By finding index of the cluster which maximises W(C) for that k
optimal_idx = {}
optimal_clusters = []
optimal_W = []
## average_W = []
for k in range(2, 11):
    optimal_idx[k] = np.argmin(k_means_all[k][1])
    optimal_W.append(k_means_all[k][1][optimal_idx[k]])
    ## average_W.append(np.mean(k_means_all[k][1]))
    
plt.plot(range(2, 11), optimal_W, '-x')
plt.title("Average Within-Cluster Distance vs k [k Means] 1")
plt.xlabel("k")
plt.ylabel("W(C)")
plt.grid()
plt.show()

"""
plt.plot(range(2, 11), average_W, '-x')
plt.title("Average Within-Cluster Distance vs k [k Means] 2")
plt.xlabel("k")
plt.ylabel("W(C)")
plt.grid()
plt.show()
"""

"""
2.1.2
"""

def total_sum_squares(X):
    centroid = np.mean(X, axis=0, keepdims=True)
    
    tss = np.sum(np.square(np.linalg.norm(X - centroid, axis = 1)))
    
    return tss

def within_sum_squares(X, labels):
    k = len(np.unique(labels))
    X_labels = np.append(X, labels.reshape(-1,1), axis=1)
    
    ssw = 0
    
    for i in range(k):
        
        # get all the elements in cluster k
        cluster_elements = np.array([x for x in X_labels if x[-1]==i])[:,:-1]
        
        # find the within cluster total sum of squares and add it to the total
        ssw += total_sum_squares(cluster_elements)
    
    return ssw

def CH_score(X, labels):
    k = len(np.unique(labels))
    N = X.shape[0]
    
    # Within sum of squares
    ssw = within_sum_squares(X, labels)
    
    # Between sum of squares
    ssb = total_sum_squares(X) - ssw
    
    # CH score
    score = ssb / ssw * (N - k) / (k - 1)
    
    return score

# Measure the CH score for each of the 100 iterations and store them in a dictionary
CH_scores = {}

# k loops
for k in range(2, 11):
    label_list, _ = k_means_all[k]

    print(k)
    CH_scores[k] = []
        
    # 100 loops
    for labels in label_list:
        CH_scores[k].append(CH_score(feature_matrix, labels))
        
average_CH_scores = []
for k in range(2, 11):
    average_CH_scores.append(np.mean(CH_scores[k]))


plt.plot(range(2, 11), average_CH_scores, '-x')
#plt.title("Average Calinski-Harabasz Score vs k [k Means] 1")
plt.xlabel("k")
plt.ylabel("CH score")
#plt.grid()
plt.show()

"""
2.1.3
"""
# We have 100 CH scores and 100 W(C) values for each k
# Create 9 x 100 matrix of the W(C) scores
W_matrix = np.zeros((9, 100))
CH_matrix = np.zeros((9, 100))
for k in range(2, 11):
    W_matrix[k-2, :] = k_means_all[k][1]
    CH_matrix[k-2, :] = CH_scores[k]

# Compute the RELATIVE variances of each of the metrics for a fixed k
W_rel_variance = np.var(W_matrix, axis=1) / np.mean(W_matrix, axis=1)
CH_rel_variance = np.var(CH_matrix, axis=1) / np.mean(CH_matrix, axis=1)

# Plot them
plt.plot(range(2, 11), W_rel_variance, '-o', label='W(C) variance')
plt.plot(range(2, 11), CH_rel_variance, '-o', label='CH score variance')
plt.title("Variances of different metrics of k Means Algorithm vs k")
plt.legend()
plt.grid()
plt.show()
plt.show()

# Robustness wouldn't show much variance etc
