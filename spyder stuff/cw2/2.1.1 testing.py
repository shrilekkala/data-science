# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# generate data
X, y = make_blobs(n_samples=200, n_features=2, centers=3, random_state=26, center_box=(0, 20))

# visualising the clusters
plt.figure(figsize=(10,8))
plt.scatter(X[:,0], X[:,1]);

n_samples, n_features = X.shape

# number of clusters k
k = 30

# labels: assign every sample to a cluster at random
np.random.seed(123)
labels = np.random.randint(low=0, high=k, size=n_samples)
X_labels = np.append(X, labels.reshape(-1,1), axis=1)

# computing the centroids of each of the k clusters
centroids = np.zeros((k, n_features))
for i in range(k):
  centroids[i] = np.mean([x for x in X_labels if x[-1]==i], axis=0)[0:2]
  
# check
centroids

# plot centroids
plt.figure(figsize=(10,8))
plt.scatter(X[:,0], X[:,1])
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=120);

max_iter = 15
new_labels = np.zeros(len(X))
difference = 0

# k-means algorithm
for i in range(max_iter):
    print('Iteration:', i)
    # distances: between data points and centroids
    distances = np.array([np.linalg.norm(X - c, axis=1) for c in centroids])
    # new_labels: computed by finding centroid with minimal distance
    new_labels = np.argmin(distances, axis=0)

    if (labels==new_labels).all():
        # labels unchanged
        labels = new_labels
        print('Labels unchanged! Terminating k-means.')
        break
    else:
        # labels changed
        # difference: percentage of changed labels
        difference = np.mean(labels!=new_labels)
        print('%4f%% labels changed' % (difference * 100))
        labels = new_labels
        for c in range(k):
            # update centroids by taking the mean over associated data points
            if (labels == c).any():
              centroids[c] = np.mean(X[labels==c], axis=0)
              
plt.figure(figsize=(10,8))
plt.scatter(X[:,0], X[:,1], c=labels);
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=120)