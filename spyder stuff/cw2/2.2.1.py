import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import scipy as sc


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
        
"""
2.2.1
"""

# Some preprocessing of the data
def normalize(X):
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std_filled = std.copy()
    std_filled[std==0] = 1.
    Xbar = ((X-mu)/std_filled)
    return Xbar

normalized_feature_matrix = normalize(feature_matrix)

from scipy.sparse import linalg

def pca_function(X, k):
    n = X.shape[0]

    # create covariance matrix S
    C = X.T @ X / (n-1)

    # compute eigenvalues and eigenvectors using the eigsh scipy function
    ## LM means find the k largest in magnitude (eigenvalues)
    eigenvalues, eigenvectors = linalg.eigsh(C, k, which="LM", return_eigenvectors=True) 

    # sorting the eigenvectors and eigenvalues from largest to smallest eigenvalue
    sorted_index = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_index]
    eigenvectors = eigenvectors[:,sorted_index ]

    # transform our data
    X_pca = X.dot(eigenvectors)

    return X_pca, eigenvectors, eigenvalues

# apply the pca function for d in [1, 10] and store results in a dictionary
PCA_info = {}
for d in range(1, 11):
    print("Loop d = " + str(d))
    PCA_info[d] = pca_function(normalized_feature_matrix, d)

"""
i)
"""
### use optimal label from before ###
optimal_label = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
                          1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1])

X_pca_1 = PCA_info[1][0]
X_pca_2 = PCA_info[2][0]
X_pca_3 = PCA_info[3][0]

plt.scatter(X_pca_1.T, np.zeros(34), marker='x', c=optimal_label)
plt.title("Plot of the Dataset Projected onto 1-D PCA space")
plt.xlabel("PCA Dimension 1")
ax = plt.gca()
ax.axes.yaxis.set_visible(False)
plt.show()

plt.scatter(X_pca_2[:,0], X_pca_2[:,1], marker='o', c=optimal_label)
plt.title("Plot of the Dataset Projected onto 2-D PCA space")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.show()

fig = plt.figure(figsize=(4, 4))
ax = Axes3D(fig) 
plt.title("Plot of the Dataset Projected onto 3-D PCA space")
ax.scatter(X_pca_3[:,0], X_pca_3[:,1], X_pca_3[:,2], marker='o', c=optimal_label)
ax.set_xlabel("PCA Dimension 1")
ax.set_ylabel("PCA Dimension 2")
ax.set_zlabel("PCA Dimension 3")
ax.view_init(25, 25) 
plt.show()


# Check
from sklearn.decomposition import PCA
pca = PCA(n_components=10, svd_solver='full')
X_sk_pca = pca.fit_transform(normalized_feature_matrix)
plt.scatter(X_sk_pca[:,0],X_sk_pca[:,1])
plt.show()
pca.explained_variance_ratio_

"""
2.2.2
"""
# Compute total variance
C = 1.0/(len(normalized_feature_matrix)-1) * np.dot(normalized_feature_matrix.T, normalized_feature_matrix)
all_eigenvalues, _ = np.linalg.eig(C)
total_variance = abs(all_eigenvalues.sum())

# Compute explained variances
explained_variances = {}

for d in range(1, 11):
    e_values = PCA_info[d][2]
    explained_variances[d] = e_values /  total_variance

## Proportion
plt.plot(range(1, 11), explained_variances[10], marker='o', label="Proportion of Explained Variance")
# plt.plot(range(1, 11), cumulative_explained_variances, marker='o', label="Cumulative")
plt.title("Explained Variance of PCA approximations of dimension d")
plt.xlabel("Principal Component d")
plt.ylabel("Proportion of Explained Variance")
plt.grid()
plt.show()

## Cumulative
cumulative_explained_variances = np.cumsum(explained_variances[10])
plt.plot(range(1, 11), cumulative_explained_variances, marker='.', label="Cumulative Explained Variance")
# plt.plot(range(1, 11), cumulative_explained_variances, marker='o', label="Cumulative")
plt.title("Explained Variance of PCA approximations of dimension d")
plt.xlabel("Principal Component d")
plt.ylabel("Cumulative Explained Variance")
plt.grid()
plt.show()
## first 2 components contain approximately 50% of the variance
## while you need around 50 components to describe close to 90% of the variance

# spectral decomposition of F^T F
F = normalized_feature_matrix
all_e_vals, all_e_vecs = np.linalg.eig(F.T @ F)


F_mean = np.mean(feature_matrix, axis = 0)
