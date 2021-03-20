import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    eigenvectors = eigenvectors[:,sorted_index]

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


def PCA_plotter(X_pca_1, X_pca_2, X_pca_3, labels = None):
    fig = plt.figure(figsize=(15, 4))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    fig.subplots_adjust(hspace=0.2, wspace=0.3)
    
    # 1D plot for d = 1
    ax0.scatter(X_pca_1.T, np.zeros(34), marker='x', c=labels)
    ax0.set_title("Projection onto 1-D PCA space")
    ax0.set_xlabel("PCA Dimension 1")
    ax0.axes.yaxis.set_visible(False)
    
    # 2D plot for d = 2
    ax1.scatter(X_pca_2[:,0], X_pca_2[:,1], marker='o', c=labels)
    ax1.set_title("Projection onto 2-D PCA space")
    ax1.set_xlabel("PCA Dimension 1")
    ax1.set_ylabel("PCA Dimension 2")
    
    # 3D plot for d = 3
    ax2.set_title("Projection onto 3-D PCA space")
    ax2.scatter(X_pca_3[:,0], X_pca_3[:,1], X_pca_3[:,2], marker='o', c=labels)
    ax2.set_xlabel("PCA Dimension 1")
    ax2.set_ylabel("PCA Dimension 2")
    ax2.set_zlabel("PCA Dimension 3")
    ax2.view_init(25, 25) 
    plt.show()

X_pca_1 = PCA_info[1][0]
X_pca_2 = PCA_info[2][0]
X_pca_3 = PCA_info[3][0]

PCA_plotter(X_pca_1, X_pca_2, X_pca_3)
PCA_plotter(X_pca_1, X_pca_2, X_pca_3, labels = optimal_label)

# Check
"""
from sklearn.decomposition import PCA
pca = PCA(n_components=10, svd_solver='full')
X_sk_pca = pca.fit_transform(normalized_feature_matrix)
plt.scatter(X_sk_pca[:,0],X_sk_pca[:,1])
plt.show()
pca.explained_variance_ratio_
"""

"""
2.2.2
"""
# Compute total variance
C = 1.0/(len(normalized_feature_matrix)-1) * np.dot(normalized_feature_matrix.T, normalized_feature_matrix)
## equivalent to np.cov(normalized_feature_matrix.T)
all_eigenvalues, _ = np.linalg.eig(C)
total_variance = abs(all_eigenvalues.sum())

C2 = np.dot(normalized_feature_matrix.T, normalized_feature_matrix)
all_eigenvalues2, _ = np.linalg.eig(C2)
total_variance2 = abs(all_eigenvalues2.sum())

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

## Cumulative Explained Variance
"""
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
"""



## As we normalised F, (I - 1/N 1 1.T)F = F (see page 16 of slides)
## So F.T F is exactly the sample covariance matrix Cx
## The eigenvectors of Cx are the basis used in PCA, i.e. principal components (page 10)

## So when we take the spectral decomposition of F^T F, we recover the variances of the components of PCA
## We can then plot these variances cumulatively as a proportion of the total variance:

# spectral decomposition of F^T F
F = normalized_feature_matrix
all_e_vals, all_e_vecs = np.linalg.eig(F.T @ F)
## Cumulative eigenvalue sums
plt.plot(range(1, 101),(abs(np.cumsum(all_e_vals)) / abs(all_e_vals.sum()))[:100], marker='.')
plt.title("Cumulative Graph of Proportional Eigenvalues of F^T F")
plt.xlabel("Eigenvalue Number (ordered in decreasing magnitude)")
plt.ylabel("Cumulative Sum")
plt.grid()
plt.show()    

## Just examine the first 20
plt.plot(range(1, 21),(abs(np.cumsum(all_e_vals)) / abs(all_e_vals.sum()))[:20], marker='.')
plt.title("Cumulative Graph of Proportional Eigenvalues of F^T F")
plt.xlabel("Eigenvalue Number (ordered in decreasing magnitude)")
plt.ylabel("Cumulative Sum")
plt.gca().set_xticks(np.arange(0,21))
plt.grid()
plt.show()    

## first 2 components contain approximately 50% of the variance
## and you only need around 11 of the 100 components to describe over 90% of the variance

## (We also know that in SVD the singular values of F are the square root of the e/vals of F^T F = Cx)