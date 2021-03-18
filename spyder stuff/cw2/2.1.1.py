import numpy as np
import csv
#import matplotlib.pyplot as plt

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
2.1.1
"""
### NB whether to standardise at the beginning or not

# Create the NxN (34x34) distance matrix
difference = feature_matrix[:, :, None] - feature_matrix[:, :, None].T
D = np.sqrt((difference**2).sum(1))
