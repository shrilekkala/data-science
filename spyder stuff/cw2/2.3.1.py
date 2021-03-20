# importsimport numpy as npimport matplotlib.pyplot as pltimport csvimport networkx as nxfrom scipy.sparse import linalgimport pandas as pdimport seaborn as snsnp.random.seed(1024)# Load the matrices# Feature MatrixF = np.loadtxt(open("data/feature_matrix_karate_club.csv", "rb"), delimiter=",", skiprows=1)individuals = F[:, 0]feature_matrix = F[:, 1:]A = np.loadtxt(open("data/karate_club_graph.csv", "rb"), delimiter=",", skiprows=1)adjacency_matrix = A[:, 1:]S = list(csv.reader(open("data/ground_truth_karate_club.csv")))true_split = S[1:]# Convert to numpy array with "Mr Hi" being 0 and "Officer" being 1true_split_np = np.vstack((np.arange(34), np.zeros(34))).Tfor i in range(34):    if true_split[i][1] == "Officer":        true_split_np[i, 1] = 1              """Network X"""      network_g = nx.karate_club_graph()# extracting adjacency matrixnetwork_A = nx.adjacency_matrix(network_g).toarray().astype(np.float64)# build graph from CSV adjacency matrixG = nx.from_numpy_array(adjacency_matrix)G.nodesG.edgesG.number_of_nodes()G.number_of_edges()nx.draw(G)plt.show()"""2.3.1"""def degree_cent(A):    # compute degrees from the adjacency matrix    degree = A.sum(axis=1)        # compute the total number of edges    e = np.sum(A) / 2        # divide degree by 2E to get the centrality    return degree / (2 * e)    def pagerank_cent(A):    # compute the inverse of the degree matrix    Dinv = np.diag(np.reciprocal(A.sum(axis=1)))        # number of nodes    N = A.shape[0]        # set the teleportation parameter to the customary value    alpha = 0.85            # randomly initialise the vector c_pr    c_pr = np.random.randn(N)        # compute the page rank centrality via the power iteration method        max_iterations = 100    for i in range(max_iterations):        old_c_pr = c_pr        c_pr = alpha * A @ Dinv @ old_c_pr + (1 - alpha) / N                # check for convergence        if np.linalg.norm(c_pr - old_c_pr) < 1e-9:            break        return c_prdef eigenvector_cent(A):    # compute the eigenvector associated with the largest eigenvalue    eigenvalue, eigenvector = linalg.eigsh(A, 1, which="LM", return_eigenvectors=True)        # sometimes scipy returns negative eigenvector instead, so change sign accordingly    if eigenvector[0][0] < 0:        eigenvector = -1 * eigenvector    return eigenvector.T[0]# compute the three centralitiesc_pagerank = pagerank_cent(adjacency_matrix)c_degree = degree_cent(adjacency_matrix)c_eigenvec = eigenvector_cent(adjacency_matrix)# compute the centralities via NetworkXc_pr = np.array(list(nx.pagerank(network_g, alpha=0.85).values()))degree_centrality = np.array(list(nx.centrality.degree_centrality(network_g).values()))c_d = degree_centrality / np.sum(degree_centrality)c_e = np.array(list(nx.eigenvector_centrality(network_g).values()))# check they are all the sameprint(np.linalg.norm(c_pagerank - c_pr))print(np.linalg.norm(c_degree - c_d))print(np.linalg.norm(c_eigenvec - c_e))# Report the values of the centralitiescentralities_comparison = pd.DataFrame(index = ["Node " + str(i) for i in range(1, 35)],                                       columns = ["PageRank", "Degree", "Eigenvector"],                                       data = np.vstack((c_pagerank, c_degree, c_eigenvec)).T)centralities_comparison# Plot the standardised centralities (so that their sum is 1)# Note sum of c_degree and c_pagerank is already oneplt.plot(range(1, 35), c_degree, 'rx', label = "Degree")plt.plot(range(1, 35), c_pagerank, 'bx', label = "PageRank")plt.plot(range(1, 35), c_eigenvec / np.sum(c_eigenvec), 'gx', label = "Eigenvector")plt.xlabel("Node Number")plt.ylabel("Standardised Centrality Measures")plt.legend()plt.title("Centralities of all the Nodes of the Feature Matrix")plt.show()pagerank_ranking = np.argsort(c_pagerank)degree_ranking = np.argsort(c_degree)eigenvec_ranking = np.argsort(c_eigenvec)np.vstack((pagerank_ranking, degree_ranking, eigenvec_ranking))sorted_centralities = pd.DataFrame(index = ["PageRank", "Degree", "Eigenvector"],                                   columns = ["Highest Centrality"] + [" " for i in range(32)] + ["Lowest Centrality"],                                   data=np.vstack((pagerank_ranking,                                                    degree_ranking,                                                    eigenvec_ranking)))sorted_centralities# 11 appears in top 5 of all three centralities# 16, 22, 26 appear in top 5 of two of the three centralities# Correlation Plotsprint(centralities_comparison.corr())sns.heatmap(centralities_comparison.corr(), annot=True, fmt='.3f')plt.suptitle("Heatmap of the Correlation Matrix for Centrality Measures")plt.show()"""Using NetworkX valuesdf_nx_graph_centrality = pd.DataFrame(np.array([c_pr, c_d, c_e]).T,                                       columns=["PageRank Nx", "Degree Nx", "Eigenvector Nx"])sns.heatmap(df_nx_graph_centrality.corr(), annot=True, fmt='.3f')plt.show()"""# Pair Plotsfig = sns.pairplot(centralities_comparison, kind='reg')plt.suptitle("Pair Plots comparing Centrality Measures",fontsize=15, y=1.05)plt.show()# centrality ranking discussion as in CW3"""2.3.2"""