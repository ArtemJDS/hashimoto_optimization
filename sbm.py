import numpy as np

def create_sbm_matrix(N, m, w_ins, w_outs, seed = 1):
    np.random.seed(seed)
    
    cluster_sizes = [N // m] * m
    for i in range(N % m):  
        cluster_sizes[i] += 1

    adjacency_matrix = np.zeros((N, N))

    cluster_indices = np.cumsum([0] + cluster_sizes)  

    for i in range(m):
        for j in range(m):
            start_i, end_i = cluster_indices[i], cluster_indices[i + 1]
            start_j, end_j = cluster_indices[j], cluster_indices[j + 1]

            if i == j:
                probability = w_ins[i]
            else:
                probability = max(w_outs[i], w_outs[j])  

            block = np.random.rand(end_i - start_i, end_j - start_j) < probability
            adjacency_matrix[start_i:end_i, start_j:end_j] = block

    adjacency_matrix = np.triu(adjacency_matrix) + np.triu(adjacency_matrix, 1).T
    adjacency_matrix = adjacency_matrix.astype(int)
    return adjacency_matrix


N = 100 # number of nodes
m = 2 # number of clusters
w_ins = [0.5, 0.5] # probability of contact inside clusters 
w_outs = [0.05, 0.05] # probability of contact outside clusters 
matrix = create_sbm_matrix(N, m, w_ins, w_outs, seed = 1)