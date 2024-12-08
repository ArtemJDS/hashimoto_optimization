import numpy as np

def create_sbm_matrix(N, m, w_ins, w_outs, seed):
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

def construct_non_backtracking_matrix_parallel(A, flow = True):

    N = A.shape[0] 
    num_edges = 0

    for i in range(N):
        for j in range(N):
            if A[i, j] != 0:
                num_edges += 1

    edges = np.zeros((num_edges, 2), dtype=np.int32)

    edge_idx = 0
    for i in range(N):
        for j in range(N):
            if A[i, j] != 0:
                edges[edge_idx, 0] = i
                edges[edge_idx, 1] = j
                edge_idx += 1

    B = np.zeros((num_edges, num_edges), dtype=np.complex128)

    coefs = np.sum(A, axis = 1) - 1
    for idx1 in range(num_edges):
        i, j = edges[idx1, 0], edges[idx1, 1]
        for idx2 in range(num_edges):
            k, l = edges[idx2, 0], edges[idx2, 1]
            if i == l and j != k:
                if flow is True:
                    B[idx1, idx2] = A[k,l] / coefs[i] 
                else:
                    B[idx1, idx2] = A[k,l]

    return B

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def find_vector_y_default(x, a):
    norm_x_squared = np.dot(x, x)
    k = a / norm_x_squared
    y = k * x
    return y

def normalize(A):
    X = A.copy()
    X = (X.T / (np.abs(X.T).sum(0)-1)).T

    return X
