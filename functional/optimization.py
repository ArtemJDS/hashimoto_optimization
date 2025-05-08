import numpy as np
from functional.pseudohashimoto import compute_M, compute_M_weighted
from functional.basic_matrix_functions import find_vector_y_default, normalize_adjacency_matrix
from scipy.optimize import minimize 

class Optimizer:

    def __init__(self, C, target_eigenvalues, tolerance = None):
        self.C = C
        self.C_initial_flat = self.C.flatten()
        self.n = C.shape[0]
        self.best = [float('inf')]
        self.target_eigenvalues = target_eigenvalues
        self.tolerance = tolerance
        self.iter = 0

    def objective_function_symmetic(self, C_flat, n, us, Ks, C):

        '''
        Function for symmetric case
        '''

        C_matrix = C_flat.reshape((n, n))
        # C_matrix = (C_matrix + C_matrix.T)/2

        loss = 0
        for u, K in zip(us, Ks):
            M_C_star = compute_M(u, C_matrix)
            M_C = compute_M(u, C)
            loss += np.linalg.norm((M_C_star - (M_C - K)))
        l = loss
        if l < self.best[0]:
            self.best = (l,C_matrix, M_C_star, M_C, K)
            print(l)
        C_matrix = np.clip(C_matrix, 0, None)
        # C_matrix = normalize(C_matrix)        
        return l
    
    def objective_function_nonsymmetic(self, C_flat, n, us, Ks, C):

        '''
        Function for nonsymmetric case
        '''

        C_matrix = C_flat.reshape((n, n))
        C_matrix = np.clip(C_matrix, 0, None)
        # C_matrix = (C_matrix + C_matrix.T)/2
        # C_matrix = normalize_adjacency_matrix(C_matrix)

        loss = 0
        for u, K in zip(us, Ks):
            M_C_star = compute_M_weighted(u, C_matrix)
            M_C = compute_M_weighted(u, C)
            loss += np.linalg.norm((M_C_star - (M_C - K)))
        l = loss

          


        if l < self.best[0]:
            self.best = (l,C_matrix, M_C_star, M_C, K)
            print(l)
      
        return l

    def optimization(self, M_matrix_target_eigenvalues = [0]):
        us = []
        Ks = []
        
        assert len(self.target_eigenvalues) == len(M_matrix_target_eigenvalues), 'Number of target eigenvalues is not equal to eigenvalues to optimize'
        for lambd, ind in zip(self.target_eigenvalues, M_matrix_target_eigenvalues):
            u = 1/lambd
            # K = np.zeros_like(self.C, complex)
            M = compute_M_weighted(u, normalize_adjacency_matrix(self.C))
            eig = np.linalg.eig(M)
            sort = np.argsort(np.abs(eig.eigenvalues))
            eigenvalues = eig.eigenvalues[sort]
            eigenvectors = eig.eigenvectors[:, sort]
            K = np.outer(eigenvectors[:, ind], \
                        find_vector_y_default(eigenvectors[:, ind], eigenvalues[ind]))
            us.append(u)
            Ks.append(K.copy())
        


        result = minimize(self.objective_function_nonsymmetic, self.C_initial_flat, method='Powell',
                        args = (self.n, us, Ks, normalize_adjacency_matrix(self.C)),
                        tol = self.tolerance if self.tolerance else 0
                            )     
