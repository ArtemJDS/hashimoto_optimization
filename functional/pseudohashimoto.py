import numpy as np
from functional.basic_matrix_functions import get_normalization_value

def compute_A(u, C):
    return np.array([[u * C[i, j] / (1 - u**2 * C[i, j]**2) for j in range(C.shape[1])]  \
                     for i in range(C.shape[0])
                     ], dtype=complex)

def compute_D(u, C):
    D = np.zeros_like(C, complex)
    n = D.shape[0]
    for i in range(n):
        for j in range(n):
                D[i, i] += u**2 * C[i, j]**2 / (1 - u**2 * C[i, j]**2)
    return D

def compute_M(u, C):
    A = compute_A(u, C)
    D = compute_D(u, C)
    I = np.eye(*C.shape)
    return I - A + D

def compute_A_weighted(u, C):
    return np.array([[u * C[i, j] * 1/get_normalization_value(i,j, C) / (1 - u**2 * C[i, j] **2 * 1/get_normalization_value(i,j, C)**2) for j in range(C.shape[1])]  \
                     for i in range(C.shape[0])                                                                               # i, j 
                     ])

def compute_D_weighted(u, C):
    D = np.zeros_like(C, complex)
    n = D.shape[0]
    for i in range(n):
        for j in range(n):
                D[i, i] += u**2 * C[i, j]**2  * 1/get_normalization_value(j,i , C)**2/ (1 - u**2 * C[i, j]**2 * 1/get_normalization_value(j, i , C)**2 )
    return D

def compute_M_weighted(u, C):
    A = compute_A_weighted(u, C)
    D = compute_D_weighted(u, C)
    I = np.eye(*C.shape)
    return I - A + D