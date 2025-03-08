import numpy as np
from functional.basic_matrix_functions import get_symmetric_component, get_skew_symmetric_component, normalize_adjacency_matrix

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

def compute_A_weighted(u, C_sm, C_sk):
    A = np.zeros_like(C_sm)
    l = A.shape[0]
    for i in range(l):
         for j in range(l):
              A[i,j] = 2 * u *(C_sk[i,j]-C_sm[i,j])/(C_sk[i,j]**2*u**2 - C_sm[i,j]**2*u**2 + 1)
    return A

def compute_D_weighted(u, C_sm, C_sk):
    D = np.zeros_like(C_sm, complex)
    n = D.shape[0]
    for i in range(n):
        for j in range(n):
                D[i, i] += 2 * u**2 * (-C_sk[i,j]+C_sm[i,j]) * (C_sk[i,j]+C_sm[i,j])/(C_sk[i,j]**2*u**2 - C_sm[i,j]**2*u**2 + 1)
                
    return D

def compute_M_weighted(u, C):
    C = normalize_adjacency_matrix(C)
    C_sm, C_sk = get_symmetric_component(C), get_skew_symmetric_component(C)
    A = compute_A_weighted(u, C_sm, C_sk)
    D = compute_D_weighted(u, C_sm, C_sk)
    I = np.eye(*C.shape)
    return I + 1/2*(A + D)

def compute_A_sk(u, C):
    A = np.zeros_like(C)
    l = A.shape[0]
    for i in range(l):
         for j in range(l):
              A[i,j] = 2*C[i,j]*u/(C[i,j]**2*u**2 + 1)
    return A

def compute_D_sk(u, C):
    D = np.zeros_like(C, complex)
    n = D.shape[0]
    for i in range(n):
        for j in range(n):
                D[i, i] += -2*C[i,j]**2*u**2/(C[i,j]**2*u**2 + 1)
    return D

def compute_M_sk(u, C):
    A = compute_A_sk(u, C)
    D = compute_D_sk(u, C)
    I = np.eye(*C.shape)
    return I + 1/2*(A + D)