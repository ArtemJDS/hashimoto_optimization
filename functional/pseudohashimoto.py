import numpy as np

def compute_A(u, C):
    return np.array([[u * C[i, j] / (1 - u**2 * C[i, j]**2) for j in range(C.shape[1])]  \
                     for i in range(C.shape[0])
                     ])

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
