import numpy as np
import scipy.linalg as la

from .utils import soft_thresh

"""
Functions which implement variants of ADMM.
"""


def update_x(A, b, L, y, z, rho):
    print((A.T@b + L.T@y).shape)
    x = la.solve((A.T@A + rho*L.T@L), (A.T@b + L.T@y + rho*L.T@z))
    return x

    
def update_z(L, x, y, rho):
    z = soft_thresh(L @ x - y/rho, 0.9)
    return z


def admm(A, b, L, max_iter, x_true, rho):
    if x_true.shape[1] >1:
        x_true = x_true.flatten()

    [s,n] = L.shape

    z = np.zeros((s,1))
    y = np.zeros((n,1))

    m = x_true.shape
    
    for i in range(max_iter+1):
        temp = update_x(A, b, L, y, z, rho)
        x = update_x(A, b, L, y, z, rho)
        z = update_z(L, x, y, rho)
        y = y + rho*(z-L@x)

    return x