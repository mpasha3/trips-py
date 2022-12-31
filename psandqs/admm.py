import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

from psandqs.utils import *

"""
Functions which implement variants of ADMM.
"""


def update_x(A, b, L, y, z, rho):

    if isinstance(A, LinearOperator):

        x = spla.spsolve((A.T@A + rho*L.T@L).tosparse(), (A.T@b + L.T@y + rho*L.T@z), permc_spec='NATURAL')

    else:
        x = la.solve((A.T@A + rho*L.T@L).todense(), (A.T@b + L.T@y + rho*L.T@z))

    x = x.reshape(A.shape[1], b.shape[1])
    return x

    
def update_z(L, x, y, rho):

    z = soft_thresh(L @ x - y/rho, 0.9)
    return z


def admm(A, b, L, max_iter, rho):

    [s,n] = L.shape
    x_history = []

    z = np.zeros((s,1))
    y = np.zeros((n,1))

    
    for i in range(max_iter+1):
        x = update_x(A, b, L, y, z, rho)
        z = update_z(L, x, y, rho)
        y = y + rho*(z-L@x)
        x_history.append(x)

    return x, x_history


if __name__ == "__main__":

    A = np.random.rand(10, 10)
    b = np.random.rand(10, 1)

    I = np.random.rand(10,10)


    out = admm(A, b, np.flip(I), max_iter=100, rho=10**(-3))

    print(out)