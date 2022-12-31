import numpy as np

from scipy import linalg as la

#from .decompositions import generalized_golub_kahan
#from .utils import soft_thresh

from psandqs import decompositions, utils

"""
Functions which implement variants of Bregman iteration.
"""


def bregman(A, b, mu, noise_norm, iterations):
    [m, n] = A.shape
    s = np.zeros((n,1))
    z = np.zeros((n,1))
    xhat = np.zeros((n,1))
    # Bregman method needs the constant eta that we find as \|B\|_2^2, 
    # where B is a matrix of small dimension resulting from Golub-Kahan decomposition.
    (U, B, V) = decompositions.generalized_golub_kahan(A, b, gk_eta=1.01, gk_delta=noise_norm, n_iter=100)
    eta = la.norm(B)

    history = []

    for i in range(1, iterations + 1):

        z = z + A.T@(b-A@xhat)
        xhat = (0.9/eta)*utils.soft_thresh(z, mu)
        history.append(xhat)
        if (np.linalg.norm(A@xhat - b) < noise_norm):
            it = i
            break
    return xhat, history

if __name__ == "__main__":

    A = np.random.rand(10, 10)
    b = np.random.rand(10, 1)

    I = np.random.rand(10,10)


    out = bregman(A, b, 0.01, 100, iterations=100)

    print(out)