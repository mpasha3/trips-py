"""
Functions which implement Krylov decompositions.
"""

from random import betavariate
from re import X
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import newton


def generalized_golub_kahan(A, guess, n_iter):

    (rows, cols) = A.shape

    betas = np.zeros(shape=(n_iter))
    alphas = np.zeros(shape=(n_iter))

    U = np.zeros((rows, n_iter+1))
    V = np.zeros((cols, n_iter))

    U[:,0] = (guess/np.linalg.norm(guess)).flatten()

    for ii in range(n_iter):

        V[:,ii] = A.T @ U[:,ii] - betas[ii-1] * V[:,ii-1]
        alphas[ii] = np.linalg.norm(V[:,ii])
        V[:,ii] = V[:,ii]/alphas[ii]

        U[:,ii+1] = A @ V[:,ii] - alphas[ii] * U[:,ii]
        betas[ii] = np.linalg.norm(U[:,ii+1])
        U[:,ii+1] = U[:,ii+1]/betas[ii]


    return (U,betas,alphas,V)


def lanczos_biortho_pasha(A, guess, iter):

    # dimensions
    N = len(guess)
    M = len(A.T @ guess)

    # preallocate
    U = np.zeros(shape=(N, iter+1))
    V = np.zeros(shape=(M, iter))

    v = np.zeros(shape=(M))


    # normalize initial guess
    beta = np.linalg.norm(guess)

    assert beta != 0

    u = guess/beta

    U[:,0] = u

    # begin bidiagonalization

    for ii in range(0,iter):

        r = A.T @ u
        r = r - beta*v

        for jj in range(0,ii-1): # reorthogonalization

            r = r - (V[:,jj].T @ r) * V[:,jj]

        alpha = np.linalg.norm(r)

        v = r/alpha


        V[:,ii] = v.flatten()

        p = A @ v

        p = p - alpha*u


        for jj in range(0, ii):

            p = p - (U[:,jj].T @ p) * U[:,jj]

        beta = np.linalg.norm(p)

        u = p / beta

        U[:, ii+1] = u

    return (U, beta, V)