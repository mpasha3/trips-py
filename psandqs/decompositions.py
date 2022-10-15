"""
Functions which implement Krylov decompositions.
"""

from random import betavariate
from re import X
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import newton

from tqdm import tqdm


def generalized_golub_kahan_deprecated(A, guess, n_iter):

    (rows, cols) = A.shape

    betas = np.zeros(shape=(n_iter))
    alphas = np.zeros(shape=(n_iter))

    U = np.zeros((rows, n_iter+1))
    V = np.zeros((cols, n_iter))

    U[:,0] = (guess/np.linalg.norm(guess)).flatten()

    for ii in tqdm(range(n_iter), desc='generating golub-kahan basis...'):

        V[:,ii] = A.T @ U[:,ii] - betas[ii-1] * V[:,ii-1]
        alphas[ii] = np.linalg.norm(V[:,ii])
        V[:,ii] = V[:,ii]/alphas[ii]

        U[:,ii+1] = A @ V[:,ii] - alphas[ii] * U[:,ii]
        betas[ii] = np.linalg.norm(U[:,ii+1])
        U[:,ii+1] = U[:,ii+1]/betas[ii]


    return (U,betas,alphas,V)


def generalized_golub_kahan(A, b, n_iter, dp_stop=False, **kwargs):

    eta = kwargs['gk_eta'] if ('gk_eta' in kwargs) else 0.001
    delta = kwargs['gk_delta'] if ('gk_delta' in kwargs) else 0.001

    (rows, cols) = A.shape

    betas = np.zeros(shape=(1)) # start the set of alphas and betas in the bidiagonal matrix
    alphas = np.zeros(shape=(1)) # return matrix instead of aarrays

    U = np.zeros((rows, 1+1)) # start the bases at dimension of iter and iter+1
    V = np.zeros((cols, 1))

    U[:,0] = (b/np.linalg.norm(b)).flatten() # initialize U with the normalized guess
                                       
    iterations = 0
                                       
    res_norm = np.inf # set for the while condition
                                       
    for ii in tqdm(range(n_iter), desc='generating golub-kahan basis...'):

        if ((dp_stop == True) and (res_norm <= eta*delta)):
            print("discrepancy principle satisfied, halting early.")
            break
        
        if iterations != 0:
            U = np.pad(U, ((0,0), (0,1)) )  # at each iteration that doesn't satisfy the discrepancy principle,
            V = np.pad(V, ((0,0), (0,1)) )  # add an additional column to the bases and entry to the bidiagonal entries.
            betas = np.pad(betas, ((0,1)) )
            alphas = np.pad(alphas, ((0,1)) )

        V[:,iterations] = A.T @ U[:,iterations] - betas[iterations-1] * V[:,iterations-1] # perform bidiagonalization as before
        alphas[iterations] = np.linalg.norm(V[:,iterations])
        V[:,iterations] = V[:,iterations]/alphas[iterations]

        U[:,iterations+1] = A @ V[:,iterations] - alphas[iterations] * U[:,iterations]
        betas[iterations] = np.linalg.norm(U[:,iterations+1])
        U[:,iterations+1] = U[:,iterations+1]/betas[iterations]

        if (dp_stop == True):
                                       
            B = np.pad(np.diag(alphas),( (0,1),(0,0) )) + np.pad(np.diag(betas), ( (1,0),(0,0) ) ) # constrct B from the bidiagonal entries
                                        
            bhat = U.T @ b
                                        
            y = np.linalg.lstsq(B, bhat, rcond=None)[0] # solve the least squares problem
            
            x = V @ y # project back
                                        
            res_norm = np.linalg.norm(A @ x - b) # and find the norm of the residual
                                       
        iterations += 1


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