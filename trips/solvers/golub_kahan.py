#!/usr/bin/env python
"""
Definition of test problems
--------------------------------------------------------------------------
Created December 10, 2022 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha and Connor Sanderford"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com and csanderf@asu.edu; connorsanderford@gmail.com"
"""
Functions which implement decompositions based on Krylov subspaces.
"""
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import newton

def golub_kahan(A, b, n_iter, dp_stop = False, **kwargs):
    """
    Description: Computes the rank-n Golub-Kahan factorization of A, with initial guess the given vector b.

    Inputs:
    A: the matrix to be factorized.

    b: an initial guess for the first basis vector of the factorization.

    n_iter: the number of iterations over which to factorize A.

    dp_stop: whether or not to use the discrepancy principle to halt further factorization. Defaults to false.

    Outputs:
    U (m x n_iter+1) an orthonormal matrix

    V (n x n_iter) an orthonormal matrix

    S (n_iter+1 x n_iter) a bidiagonal matrix.
    
    Calling with the minimal number of arguments:

    (U,S,V) = generalized_golub_kahan(A, b, n_iter)

    Calling with all the arguments necessary for discrepancy principle stopping:

    (U,S,V) = generalized_golub_kahan(A, b, n_iter, dp_stop=True, gk_eta=1.001, gk_delta=0.001)
    
    """

    eta = kwargs['gk_eta'] if ('gk_eta' in kwargs) else 1.001
    delta = kwargs['gk_delta'] if ('gk_delta' in kwargs) else 0.001

    (rows, cols) = A.shape

    betas = np.zeros(shape=(1)) # start the set of alphas and betas in the bidiagonal matrix
    alphas = np.zeros(shape=(1)) # return matrix instead of arrays

    U = np.zeros((rows, 1+1)) # start the bases at dimension of iter and iter+1
    V = np.zeros((cols, 1))

    U[:,0] = (b/np.linalg.norm(b)).flatten() # initialize U with the normalized guess
                                       
    iterations = 0
                                       
    res_norm = np.inf # set for the while condition
                                       
    for ii in tqdm(range(n_iter), desc='generating basis...'):

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
                                       
            S = np.pad(np.diag(alphas),( (0,1),(0,0) )) + np.pad(np.diag(betas), ( (1,0),(0,0) ) ) # constrct B from the bidiagonal entries
                                        
            bhat = U.T @ b
                                        
            y = np.linalg.lstsq(S, bhat, rcond=None)[0] # solve the least squares problem
            
            x = V @ y # project back
                                        
            res_norm = np.linalg.norm(A @ x - b) # and find the norm of the residual
                                       
        iterations += 1


    S = np.zeros(shape=(alphas.shape[0]+1, alphas.shape[0]) )
    S[range(0,alphas.shape[0]), range(0,alphas.shape[0])] = alphas
    S[range(1,alphas.shape[0]+1), range(0,alphas.shape[0])] = betas


    return (U,S,V)

