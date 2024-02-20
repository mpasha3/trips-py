#!/usr/bin/env python
"""
Functions which implement decompositions based on Krylov subspaces.
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, Connor Sanderford, and Ugochukwu Obinna Ugwu"
__affiliations__ = 'MIT and Tufts University, University of Bath, Arizona State University, and Tufts University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu; connorsanderford@gmail.com; Ugochukwu.Ugwu@tufts.edu"

import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.sparse import diags

from tqdm import tqdm
def arnoldi(A: 'np.ndarray[np.float]', b: 'np.ndarray[np.float]', n_iter: int, dp_stop=False, **kwargs ) -> 'Tuple[np.ndarray[np.float], np.ndarray[np.float]]':
    """
    Description: Computes the rank-n Arnoldi factorization of A, with initial guess a given vector b.
    Input: 
        A: the matrix to be factorized.

        b: an initial guess for the first basis vector of the factorization.

        n_iter: the number of iterations over which to factorize A.

        dp_stop: whether or not to use the discrepancy principle to halt further factorization. Defaults to false.
    Output:
        Q (m x n_iter+1) an orthonormal matrix,
        H (n_iter+1 x n_iter) an upper Hessenberg matrix.

    Calling with the minimal number of arguments:

        (Q,H) = arnoldi(A, b, n_iter)

    Calling with all the arguments necessary for discrepancy principle stopping:

        (Q,H) = arnoldi(A, b, n_iter, dp_stop=True, gk_eta=1.001, gk_delta=0.001)
    """
    if ('shape' in kwargs):
        shape = kwargs['shape'] 
    elif type(A) == 'function' and shape not in kwargs['shape']:
        raise ValueError("The observation matrix A is a function. The shape must be given as an input")
    else:
        sx = A.shape[0]
        sy = A.shape[0]
        shape = [sx, sy]

    # If eta is not in the arguments, set it to the default value 1.001
    eta = kwargs['gk_eta'] if ('gk_eta' in kwargs) else 1.001
    delta = kwargs['gk_delta'] if ('gk_delta' in kwargs) else 0.001
    
    if (A.shape[0] is not A.shape[1]):
        raise ValueError("Arnoldi can not be used. The operator is not square")
    
    (rows, cols) = A.shape

    # preallocate

    Q = np.zeros((rows, 1+1))
    H = np.zeros((1+1, 1))

    # normalize b
    b = b/np.linalg.norm(b)

    # b is first basis vector
    Q[:, 0] = b.flatten()#reshape((-1,1))

    iterations = 0

    res_norm = np.inf

    for ii in tqdm(range(0,n_iter), desc = "generating basis..."): # for each iteration over the method:

        if ((dp_stop==True) and (res_norm <= eta*delta)):

            print('discrepancy principle satisfied, stopping early.')

            break

        if iterations != 0:
            Q = np.pad(Q, ((0,0), (0,1)) )  # at each iteration that doesn't satisfy the discrepancy principle,
            H = np.pad(H, ((0,1), (0,1)) )  # add an additional column to the bases and entry to the bidiagonal entries.

        b_nplus1  = A @ Q[:,ii] # generate the next vector in the Krylov subspace

        for jj in range(0,iterations): # for each iteration *that has been previously completed*:

            H[jj,ii] = np.dot( Q[:,jj], b_nplus1 ) # calculate projections of the new Krylov vector onto previous basis elements

            b_nplus1 = b_nplus1 - H[jj,ii] * Q[:,jj] # and orthogonalize the new Krylov vector with respect to previous basis elements

        if ii < n_iter:
            H[ii+1, ii] = np.linalg.norm(b_nplus1, 2)

            if H[ii+1,ii] == 0:
                return (Q,H)

            Q[:, ii+1] = b_nplus1/H[ii+1,ii]

        if (dp_stop == True):

            bhat = Q[:,:-1].T @ b

            y = np.linalg.lstsq(H[:-1,:].T @ H[:-1,:], bhat, rcond=None)[0]

            x = Q[:,:-1] @ y

            res_norm = np.linalg.norm(A @ x - b)

        iterations += 1

    return (Q,H)

def golub_kahan(A, b, n_iter, dp_stop=False, **kwargs):
    """
    Description: Computes the Golub-Kahan factorization of A, with initial guess the given vector b.

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

    U[:,0] = (b/np.linalg.norm(b)).flatten()#reshape((-1,1)) # initialize U with the normalized guess
                                       
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

def arnoldi_update(A, V, H):

    k = H.shape[0]
    vtemp = V[:,-1]

    vtemp  = A @ vtemp # generate the next vector in the Krylov subspace

    htemp = np.zeros((k,1))

    for jj in range(0,k): # for each iteration *that has been previously completed*:
        htemp[jj] = np.dot( V[:,jj], vtemp ) # calculate projections of the new Krylov vector onto previous basis elements
        vtemp = vtemp - htemp[jj] * V[:,jj] # and orthogonalize the new Krylov vector with respect to previous basis elements

    if k == 1:
        H = htemp
    else:
        H = np.hstack((H, htemp))
    htemp = np.zeros((1,k))
    htemp[:,-1] = np.linalg.norm(vtemp)
    H = np.vstack((H, htemp))
    V = np.hstack((V, vtemp.reshape((-1,1))/H[-1,-1]))
    return (V, H)

def golub_kahan_update(A, U, S, V):

    k = S.shape[0]
    utemp = U[:,-1]
    if k == 1:
        v = A.T@utemp
    else:
        v = A.T@utemp - S[k-1,k-2]*V[:,k-2]
    alpha = np.linalg.norm(v)   
    v = v/alpha
    u = A@v - alpha*utemp
    beta = np.linalg.norm(u)
    u = u/beta
    U = np.hstack((U,u.reshape((-1,1))))
    if k == 1:        
        V = v.reshape((-1,1))
    else:
        V = np.hstack((V,v.reshape((-1,1))))
    temp1 = np.zeros(k,); temp1[-1] = alpha
    temp2 = np.zeros(k,); temp2[-1] = beta
    if k == 1:
        S = np.array([temp1,temp2])
    else:
        S = np.hstack((S, temp1.reshape((-1,1))))
        S = np.vstack((S, temp2.reshape((1,-1))))
    return (U,S,V)

def gsvd(A, B):
    vm1, p = A.shape
    vm2, p = B.shape
    if not (vm1 == vm2 and vm1 >= vm2 >= p):
        raise ValueError("Invalid input dimensions. A should be of size mxp, and B should be of size nxp with m >= n >= p.")
    QA, A = np.linalg.qr(A,'reduced')
    m = p
    QB, B = np.linalg.qr(B,'reduced')
    n = p
    Q, R = np.linalg.qr(np.concatenate((A, B),axis=0),'reduced')
    U, V, Z, C, S = csd(Q[:m, :], Q[m:m+n, :])
    X = R.T @ Z
    U = QA @ U
    V = QB @ V
    return U, V, X, C, S
def csd(Q1, Q2):
    m, p = Q1.shape
    n = Q2.shape[0]
    U, C, Z = np.linalg.svd(Q1)
    C = np.diag(C)
    Z = Z.T
    q = min(m, p)
    i = np.arange(q)
    j = np.arange(q-1, -1, -1)
    C[i, i] = C[j, j]
    U[:, i] = U[:, j]
    Z[:, i] = Z[:, j]
    S = Q2 @ Z
    k = np.max(np.concatenate(([0], 1+np.where(np.diag(C) <= 1/np.sqrt(2))[0])))
    V, _ = np.linalg.qr(S[:,:k],'complete')
    S = V.T @ S
    r = min(k, m)
    S[:, :r] = diagf(S[:, :r])
    if k < min(n, p):
        r = min(n, p)
        i = np.arange(k, n)
        j = np.arange(k, r)
        UT, ST, VT = np.linalg.svd(S[np.ix_(i, j)])
        ST = np.diag(ST)
        if k > 0:
            S[:k, j] = 0
        S[np.ix_(i,j)] = ST
        C[:, j] = C[:, j] @ VT.T
        V[:, i] = V[:, i] @ UT
        Z[:, j] = Z[:, j] @ VT.T
        i = np.arange(k, q)
        Q, R = np.linalg.qr(C[np.ix_(i, j)])
        C[np.ix_(i, j)] = diagf(R)
        U[:, i] = U[:, i] @ Q
    U, C = diagp(U, C, max(0, p - m))
    C = np.real(C)
    V, S = diagp(V, S, 0)
    S = np.real(S)
    return U, V, Z, C, S
def diagk(X, k):
    if not np.isscalar(X) and not np.isscalar(k):
        m, n = X.shape
        if k >= 0:
            diag_len = min(n - k, m)
            diag = X[:diag_len, k:k+diag_len]
        else:
            diag_len = min(m + k, n)
            diag = X[-k:-k+diag_len, :diag_len]
    else:
        diag = np.diagonal(X, k)
    return diag.flatten()
def diagf(X):
    return np.triu(np.tril(X))
def diagp(Y, X, k):
    D = diagk(X, k)
    j = np.where(np.real(D) < 0) or np.where(np.imag(D) != 0)
    j = np.asarray(j, dtype=int).flatten()
    D = np.diag(np.conj(D[j]) / np.abs(D[j]))
    Y[:, np.ix_(j)] = Y[:, np.ix_(j)] @ D.T
    X[j, :] = D @ X[j, :]
    X = X + 0  # Use "+0" to set possible -0 elements to 0
    return Y, X