"""
Functions which implement Krylov decompositions.
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import newton

from tqdm import tqdm

def arnoldi(A: 'np.ndarray[np.float]', b: 'np.ndarray[np.float]', n_iter: int, dp_stop=False, **kwargs ) -> 'Tuple[np.ndarray[np.float], np.ndarray[np.float]]':
    """
    computes the rank-n Arnoldi factorization of A, with initial guess b.

    returns Q (m x n_iter+1), an orthonormal matrix, and H (n_iter+1 x n_iter), an upper Hessenberg matrix.

    A: the matrix to be factorized.

    b: an initial guess for the first basis vector of the factorization.

    n_iter: the number of iterations over which to factorize A.

    dp_stop: whether or not to use the discrepancy principle to halt further factorization. Defaults to false.

    Calling with the minimal number of arguments:

    (Q,H) = arnoldi(A, b, n_iter)

    Calling with all the arguments necessary for discrepancy principle stopping:

    (Q,H) = arnoldi(A, b, n_iter, dp_stop=True, gk_eta=1.001, gk_delta=0.001)
    """

    eta = kwargs['gk_eta'] if ('gk_eta' in kwargs) else 1.001
    delta = kwargs['gk_delta'] if ('gk_delta' in kwargs) else 0.001

    (rows, cols) = A.shape

    # preallocate

    Q = np.zeros((rows, n_iter+1))
    H = np.zeros((n_iter+1, 1))

    # normalize b
    b = b/np.linalg.norm(b)

    # b is first basis vector
    Q[:, 0] = b.flatten()

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


def arnoldi_1(A: 'np.ndarray[np.float]', n: int, q_0: 'np.ndarray[np.float]' ) -> 'Tuple[np.ndarray[np.float], np.ndarray[np.float]]':
    """
    computes the rank-n Arnoldi factorization of A, with initial guess q_0.

    returns Q (m x n), an orthonormal matrix, and H (n+1 x n), an upper Hessenberg matrix.
    """

    # preallocate

    Q = np.zeros((A.shape[0], n+1))
    H = np.zeros((n+1, n))

    # normalize q_0
    q_0 = q_0/np.linalg.norm(q_0, ord=2)

    # q_0 is first basis vector
    Q[:, 0] = q_0[:,0]

    for ii in tqdm(range(0,n), desc = "generating basis..."): # for each iteration over the method:

        q_nplus1  = A @ Q[:,ii] # generate the next vector in the Krylov subspace

        for jj in range(0,n): # for each iteration *that has been previously completed*:

            H[jj,ii] = np.dot( Q[:,jj], q_nplus1 ) # calculate projections of the new Krylov vector onto previous basis elements

            q_nplus1 = q_nplus1 - H[jj,ii] * Q[:,jj] # and orthogonalize the new Krylov vector with respect to previous basis elements

        if ii < n:
            H[ii+1, ii] = np.linalg.norm(q_nplus1, 2)

            if H[ii+1,ii] == 0:
                return (Q,H)

            Q[:, ii+1] = q_nplus1/H[ii+1,ii]


    return (Q,H)



def generalized_golub_kahan(A, b, n_iter, dp_stop=False, **kwargs):
    """
    computes the rank-n Golub-Kahan factorization of A, with initial guess b.

    returns U (m x n_iter+1), an orthonormal matrix, V (n x n_iter), an orthonormal matrix, and B (n_iter+1 x n_iter), a bidiagonal matrix.

    A: the matrix to be factorized.

    b: an initial guess for the first basis vector of the factorization.

    n_iter: the number of iterations over which to factorize A.

    dp_stop: whether or not to use the discrepancy principle to halt further factorization. Defaults to false.

    Calling with the minimal number of arguments:

    (U,B,V) = generalized_golub_kahan(A, b, n_iter)

    Calling with all the arguments necessary for discrepancy principle stopping:

    (U,B,V) = generalized_golub_kahan(A, b, n_iter, dp_stop=True, gk_eta=1.001, gk_delta=0.001)
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
                                       
            B = np.pad(np.diag(alphas),( (0,1),(0,0) )) + np.pad(np.diag(betas), ( (1,0),(0,0) ) ) # constrct B from the bidiagonal entries
                                        
            bhat = U.T @ b
                                        
            y = np.linalg.lstsq(B, bhat, rcond=None)[0] # solve the least squares problem
            
            x = V @ y # project back
                                        
            res_norm = np.linalg.norm(A @ x - b) # and find the norm of the residual
                                       
        iterations += 1


    B = np.zeros(shape=(alphas.shape[0]+1, alphas.shape[0]) )
    B[range(0,alphas.shape[0]), range(0,alphas.shape[0])] = alphas
    B[range(1,alphas.shape[0]+1), range(0,alphas.shape[0])] = betas


    return (U,B,V)


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
