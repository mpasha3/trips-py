#!/usr/bin/env python
"""
Builds function for Golub Kahan Tikhonov
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha and Silvia Gazzola"
__affiliations__ = 'MIT and Tufts University, University of Bath'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk;"

from pylops import Identity
from trips.utilities.reg_param.discrepancy_principle import *
from trips.utilities.reg_param.gcv import *
from scipy import linalg as la
from trips.utilities.decompositions import golub_kahan
import os
import sys
import numpy as np

def Golub_Kahan_Tikhonov(A, b, n_iter=3, regparam='gcv', **kwargs):
    """
    Description: Computes the Golub-Kahan-Tikhonov solution as follows:
     Step 1: A fixed number of iterations of Golub-Kahan is performed
     Step 2: On the projected subspace computed in Step 1 there is computed the best regularization parameter is chosen. 
     and then the regularized Tikhonov problem is solved.
    Input: 
        A: the matrix to be factorized.

        b: an initial guess for the first basis vector of the factorization.

        n_iter: the number of iterations over which to factorize A in the Arnoldi procedure.

        reg_param: Defines the regularization parameter. 
        The user can choose 'gcv', 'dp' or a nonnegative scalar
        Default value is set to 'gcv'.
    Output:
        x: the computed solution as a vector
        lambdah: the computed regularization parameter

    Calling with the minimal number of arguments:

        x, lambda = golub_kahan_tikhonov(A, b, n_iter = 3, regparam = 'gcv')

    Calling with all the arguments necessary for discrepancy principle stopping:

        x, lambda = golub_kahan_tikhonov(A, b, n_iter = 3, regparam = 'gcv', dp_stop=True, gk_eta=1.001, gk_delta=0.001)
    """
    delta = kwargs['delta'] if ('delta' in kwargs) else None

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False

    if (regparam == 'dp' or dp_stop != False) and delta == None:
        raise Exception("""A value for the noise level delta was not provided and the discrepancy principle cannot be applied. 
                    Please supply a value of delta based on the estimated noise level of the problem, or choose the regularization parameter according to gcv or a different stopping criterion.""")

    (U, B, V) = golub_kahan(A, b, n_iter=3, dp_stop=0)

    bhat = U.T @ b

    L = Identity(B.shape[1], B.shape[1])
    lambda_history = []
    if regparam == 'gcv':
        Q_A, R_A, _ = la.svd(B, full_matrices=False) # this is a factorization of the projected matrix
        R_A = np.diag(R_A)
        R_L = Identity(B.shape[1])
        lambdah = generalized_crossvalidation(Q_A, R_A, R_L, bhat, variant = 'modified', fullsize = A.shape[0], **kwargs)
    elif regparam == 'dp':
        lambdah = discrepancy_principle(U, B, L, b, **kwargs)
    else:
        lambdah = regparam
    L = L.todense() if isinstance(L, LinearOperator) else L 
    y = np.linalg.lstsq(np.vstack((B, np.sqrt(lambdah)*L)), np.vstack((bhat.reshape((-1,1)), np.zeros((B.shape[1],1)))), rcond=None)[0]
    x = V@y
    return (x, lambdah)