#!/usr/bin/env python
"""
Definition of test problems
--------------------------------------------------------------------------
Created June 23rd, 2023 for TRIPs-Py library
"""
from pylops import Identity
from trips.parameter_selection.discrepancy_principle import *
from trips.parameter_selection.gcv import *
from scipy import linalg as la
from ..decompositions import golub_kahan
__authors__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha and Connor Sanderford"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com and csanderf@asu.edu; connorsanderford@gmail.com"
import os
import sys
import numpy as np
sys.path.insert(
    0, '/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package')


def golub_kahan_tikhonov(A, b, n_iter=3, regparam='gcv', **kwargs):
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

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False

    (U, B, V) = golub_kahan(A, b, n_iter=3, dp_stop=0)

    bhat = U.T @ b

    L = Identity(B.shape[1], B.shape[1])

    if regparam == 'gcv':
        lambdah = generalized_crossvalidation(B, bhat, L, **kwargs)

    elif regparam == 'dp':
        lambdah = discrepancy_principle(B, bhat, L, **kwargs)['x'].item()

    else:
        lambdah = regparam

    L = L.todense() if isinstance(L, LinearOperator) else L

    # solve performs as backslash in Matlab
    y = la.solve(B.T@B + 0.1*L.T@L, B.T@bhat)

    x_golub_kahan = V @ y

    return x_golub_kahan, lambdah
