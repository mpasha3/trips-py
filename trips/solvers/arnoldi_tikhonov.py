#!/usr/bin/env python
"""
Definition of test problems
--------------------------------------------------------------------------
Created June 23rd, 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha and Connor Sanderford"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com and csanderf@asu.edu; connorsanderford@gmail.com"
import os, sys
import numpy as np
sys.path.insert(0, '/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package')

from ..decompositions import arnoldi
import numpy as np
from scipy import linalg as la
from trips.parameter_selection.gcv import *
from trips.parameter_selection.discrepancy_principle import *
from pylops import Identity

def arnoldi_tikhonov(A, b, n_iter = 3, regparam = 'gcv', **kwargs):
    """
    Description: Computes the Arnoldi-Tikhonov solution as follows:
     Step 1: A fixed number of iterations of Arnoldi is performed
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

        x, lambda = arnoldi_tikhonov(A, b, n_iter = 3, regparam = 'gcv')

    Calling with all the arguments necessary for discrepancy principle stopping:

        x, lambda = arnoldi_tikhonov(A, b, n_iter = 3, regparam = 'gcv', dp_stop=True, gk_eta=1.001, gk_delta=0.001)
    """
    if ('shape' in kwargs):
        shape = kwargs['shape'] 
    elif type(A) == 'function' and shape not in kwargs['shape']:
        raise ValueError("The observation matrix A is a function. The shape must be given as an input")
    else:
        sx = A.shape[0]
        sy = A.shape[0]
        shape = [sx, sy]
        
    projection_method = kwargs['projection_method'] if ('projection_method' in kwargs) else 'auto'

    regparam_sequence = kwargs['regparam_sequence'] if ('regparam_sequence' in kwargs) else [0.1*(0.5**(x)) for x in range(0,1)]

    if shape[0] != shape[1]:

        raise ValueError("The observation matrix A must be square for this method.")

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False


    (Vdp1,H) = arnoldi(A, b, n_iter, dp_stop, **kwargs)

    
    Vd = Vdp1[:, 0:-1]
    bhat = Vdp1.T @ b

    L = Identity(H.shape[1], H.shape[1])

    delta = kwargs['delta'] if ('delta' in kwargs) else None
    eta = kwargs['eta'] if ('eta' in kwargs) else 1.01

    if regparam == 'gcv':
        lambdah = generalized_crossvalidation(H, bhat, L, **kwargs)

    elif regparam == 'dp':
        y = np.linalg.lstsq(H, bhat.reshape((-1,1)))[0]
        nrmr = np.linalg.norm(H@y - bhat)
        if nrmr <= eta*delta:
            lambdah = discrepancy_principle(H, bhat, L, **kwargs)#['x'].item()
        else:
            lambdah = 0
    else:
        lambdah = regparam

    L = L.todense() if isinstance(L, LinearOperator) else L

    y = la.solve(H.T@H + lambdah*L.T@L, H.T@bhat) #solve performs as backslash in Matlab

    x = Vd @ y
    return x, lambdah



