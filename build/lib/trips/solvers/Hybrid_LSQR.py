#!/usr/bin/env python
"""
Builds function for Hybrid_LSQR
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Silvia Gazzola and Mirjeta Pasha"
__affiliations__ = 'University of Bath and (MIT and Tufts University)'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "sg968@bath.ac.uk; mpasha@mit.edu; mirjeta.pasha1@gmail.com;"

from ..utilities.decompositions import golub_kahan_update
import numpy as np
from scipy import linalg as la
from trips.utilities.reg_param.gcv import *
from trips.utilities.reg_param.discrepancy_principle import *
from pylops import Identity
from trips.solvers import Tikhonov
from tqdm import tqdm
from collections.abc import Iterable

def Hybrid_LSQR(A, b, n_iter = 100, regparam = 'gcv', x_true=None, **kwargs):
    """
    Description: Hybrid version of the LSQR method

    Inputs: 

    A: the matrix of the system to be solved

    b: the available data (right-hand-side vector)

    Optional inputs:

    n_iter: the maximum number of iterations to be performed
    
    regparam: a value or a method to find the regularization parameter for the projected problems

    x_true: true solution (allows us to returns error norms with respect to x_true at each iteration)

    Outputs: 

    x: the computed solution

    info: a dictionary with the following items


    Example of calling the method: 

    (x, x_history, k) = TP_cgls(A, b, x_0, max_iter = 30, tol = 0.001)

    """
    delta = kwargs['delta'] if ('delta' in kwargs) else None

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False

    if (regparam == 'dp' or dp_stop != False) and delta == None:
        raise Exception("""A value for the noise level delta was not provided and the discrepancy principle cannot be applied. 
                    Please supply a value of delta based on the estimated noise level of the problem, or choose the regularization parameter according to gcv.""")
    
    n = A.shape[1]
    beta = np.linalg.norm(b)
    U = b.reshape((-1,1))/beta
    B = np.empty(1)
    V = np.empty((n,1))
    x_history = []
    lambda_history = []
    residual_history = []
    bhat = np.zeros(1,); bhat[0] = beta

    for ii in tqdm(range(n_iter), 'running Golub-Kahan bidiagonalization algorithm...'):
        (U, B, V) = golub_kahan_update(A, U, B, V)
        bhat = np.append(bhat, 0)
        L = Identity(B.shape[1], B.shape[1])
        if ii == 0:
            lambdah = 0
        else:
            if regparam == 'gcv':
                Q_A, R_A, _ = la.svd(B, full_matrices=False) # this is a factorization of the projected matrix
                R_A = np.diag(R_A)
                R_L = Identity(B.shape[1])
                lambdah = generalized_crossvalidation(Q_A, R_A, R_L, bhat, variant = 'modified', fullsize = A.shape[0], **kwargs)
            elif regparam == 'dp':
                lambdah = discrepancy_principle(U, B, L, b, **kwargs)
                if (dp_stop==True):
                    print('discrepancy principle satisfied, stopping early.')
                    lambda_history.append(lambdah)
                    L = L.todense() if isinstance(L, LinearOperator) else L # I am actually defining it, so check what is the case
                    y = np.linalg.lstsq(np.vstack((B, np.sqrt(lambdah)*L)), np.vstack((bhat.reshape((-1,1)), np.zeros((B.shape[1],1)))), rcond=None)[0]
                    x = V[:,:-1] @ y
                    break
            else:
                lambdah = regparam
            lambda_history.append(lambdah)    
            L = L.todense() if isinstance(L, LinearOperator) else L
            # y = Tikhonov(B, bhat, L, regparam = lambdah)[0]
            y = np.linalg.lstsq(np.vstack((B, np.sqrt(lambdah)*L)), np.vstack((bhat.reshape((-1,1)), np.zeros((B.shape[1],1)))), rcond=None)[0]
            x = V @ y
            x = x.reshape((-1,1))
            x_history.append(x)
        if (x_true is not None):
            x_true_norm = la.norm(x_true.reshape(-1,1))
            rre_history = [la.norm(x - x_true.reshape(-1,1))/x_true_norm for x in x_history]
            info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relError': rre_history, 'relResidual': residual_history, 'its': ii}
        else:
            info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relResidual': residual_history, 'its': ii}
    return (x, info)

    

