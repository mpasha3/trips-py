#!/usr/bin/env python
"""
Builds functions for generalized cross validation
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, Connor Sanderford"
__affiliations__ = 'MIT and Tufts University, University of Bath, Arizona State University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu; connorsanderford@gmail.com;"

import numpy as np 
from scipy.optimize import newton, minimize
import scipy.linalg as la
import scipy.optimize as op
from pylops import Identity, LinearOperator
from ..utilities.utils import operator_qr, operator_svd, is_identity

"""
Generalized crossvalidation
"""

def gcv_numerator(reg_param, Q_A, R_A, R_L, b, **kwargs):

    variant = kwargs['variant'] if ('variant' in kwargs) else 'standard'

    # the observation term:

    R_A_2 = R_A.T @ R_A

    R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

    # The regularizer term:

    R_L_2 = (R_L.T @ R_L)
    
    R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2

    # the inverse term:

    inverted = la.solve( ( R_A_2 + reg_param * R_L_2), (R_A.T @ Q_A.T @ b) )

    # return np.sqrt((np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2 + np.linalg.norm(b - Q_A@(Q_A.T@b))**2)
    if variant == 'modified':
        return ((np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2 + np.linalg.norm(b - Q_A@(Q_A.T@b))**2)
    else:
        return (np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2

def gcv_denominator(reg_param, R_A, R_L, b, **kwargs):

    variant = kwargs['variant'] if ('variant' in kwargs) else 'standard'
    # print(variant)
    # the observation term:

    R_A_2 = R_A.T @ R_A

    R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

    # The regularizer term:

    R_L_2 = (R_L.T @ R_L)

    R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2

    inverted = la.solve( ( R_A_2 + reg_param * R_L_2), R_A.T )

    if variant == 'modified':
       m = kwargs['fullsize'] # probably this can be b.size -- NOT FOR HYBRID SOLVERS!
       # trace_term = (m - R_A.shape[1]) - np.trace(R_A @ inverted) # b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities 
       trace_term = m - np.trace(R_A @ inverted) # b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities 
    else:
        # in this way works even if we revert to the fully projected pb (call with Q_A.T@b)
        # trace_term = b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities
        trace_term = R_A.shape[1] - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities
    
    return trace_term**2

def generalized_crossvalidation(Q_A, R_A, R_L, b, **kwargs):

    if 'tol' in kwargs:
        tol = kwargs['tol']
    else:
        tol = 10**(-12)

    if 'gcvtype' in kwargs:
        gcvtype = kwargs['gcvtype']
    else:
        gcvtype = 'tikhonov'
    
    # function to minimize
    # if gcvtype == 'tikhonov':    
    gcv_func = lambda reg_param: gcv_numerator(reg_param, Q_A, R_A, R_L, b) / gcv_denominator(reg_param, R_A, R_L, b, **kwargs)
    lambdah = op.fminbound(func = gcv_func, x1 = 1e-09, x2 = 1e2, args=(), xtol=1e-12, maxfun=1000, full_output=0, disp=0) ## should there be tol here?
    # elif gcvtype == 'tsvd':
    #     m = Q_A.shape[0]
    #     n = R_L.shape[1]
    #     gcv_vals = []
    #     bhat = Q_A.T@b
    #     f = np.ones((m,1))
    #     for i in range(n):
    #         f[n-(i+1),] = 0
    #         fvar = np.concatenate((1 - f[:n,], f[n:,]))
    #         coeff = (fvar*bhat)**2
    #         gcv_numerator = np.sum(coeff)
    #         gcv_denominator = (m - (n-(i+1)))**2
    #         gcv_vals.append(gcv_numerator/gcv_denominator)   
    #     lambdah = n - (gcv_vals.index(min(gcv_vals))+1)
    # elif gcvtype == 'tgsvd':
    #     m = Q_A.shape[0]
    #     n = R_L.shape[1]
    #     p = R_L.shape[0]
    #     gcv_vals = []
    #     bhat = Q_A.T@b
    #     f = np.ones((m,1))
    #     for i in range(n):
    #         f[i,] = 0
    #         fvar = np.concatenate((1 - f[:n,], f[n:,]))
    #         coeff = (fvar*bhat)**2
    #         gcv_numerator = np.sum(coeff)
    #         gcv_denominator = (m - (n-(i+1)) - (n-p))**2
    #         gcv_vals.append(gcv_numerator/gcv_denominator)   
    #     lambdah = gcv_vals.index(min(gcv_vals))
    
    return lambdah