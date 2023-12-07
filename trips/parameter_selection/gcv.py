#!/usr/bin/env python
"""
Builds functions for generalized cross validation
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, Connor Sanderford, and Ugochukwu Obinna Ugwu"
__affiliations__ = 'Tufts University, University of Bath, Arizona State University, and Tufts University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu; connorsanderford@gmail.com; Ugochukwu.Ugwu@tufts.edu"

import numpy as np 
from scipy.optimize import newton, minimize
import scipy.linalg as la
import scipy.optimize as op
from pylops import Identity, LinearOperator
from ..utilities.utils import operator_qr, operator_svd, is_identity

"""
Generalized crossvalidation
"""

def gcv_numerator(reg_param, Q_A, R_A, R_L, b):

    # the observation term:

    R_A_2 = R_A.T @ R_A

    R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

    # The regularizer term:

    R_L_2 = (R_L.T @ R_L)
    
    R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2

    # the inverse term:

    inverted = la.solve( ( R_A_2 + reg_param * R_L_2), (R_A.T @ Q_A.T @ b) )

    return np.sqrt((np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2 + np.linalg.norm(b - Q_A@(Q_A.T@b))**2)

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
       m = kwargs['fullsize']
       trace_term = (m - R_A.shape[1]) - np.trace(R_A @ inverted) # b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities 
    else:
        # in this way works even if we revert to the fully projected pb (call with Q_A.T@b)
        trace_term = b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities
    
    return trace_term**2

def generalized_crossvalidation(Q_A, R_A, R_L, b, **kwargs):

    if 'tol' in kwargs:
        tol = kwargs['tol']
    else:
        tol = 10**(-12)

    # function to minimize
    gcv_func = lambda reg_param: gcv_numerator(reg_param, Q_A, R_A, R_L, b) / gcv_denominator(reg_param, R_A, R_L, b, **kwargs)
    lambdah = op.fminbound(func = gcv_func, x1 = 1e-09, x2 = 1e2, args=(), xtol=1e-12, maxfun=1000, full_output=0, disp=0)
    return lambdah