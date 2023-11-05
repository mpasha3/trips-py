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

import numpy as np 
from scipy.optimize import newton, minimize
import scipy.linalg as la
import scipy.optimize as op

from pylops import Identity, LinearOperator

from ..utils import operator_qr, operator_svd, is_identity

#separate into two modules

"""
Generalized crossvalidation
"""

def gcv_numerator(reg_param, Q_A, R_A, Q_L, R_L, b):

    # the observation term:

    R_A_2 = R_A.T @ R_A

    R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

    # The regularizer term:

    R_L_2 = (R_L.T @ R_L)
    
    R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2

    # the inverse term:

    inverted = la.solve( ( R_A_2 + reg_param * R_L_2), (R_A.T @ Q_A.T @ b) )

    # times R_A, minus b, norm
    return np.sqrt((np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2 + np.linalg.norm(b - Q_A@(Q_A.T@b))**2)
    #return np.linalg.norm( R_A @ inverted - Q_A.T @ b )

def gcv_denominator(reg_param, Q_A, R_A, Q_L, R_L, b):

    # the observation term:

    R_A_2 = R_A.T @ R_A

    R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

    # The regularizer term:

    R_L_2 = (R_L.T @ R_L)

    R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2

    # the inverse term:

    inverted = la.solve( ( R_A_2 + reg_param * R_L_2), R_A.T@Q_A.T )

    # trace term

    times_RA = Q_A @ R_A @ inverted ###

    Id = np.eye(times_RA.shape[0])

    trace_term = np.trace(Id - times_RA)

    return trace_term**2


def generalized_crossvalidation(A, b, L, **kwargs):

    if 'tol' in kwargs:
        tol = kwargs['tol']
    else:
        tol = 10**(-12)

    # first, compute skinny factorizations.

    if is_identity(L):

        Q_A, R_A, _ = operator_svd(A)

        Q_L = Identity(L.shape[0])
        R_L = Identity(L.shape[0])

        R_A = np.diag(R_A)

    else:

        (Q_A, R_A) = operator_qr(A)

        (Q_L, R_L) = operator_qr(L)

    # function to minimize
    gcv_func = lambda reg_param: gcv_numerator(reg_param, Q_A, R_A, Q_L, R_L, b) / gcv_denominator(reg_param, Q_A, R_A, Q_L, R_L, b)
    # lambdah = minimize(method='L-BFGS-B'minimize(method='L-BFGS-B', fun=gcv_func, x0=np.zeros(shape=1) + 0.00001, bounds = [(0, None)], tol=tol)
    # lambdah = op.fmin(func=gcv_func, x0= np.zeros(shape=1) + 0.00001, args=(), xtol=0.000000001, ftol=0.000000001, maxiter=1000)
    lambdah = op.fminbound(func = gcv_func, x1 = 1e-05, x2 = 1e2, args=(), xtol=1e-12, maxfun=1000, full_output=0, disp=1)
    #print(lambdah)
    return lambdah