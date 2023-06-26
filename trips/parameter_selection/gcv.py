import numpy as np 
from scipy.optimize import newton, minimize
import scipy.linalg as la

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

    return np.linalg.norm( R_A @ inverted - Q_A.T @ b )

def gcv_denominator(reg_param, Q_A, R_A, Q_L, R_L, b):

    # the observation term:

    R_A_2 = R_A.T @ R_A

    R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

    # The regularizer term:

    R_L_2 = (R_L.T @ R_L)

    R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2

    # the inverse term:

    inverted = la.solve( ( R_A_2 + reg_param * R_L_2), R_A.T )

    # trace term

    times_RA = R_A @ inverted

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

    lambdah = minimize(method='L-BFGS-B', fun=gcv_func, x0=np.zeros(shape=1) + 0.00001, bounds = [(0, None)], tol=tol)

    return lambdah