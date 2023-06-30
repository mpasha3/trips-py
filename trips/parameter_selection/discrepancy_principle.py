import numpy as np 
from scipy.optimize import newton, minimize
import scipy.linalg as la

from ..utils import operator_qr, operator_svd
from .gcv import gcv_numerator
import warnings

"""def discrepancy_principle(A, b, L, eta, delta, **kwargs):

    if 'tol' in kwargs:
        tol = kwargs['tol']
    else:
        tol = 10**(-12)

    # first, compute skinny QR factorizations.

    (Q_A, R_A) = operator_qr(A)


    (Q_L, R_L) = operator_qr(L)

    # function to minimize

    discrepancy_func = lambda reg_param: (gcv_numerator(reg_param, Q_A, R_A, Q_L, R_L, b)**2 - (eta*delta)**2) # left term should be squared? remove outer square?

    lambdah = minimize(method='L-BFGS-B', fun=discrepancy_func, x0=np.zeros(shape=1) + 10**(-6), bounds = [(0, None)], tol=tol)

    return lambdah"""

def discrepancy_principle(A, b, L, delta = None, eta = 1.01, **kwargs):

    if not ( isinstance(delta, float) or isinstance(delta, int)):

        raise Warning("""A value for the noise level delta was not provided. A default value of 0.01 has been used. 
                      Please supply a value of delta based on the estimated noise level of the problem.""")
    
        delta = 0.01

    U, S, V = la.svd(A, full_matrices=False)
    singular_values = S**2
    singular_values.shape = (singular_values.shape[0], 1)
    bhat = U.T @ b
    beta = 1.0

    alpha = 0.0

    iterations = 0

    while (iterations < 30) or ((iterations <= 100) and (np.abs(alpha) < 10**(-16))):

        f = ((singular_values*beta + 1)**(-2)).T @ bhat - (eta*delta)**2

        f_prime = -2*  ((singular_values*beta + 1)**(-3) * singular_values).T @ bhat

        beta_new = beta - f/f_prime


        if abs(beta_new - beta) < 10**(-7)* beta:
            break

        beta = beta_new

        alpha = 1/beta_new

        iterations += 1

    return {'x':alpha}

"""def discrepancy_principle(A, b, eta, delta, **kwargs):

    U, S, V = la.svd(A.todense(), full_matrices=False)

    singular_values = S**2

    bhat = U.T @ b

    beta = 1

    for ii in range(0,30):
    
        f = ((singular_values*beta + 1)**(-2)).T @ bhat - (eta*delta)**2

        f_prime = -2*  ((singular_values*beta + 1)**(-3) * singular_values).T @ bhat

        beta_new = beta - f/f_prime

        if abs(beta_new - beta) < 10**(-3) * beta:
        
            break
        beta = beta_new

    alpha = 1/beta

    return {'x':alpha}"""

