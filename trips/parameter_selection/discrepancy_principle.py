#!/usr/bin/env python
"""
Definition of discrepancy principle for Tikhonov regularization parameter choice
--------------------------------------------------------------------------
Created June 26, 2023 for TRIPs-Py library
"""
__author__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com"

import numpy as np 
import scipy.linalg as la

from trips.utils import operator_qr, operator_svd, is_identity
import warnings

def discrepancy_principle(A, b, L, delta = None, eta = 1.01, **kwargs):

    if not ( isinstance(delta, float) or isinstance(delta, int)):

        # raise TypeError('You must provide a value for the noise level delta.')
        raise Exception("""A value for the noise level delta was not provided and the discrepancy principle cannot be applied. 
                    Please supply a value of delta based on the estimated noise level of the problem, or choose the regularization parameter according to gcv.""")

    if is_identity(L):
        Anew = A
        bnew = b
    else: ### MORE CASES TO BE CONSIDERED
        UL, SL, VL = la.svd(L)
        if L.shape[0] >= L.shape[1] and SL[-1] != 0:
            Anew = A@(VL.T@np.diag((SL)**(-1)))
            bnew = b
        elif L.shape[0] >= L.shape[1] and SL[-1] == 0:
            zeroind = np.where(SL == 0)
            #W = VL[:,zeroind]
            W = VL[zeroind,:].reshape((-1,1))
            AW = A@W
            Q_AW, R_AW = np.linalg.qr(AW, mode='reduced')
            Q_LT, R_LT = np.linalg.qr(L.T, mode='reduced')
            LAwpinv = (np.eye(L.shape[1]) - (W@np.linalg.inv(R_AW)@Q_AW.T@A))@Q_LT@np.linalg.inv(R_LT.T)
            Anew = A@LAwpinv
            xnull = W@np.linalg.inv(R_AW)@Q_AW.T@b
            bnew = b - A@xnull
        elif (L.shape[0] < L.shape[1]):
            # print(L.shape[0]-L.shape[1])
            W = VL[L.shape[0]-L.shape[1]:,:].T
            print(W.shape)
            AW = A@W
            Q_AW, R_AW = np.linalg.qr(AW, mode='reduced')
            Q_LT, R_LT = np.linalg.qr(L.T, mode='reduced')
            # print((W@np.linalg.inv(R_AW)@Q_AW.T@A).shape)
            LAwpinv = (np.eye(L.shape[1]) - (W@np.linalg.inv(R_AW)@Q_AW.T@A))@Q_LT@np.linalg.inv(R_LT.T)
            Anew = A@LAwpinv
            xnull = W@np.linalg.inv(R_AW)@Q_AW.T@b
            bnew = b - A@xnull

    U, S, V = la.svd(Anew)
    singular_values = S**2
    bhat = U.T @ bnew
    if Anew.shape[0] > Anew.shape[1]:
        print('here')
        singular_values = np.append(singular_values.reshape((-1,1)), np.zeros((Anew.shape[0]-Anew.shape[1],1)))
        testzero = la.norm(bhat[Anew.shape[1]-Anew.shape[0]:,:])**2  - (eta*delta)**2
    else:
        testzero = - (eta*delta)**2
    singular_values.shape = (singular_values.shape[0], 1)

    testzero = -1 ###
    
    beta = 1e-8
    iterations = 0

    if testzero < 0:
        while (iterations < 30) or ((iterations <= 100) and (np.abs(alpha) < 10**(-16))):
            # print(iterations)
            zbeta = (((singular_values*beta + 1)**(-1))*bhat.reshape((-1,1))).reshape((-1,1))
            f = la.norm(zbeta)**2 - (eta*delta)**2
            wbeta = (((singular_values*beta + 1)**(-1))*zbeta).reshape((-1,1))
            f_prime = 2/beta*zbeta.T@(wbeta - zbeta)

        # tikh_sol = lambda reg_param: np.linalg.lstsq(np.vstack((Anew, (1/np.sqrt(reg_param))*np.eye(Anew.shape[1]))), np.vstack((bnew.reshape((-1,1)), np.zeros((Anew.shape[1],1)))))[0]
        # discr_func_zero = lambda reg_param: (np.linalg.norm(np.matmul(Anew,tikh_sol(reg_param)).reshape((-1,1)) - (bnew.reshape((-1,1))))**2 - (eta*delta)**2)
        # tikh_sol = lambda reg_param: np.linalg.lstsq(np.vstack((A, (1/np.sqrt(reg_param))*L)), np.vstack((b.reshape((-1,1)), np.zeros((L.shape[0],1)))))[0]
        # discr_func_zero = lambda reg_param: (np.linalg.norm(np.matmul(A,tikh_sol(reg_param)).reshape((-1,1)) - (b.reshape((-1,1))))**2 - (eta*delta)**2)

        # print(f)
        # print(discr_func_zero(beta))

            beta_new = beta - f/f_prime

            if abs(beta_new - beta) < 10**(-12)* beta:
                break

            beta = beta_new
            alpha = 1/beta_new

            iterations += 1
    else:
        alpha = 0


    return alpha#{'x':alpha}

##### OLD VERSION #####

# import numpy as np 
# from scipy.optimize import newton, minimize
# import scipy.optimize as op
# import scipy.linalg as la

# from ..utils import operator_qr, operator_svd, is_identity
# from .gcv import gcv_numerator
# import warnings

# """def discrepancy_principle(A, b, L, eta, delta, **kwargs):

#     if 'tol' in kwargs:
#         tol = kwargs['tol']
#     else:
#         tol = 10**(-12)

#     # first, compute skinny QR factorizations.

#     (Q_A, R_A) = operator_qr(A)


#     (Q_L, R_L) = operator_qr(L)

#     # function to minimize

#     discrepancy_func = lambda reg_param: (gcv_numerator(reg_param, Q_A, R_A, Q_L, R_L, b)**2 - (eta*delta)**2) # left term should be squared? remove outer square?

#     lambdah = minimize(method='L-BFGS-B', fun=discrepancy_func, x0=np.zeros(shape=1) + 10**(-6), bounds = [(0, None)], tol=tol)

#     return lambdah"""

# def discrepancy_principle(A, b, L, delta = None, eta = 1.01, **kwargs):

#     if not ( isinstance(delta, float) or isinstance(delta, int)):

#         # raise TypeError('You must provide a value for the noise level delta.')
#         raise Warning("""A value for the noise level delta was not provided. A default value of 0.01 has been used. 
#                     Please supply a value of delta based on the estimated noise level of the problem.""")

#         delta = 0.01
#     valid = False ## Just to be able to call the new DP that Silvia coded
#     if is_identity(L): #and valid == True:
#         print("zero finder")
#         U, S, V = la.svd(A, full_matrices=False)
#         singular_values = S**2
#         singular_values.shape = (singular_values.shape[0], 1)
#         bhat = U.T @ b
#         beta = 1.0

#         alpha = 0.01

#         iterations = 0

#         while (iterations < 30) or ((iterations <= 100) and (np.abs(alpha) < 10**(-16))):

#             # f = ((singular_values*beta + 1)**(-2)).T @ (bhat**2) - (eta*delta)**2
#             # f_prime = -2*  ((singular_values*beta + 1)**(-3) * singular_values).T @ bhat

#             zbeta = ((singular_values*beta + 1)**(-2))*bhat
#             f = la.norm(zbeta)**2 - (eta*delta)**2
#             wbeta = ((singular_values*beta + 1)**(-2))*zbeta
#             f_prime = 2/beta*zbeta.T@(wbeta - zbeta)
        

#             beta_new = beta - f/f_prime


#             if abs(beta_new - beta) < 10**(-7)* beta:
#                 break

#             beta = beta_new

#             alpha = 1/beta_new

#             iterations += 1
#     else:
#         tikh_sol = lambda reg_param: np.linalg.lstsq(np.vstack((A, np.sqrt(reg_param)*L)), np.vstack((b.reshape((-1,1)), np.zeros((L.shape[0],1)))))[0]
#         discr_func_zero = lambda reg_param: np.linalg.norm(np.matmul(A,tikh_sol(reg_param)).reshape((-1,1)) - b.reshape((-1,1))) - (eta*delta)
#         alpha = op.fsolve(discr_func_zero, 1e-10)[0]

#     return alpha#{'x':alpha}


# """def discrepancy_principle(A, b, eta, delta, **kwargs):

#     U, S, V = la.svd(A.todense(), full_matrices=False)

#     singular_values = S**2

#     bhat = U.T @ b

#     beta = 1

#     for ii in range(0,30):
    
#         f = ((singular_values*beta + 1)**(-2)).T @ bhat - (eta*delta)**2

#         f_prime = -2*  ((singular_values*beta + 1)**(-3) * singular_values).T @ bhat

#         beta_new = beta - f/f_prime

#         if abs(beta_new - beta) < 10**(-3) * beta:
        
#             break
#         beta = beta_new

#     alpha = 1/beta

#     return {'x':alpha}"""

