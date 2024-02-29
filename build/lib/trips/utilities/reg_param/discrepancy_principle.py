#!/usr/bin/env python
"""
Definition of functions for Discrepancy principle
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, and Connor Sanderford"
__affiliations__ = 'MIT and Tufts University, University of Bath, Arizona State University,'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu; connorsanderford@gmail.com;"

import numpy as np 
import scipy.linalg as la
from trips.utilities.utils import operator_qr, operator_svd, is_identity
import warnings

def discrepancy_principle(Q, A, L, b, delta = None, eta = 1.01, **kwargs):

    if not ( isinstance(delta, float) or isinstance(delta, int)):
        raise Exception("""A value for the noise level delta was not provided and the discrepancy principle cannot be applied. 
                    Please supply a value of delta based on the estimated noise level of the problem, or choose the regularization parameter according to gcv.""")
    
    explicitProj = kwargs['explicitProj'] if ('explicitProj' in kwargs) else False

    if 'dptype' in kwargs:
        dptype = kwargs['dptype']
    else:
        dptype = 'tikhonov'

    if dptype == 'tikhonov':  
        bfull = b
        b = Q.T@b
        if is_identity(L):
            Anew = A
            bnew = b
        else:
            UL, SL, VL = la.svd(L)
            if L.shape[0] >= L.shape[1] and SL[-1] != 0:
                Anew = A@(VL.T@np.diag((SL)**(-1)))
                bnew = b
            elif L.shape[0] >= L.shape[1] and SL[-1] == 0:
                zeroind = np.where(SL == 0)
                W = VL[zeroind,:].reshape((-1,1))
                AW = A@W
                Q_AW, R_AW = np.linalg.qr(AW, mode='reduced')
                Q_LT, R_LT = np.linalg.qr(L.T, mode='reduced')
                LAwpinv = (np.eye(L.shape[1]) - (W@np.linalg.inv(R_AW)@Q_AW.T@A))@Q_LT@np.linalg.inv(R_LT.T)
                Anew = A@LAwpinv
                xnull = W@np.linalg.inv(R_AW)@Q_AW.T@b
                bnew = b - A@xnull
            elif (L.shape[0] < L.shape[1]):
                W = VL[L.shape[0]-L.shape[1]:,:].T
                AW = A@W
                Q_AW, R_AW = np.linalg.qr(AW, mode='reduced')
                Q_LT, R_LT = np.linalg.qr(L.T, mode='reduced')
                LAwpinv = (np.eye(L.shape[1]) - (W@np.linalg.inv(R_AW)@Q_AW.T@A))@Q_LT@np.linalg.inv(R_LT.T)
                Anew = A@LAwpinv
                xnull = W@np.linalg.inv(R_AW)@Q_AW.T@b
                bnew = b - A@xnull

        U, S, V = la.svd(Anew)
        singular_values = S**2
        bhat = U.T @ bnew
        if Anew.shape[0] > Anew.shape[1]:
            singular_values = np.append(singular_values.reshape((-1,1)), np.zeros((Anew.shape[0]-Anew.shape[1],1)))
            if explicitProj:
                testzero = la.norm(bhat[Anew.shape[1]-Anew.shape[0]:,:])**2 + la.norm(bfull - Q@b)**2 - (eta*delta)**2 # this is OK but need reorthogonalization
            else:
                testzero = la.norm(bhat[Anew.shape[1]-Anew.shape[0]:,:])**2 - (eta*delta)**2
        else:
            testzero = la.norm(bfull - Q@b)**2 - (eta*delta)**2
        singular_values.shape = (singular_values.shape[0], 1)
    
        beta = 1e-8
        iterations = 0

        if testzero < 0:
            while (iterations < 30) or ((iterations <= 100) and (np.abs(alpha) < 10**(-16))):
                zbeta = (((singular_values*beta + 1)**(-1))*bhat.reshape((-1,1))).reshape((-1,1))
                if explicitProj:
                    f = la.norm(zbeta)**2 + la.norm(bfull - Q@b)**2 - (eta*delta)**2 # this is OK but need reorthogonalization
                else:
                    f = la.norm(zbeta)**2 - (eta*delta)**2
                wbeta = (((singular_values*beta + 1)**(-1))*zbeta).reshape((-1,1))
                f_prime = 2/beta*zbeta.T@(wbeta - zbeta)

                beta_new = beta - f/f_prime

                if abs(beta_new - beta) < 10**(-12)* beta:
                    break

                beta = beta_new
                alpha = 1/beta_new[0,0]

                iterations += 1
        else:
            alpha = 0
    elif dptype == 'tsvd':
        m = Q.shape[0]
        n = L.shape[1]
        f = np.ones((m,1))
        bhat = Q.T@b
        alpha = n
        for i in range(n):
            f[n-(i+1),] = 0
            fvar = np.concatenate((1 - f[:n,], f[n:,]))
            coeff = (fvar*bhat)**2
            dp_val = np.sum(coeff) - (eta*delta)**2
            if dp_val < 0:
                alpha = n - (i+1)
            else:
                break
    elif dptype == 'tgsvd':
        m = Q.shape[0]
        n = L.shape[1]
        f = np.ones((m,1))
        bhat = Q.T@b
        alpha = n
        coeff = np.square(bhat)
        for i in range(n):
            coeff[n-(i+1),] = 0
            dp_val = np.sum(coeff) - (eta*delta)**2
            if dp_val >= 0:
                alpha = i
            else:
                break

    return alpha