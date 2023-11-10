#!/usr/bin/env python
"""
Definition of test problems
--------------------------------------------------------------------------
Created August 30th, 2023 for TRIPs-Py library
"""
__authors__ = "Silvia Gazzola, Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Silvia Gazzola, Mirjeta Pasha and Connor Sanderford"
__email__ = "S.Gazzola@bath.ac.uk; mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com and csanderf@asu.edu; connorsanderford@gmail.com"

from ..decompositions import arnoldi_update
import numpy as np
from scipy import linalg as la
from trips.parameter_selection.gcv import *
from trips.parameter_selection.discrepancy_principle import *
from pylops import Identity

def hybrid_gmres(A, b, n_iter, regparam = 'gcv', x_true=None, **kwargs): # what's the naming convention here?

    delta = kwargs['delta'] if ('delta' in kwargs) else None

    eta = kwargs['eta'] if ('eta' in kwargs) else 1.01

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False

    if (regparam == 'dp' or dp_stop != False) and delta == None:
        raise Exception("""A value for the noise level delta was not provided and the discrepancy principle cannot be applied. 
                    Please supply a value of delta based on the estimated noise level of the problem, or choose the regularization parameter according to gcv or a different stopping criterion.""")

    n = A.shape[1]
    beta = np.linalg.norm(b)
    V = b.reshape((-1,1))/beta
    H = np.empty(1)
    RegParam = np.zeros(n_iter,)
    x_history = []
    lambda_history = []

    for ii in range(n_iter):
        print(ii)
        (V, H) = arnoldi_update(A, V, H)
        bhat = np.zeros(ii+2,); bhat[0] = beta ###
        L = Identity(H.shape[1], H.shape[1])
        y = la.lstsq(H,bhat)[0]
        nrmr = np.linalg.norm(bhat - H@y)
        # print(H.shape)
        if ii == 0:
            lambdah = 0
        else:
            if regparam == 'gcv':
                #lambdah = generalized_crossvalidation(B, bhat, L, **kwargs)['x'].item()
                lambdah = generalized_crossvalidation(H, bhat, L)
            elif regparam == 'dp':
                if nrmr <= eta*delta:
                    lambdah = discrepancy_principle(H, bhat, L, **kwargs)
                    if (dp_stop==True):
                        print('discrepancy principle satisfied, stopping early.')
                        RegParam[ii] = lambdah
                        L = L.todense() if isinstance(L, LinearOperator) else L
                        y = np.linalg.lstsq(np.vstack((H, np.sqrt(lambdah)*L)), np.vstack((bhat.reshape((-1,1)), np.zeros((H.shape[1],1)))))[0]
                        x = V[:,:-1] @ y
                        break
                else:
                    lambdah = 0
            else:
                lambdah = regparam
            # RegParam[ii] = lambdah
            lambda_history.append(lambdah)
            L = L.todense() if isinstance(L, LinearOperator) else L
            #y = la.lstsq(np.matmul(H.T,H) + lambdah*np.matmul(L.T,L), np.matmul(H.T,bhat))[0] ## SG: potentially needs L.T*L; also why lstsq? May just need a solver for linear system; also: you may need np.matmul; also: these are not the normal eqs (you need B'*b)
            y = np.linalg.lstsq(np.vstack((H, np.sqrt(lambdah)*L)), np.vstack((bhat.reshape((-1,1)), np.zeros((H.shape[1],1)))))[0]
            x = V[:,:-1] @ y
            x_history.append(x)
        residual_history = [A@x - b for x in x_history]
        if x_true is not None:
            x_true_norm = la.norm(x_true)
            rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
            info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relError': rre_history, 'relResidual': residual_history, 'its': ii}
        else:
            info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relResidual': residual_history, 'its': ii}
    return (x, info)
    # return x, V, H, RegParam

    

