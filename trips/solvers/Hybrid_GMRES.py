#!/usr/bin/env python
"""
Builds function for Hybrid_GMRES
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Silvia Gazzola and Mirjeta Pasha"
__affiliations__ = 'University of Bath and (MIT and Tufts University)'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "sg968@bath.ac.uk; mpasha@mit.edu; mirjeta.pasha1@gmail.com;"

from ..utilities.decompositions import arnoldi_update
import numpy as np
from scipy import linalg as la
from trips.utilities.reg_param.gcv import *
from trips.utilities.reg_param.discrepancy_principle import *
from pylops import Identity
from tqdm import tqdm

def Hybrid_GMRES(A, b, n_iter, regparam = 'gcv', x_true=None, **kwargs): # what's the naming convention here?

    delta = kwargs['delta'] if ('delta' in kwargs) else None

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False

    if (regparam == 'dp' or dp_stop != False) and delta == None:
        raise Exception("""A value for the noise level delta was not provided and the discrepancy principle cannot be applied. 
                    Please supply a value of delta based on the estimated noise level of the problem, or choose the regularization parameter according to gcv or a different stopping criterion.""")
    
    n = A.shape[1]
    if A.shape[0] != n:
        raise Exception("Please check the size of the matrx A: it should be square in order to apply hybrid GMRES") 
    
    x_history = []
    lambda_history = []
    residual_history = []

    beta = np.linalg.norm(b)
    V = b.reshape((-1,1))/beta
    H = np.empty(1)
    bhat = np.zeros(1,); bhat[0] = beta
    
    for ii in tqdm(range(n_iter), 'running Arnoldi algorithm...'):
        (V, H) = arnoldi_update(A, V, H)
        bhat = np.append(bhat, 0)
        L = Identity(H.shape[1], H.shape[1])
        if ii == 0:
            lambdah = 0
        else:
            if regparam == 'gcv':
                Q_A, R_A, _ = la.svd(H, full_matrices=False)
                R_A = np.diag(R_A)
                R_L = Identity(H.shape[1])
                lambdah = generalized_crossvalidation(Q_A, R_A, R_L, bhat, **kwargs)
            elif regparam == 'dp':
                lambdah = discrepancy_principle(V, H, L, b, **kwargs)
                if (dp_stop==True):
                    print('discrepancy principle satisfied, stopping early.')
                    lambda_history.append(lambdah)
                    L = L.todense() if isinstance(L, LinearOperator) else L # I am actually defining it, so check what is the case
                    y = np.linalg.lstsq(np.vstack((H, np.sqrt(lambdah)*L)), np.vstack((bhat.reshape((-1,1)), np.zeros((H.shape[1],1)))), rcond=None)[0]
                    x = V[:,:-1] @ y
                    break
            else:
                lambdah = regparam
        lambda_history.append(lambdah)
        L = L.todense() if isinstance(L, LinearOperator) else L
        y = np.linalg.lstsq(np.vstack((H, np.sqrt(lambdah)*L)), np.vstack((bhat.reshape((-1,1)), np.zeros((H.shape[1],1)))), rcond=None)[0]
        x = V[:,:-1] @ y
        x = x.reshape((-1,1))
        x_history.append(x) # check the new shape of this
        residual_history.append(la.norm(bhat - H@y))
        if (x_true is not None):
            x_true_norm = la.norm(x_true.reshape(-1,1))
            rre_history = [la.norm(x - x_true.reshape(-1,1))/x_true_norm for x in x_history]
            info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relError': rre_history, 'relResidual': residual_history, 'its': ii}
        else:
            info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relResidual': residual_history, 'its': ii}
    return (x, info)

    

