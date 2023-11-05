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

def hybrid_gmres(A, b, n_iter, regparam = 'gcv', **kwargs): # what's the naming convention here?

    n = A.shape[1]
    beta = np.linalg.norm(b)
    V = b.reshape((-1,1))/beta
    H = np.empty(1)
    RegParam = np.zeros(n_iter,)

    for ii in range(n_iter):
        print(ii)
        (V, H) = arnoldi_update(A, V, H)
        bhat = np.zeros(ii+2,); bhat[0] = beta ###
        L = Identity(H.shape[1], H.shape[1])
        # print(H.shape)
        if ii == 0:
            lambdah = 0
        else:
            if regparam == 'gcv':
                #lambdah = generalized_crossvalidation(B, bhat, L, **kwargs)['x'].item()
                lambdah = generalized_crossvalidation(H, bhat, L)
            elif regparam == 'dp':
                lambdah = discrepancy_principle(H, bhat, L, **kwargs)
            else:
                lambdah = regparam
            RegParam[ii] = lambdah
            L = L.todense() if isinstance(L, LinearOperator) else L
            #y = la.lstsq(np.matmul(H.T,H) + lambdah*np.matmul(L.T,L), np.matmul(H.T,bhat))[0] ## SG: potentially needs L.T*L; also why lstsq? May just need a solver for linear system; also: you may need np.matmul; also: these are not the normal eqs (you need B'*b)
            y = np.linalg.lstsq(np.vstack((H, np.sqrt(lambdah)*L)), np.vstack((bhat.reshape((-1,1)), np.zeros((H.shape[1],1)))))[0]
            x = V[:,:-1] @ y
    return x, V, H, RegParam

    

