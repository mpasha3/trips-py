#!/usr/bin/env python
"""
Definition of test problems
--------------------------------------------------------------------------
Created Jun 7, 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com"

from select import select
from ..decompositions import generalized_golub_kahan, arnoldi
import numpy as np
from scipy import linalg as la
from trips.parameter_selection.gcv import *
from trips.parameter_selection.discrepancy_principle import *
def Arnoldi_Tikhonov(A, b, L = 'Identity', projection_dim=3, iter=50, dp_stop = 0, param_choice = 'manual', automatic_param = 'gcv', reg_param = 1, delta = 0, **kwargs):
    if A.shape[0] == A.shape[1]:
        b_vec = b.reshape((-1,1))
        (V,H) = arnoldi(A, b_vec, projection_dim, dp_stop)
        UU = V[:, 0:-1]
        HH = H[0:-1, :]
        bhat = UU.T.dot(b_vec)
        if param_choice == 'manual':
            y = np.linalg.solve(HH.T@HH + reg_param*(Identity(HH.shape[1]).todense()), bhat)
            x_AT = UU.dot(y)
        else:
            if automatic_param == 'gcv':
                L = np.identity(HH.shape[1], dtype='float32')
                reg_param = generalized_crossvalidation(HH, bhat, L, **kwargs)['x'] # find ideal lambda by crossvalidation
            else:
                reg_param = discrepancy_principle(HH, bhat, L, eta = 1.01, noise_norm = delta, **kwargs)['x'][0] # find ideal lambdas by crossvalidation
                L = np.identity(HH.shape[1], dtype='float32')
            y = np.linalg.solve(HH.T*HH + reg_param[0]*L, bhat)
            x_AT = UU.dot(y)
    else:
        raise Warning("The matrix is not square. Arnoldi can not be applied")
    return (x_AT)