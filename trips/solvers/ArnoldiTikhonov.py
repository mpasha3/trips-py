#!/usr/bin/env python
"""
Definition of test problems
--------------------------------------------------------------------------
Created June 23rd, 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com"

from ..decompositions import generalized_golub_kahan, arnoldi
import numpy as np
from scipy import linalg as la
from trips.parameter_selection.gcv import *
from trips.parameter_selection.discrepancy_principle import *
from pylops import Identity


def arnoldi_tikhonov(A, b, projection_dim=3, regparam = 'gcv', **kwargs):

    if A.shape[0] != A.shape[1]:

        raise ValueError("The observation matrix A must be square for this method.")

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False

    (V,H) = arnoldi(A, b, projection_dim, dp_stop, **kwargs)

    VV = V[:, 0:-1]
    HH = H[0:-1, :]
    bhat = VV.T @ b

    L = Identity(HH.shape[0], HH.shape[1])

    if regparam == 'gcv':
        lambdah = generalized_crossvalidation(HH, bhat, L, **kwargs)['x'].item()

    elif regparam == 'dp':
        lambdah = discrepancy_principle(HH, bhat, L, **kwargs)['x'].item()

    else:
        lambdah = regparam

    L = L.todense() if isinstance(L, LinearOperator) else L

    y = la.lstsq(HH.T*HH + lambdah*L, bhat)[0]

    x = VV @ y

    return x

    

