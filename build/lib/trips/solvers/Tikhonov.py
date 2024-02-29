#!/usr/bin/env python
"""
Function that solves the Tikhonov problem.
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola"
__affiliations__ = 'MIT and Tufts University, University of Bath'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk;"

import numpy as np
from pylops import Identity
from trips.utilities.reg_param.gcv import *
from trips.utilities.reg_param.discrepancy_principle import *
from collections.abc import Iterable
def Tikhonov(A, b, L, x_true, regparam = 'gcv', **kwargs):
    A = A.todense() if isinstance(A, LinearOperator) else A
    L = L.todense() if isinstance(L, LinearOperator) else L
    if regparam in ['gcv', 'GCV', 'Gcv']:
        lambdah = generalized_crossvalidation(Identity(A.shape[0]), A, L, b) #, variant = 'modified', fullsize = A.shape[0], **kwargs)
    elif regparam in ['DP', 'dp', 'Dp', 'Discrepancy Principle', 'Discrepancy principle', 'discrepancy principle']:
        lambdah = discrepancy_principle(Identity(A.shape[0]), A, L, b, **kwargs) # find ideal lambdas by discrepancy principle
    else:
        lambdah = regparam
    xTikh = np.linalg.solve(A.T@A + lambdah*L.T@L, A.T@b)
    return xTikh, lambdah