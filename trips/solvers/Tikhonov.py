#!/usr/bin/env python
"""
Definition of test problems
--------------------------------------------------------------------------
Created June 20, 2022 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha and Connor Sanderford"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com and csanderf@asu.edu; connorsanderford@gmail.com"
import numpy as np
from ..decompositions import golub_kahan, arnoldi
from ..parameter_selection.gcv import generalized_crossvalidation
from ..parameter_selection.discrepancy_principle import discrepancy_principle
from ..utils import smoothed_holder_weights
from collections.abc import Iterable
def Tikhonov(A, b, L, x_true, regparam = 'gcv', **kwargs):
    if regparam in ['gcv', 'GCV', 'Gcv']:
        lambdah = generalized_crossvalidation(A, b, L) # find ideal lambda by crossvalidation ###
        xTikh = np.linalg.solve(A.T@A + lambdah*L.T@L, A.T@b)
    elif regparam in ['DP', 'dp', 'Dp', 'Discrepancy Principle', 'Discrepancy principle', 'discrepancy principle']:
        lambdah = discrepancy_principle(A, b, L, **kwargs) # find ideal lambdas by discrepancy principle
        print(lambdah)
        xTikh = np.linalg.solve(A.T@A + lambdah*L.T@L, A.T@b)
    else:
        lambdah = regparam
        xTikh = np.linalg.solve(A.T@A + lambdah*L.T@L, A.T@b)
    return xTikh, lambdah  