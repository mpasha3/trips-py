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
__maintainer__ = "Mirjeta Pasha and Connor Sanderford"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com and csanderf@asu.edu; connorsanderford@gmail.com"

from select import select
from ..utilities.decompositions import generalized_golub_kahan
import numpy as np
from scipy import linalg as la

from tqdm import tqdm
def LSQR(A, b, projection_dim=3, dp_stop = 0, **kwargs):
    if A.shape[0] == A.shape[1]:
        b_vec = b.reshape((-1,1))
        (U, B, V) = generalized_golub_kahan(A, b_vec, n_iter = 10, projection_dim = projection_dim)
        VV = V
        UU = U[:, 0:-1]
        HH = B[0:-1, :]
        bhat = UU.T.dot(b_vec)
        y = np.linalg.solve(HH.T*HH, bhat)
        x_GKahan = VV.dot(y)
    return (x_GKahan)