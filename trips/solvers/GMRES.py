#!/usr/bin/env python
"""
Definition of test problems
--------------------------------------------------------------------------
Created December 10, 2022 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com"

from select import select
from ..decompositions import arnoldi
import numpy as np
from scipy import linalg as la

from tqdm import tqdm
"""
Functions which implement variants of GKS.
"""
def GMRES(A, b, projection_dim=3, dp_stop = 0, **kwargs):
    if A.shape[0] == A.shape[1]:
        b_vec = b.reshape((-1,1))
        (V,H) = arnoldi(A, projection_dim, b_vec, **kwargs)
        UU = V[:, 0:-1]
        HH = H[0:-1, :]
        bhat = UU.T.dot(b_vec)
        y = np.linalg.solve(HH.T*HH, bhat)
        x_GMRESS = UU.dot(y)
    return (x_GMRESS)