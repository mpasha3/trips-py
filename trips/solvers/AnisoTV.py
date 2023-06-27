#!/usr/bin/env python
"""
Definition of test problems

--------------------------------------------------------------------------
Created December 10, 2022 for TRIPs-Py library
"""
__author__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com"
from trips.io import *
from trips.operators import *
from matplotlib import transforms
from scipy import ndimage
from trips.solvers.gks import *

def AnisoTV(A, b, AA, B, nx, ny, nt, dynamic, iters, reg_param, testproblem):
    """
    Reconstruct images by Anisotropic Total Variation
    """
    if testproblem == 'gelPhantom':
        if dynamic == True:
            b_vec = b.reshape((-1,1))
            L = time_derivative_operator(nx, ny, nt)
            (x, x_history, lambdah, lambda_history) = MMGKS(A, b_vec, L, pnorm=2, qnorm=1, projection_dim=3, n_iter = iters, regparam=reg_param, x_true=None)
#            xx = np.reshape(x, (nx, ny, nt), order="F")
        else:
            xx = list(range(nt))
            L = spatial_derivative_operator(nx, ny, 1)
            for i in range(nt):
                b_vec = B[:, i].reshape((-1,1))
                (x, x_history, lambdah, lambda_history) = MMGKS(AA[i], b_vec, L, pnorm=2, qnorm=1, projection_dim=3, n_iter = iters, regparam=reg_param, x_true=None)
                xx[i] = x
    elif testproblem == 'Test':
        if dynamic == True:
            b_vec = b.reshape((-1,1))
            L = time_derivative_operator(nx, ny, nt)
            (x, x_history, lambdah, lambda_history) = MMGKS(A, b_vec, L, pnorm=2, qnorm=1, projection_dim=3, n_iter = iters, regparam=reg_param, x_true=None)
            xx = np.reshape(x, (nx, ny, nt), order="F")
        else:
            xx = list(range(nt))
            L = spatial_derivative_operator(nx, ny, 1)
            for i in range(nt):
                print(i)
                b_vec = B[i].reshape((-1,1))
                (x, x_history, lambdah, lambda_history) = MMGKS(AA[i], b_vec, L, pnorm=2, qnorm=1, projection_dim=3, n_iter = iters, regparam=reg_param, x_true=None)
                xx[i] = x
    else:
        if dynamic == True:
            b_vec = b.reshape((-1,1))
            L = spatial_derivative_operator(nx, ny, nt)
            # L = time_derivative_operator(nx, ny, nt)
            (x, x_history, lambdah, lambda_history) = MMGKS(A, b_vec, L, pnorm=2, qnorm=1, projection_dim=3, n_iter = iters, regparam=reg_param, x_true=None)
            xx = np.reshape(x, (nx, ny, nt), order="F")
        else:
            xx = list(range(nt))
            # L = spatial_derivative_operator(nx, ny, 1)
            L = first_derivative_operator_2d(nx, ny)
            for i in range(nt):
                b_vec = B[i].reshape((-1,1))
                (x, x_history, lambdah, lambda_history) = MMGKS(AA[i], b_vec, L, pnorm=2, qnorm=1, projection_dim=3, n_iter = iters, regparam=reg_param, x_true=None)
                xx[i] = x
    return xx  


