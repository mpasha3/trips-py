#!/usr/bin/env python
"""
Functions that define the regularization parameters
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, and Connor Sanderford"
__affiliations__ = 'MIT and Tufts University, Arizona State University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; csanderf@asu.edu; """

from venv import create
import numpy as np
import pylops
from scipy.ndimage import convolve
from scipy import sparse
import scipy
from scipy import linalg as la

"""regularization operators (derivatives)"""
## First derivative operator 1D
def gen_first_derivative_operator(n):
    D = scipy.sparse.diags(diagonals=np.ones(n-1), offsets=1, shape=None, format=None, dtype=None)
    L = sparse.identity(n)-D
    Lx = L[0:-1, :]
    return Lx
## First derivative operator 2D
def gen_first_derivative_operator_2D(nx, ny):
    D_x = gen_first_derivative_operator(nx)
    D_y = gen_first_derivative_operator(ny)
    IDx = sparse.kron( sparse.identity(nx), D_x)
    DyI = sparse.kron(D_y, sparse.identity(ny))
    L = sparse.vstack((IDx, DyI))
    return L

## Space time detivative operator
def gen_spacetime_derivative_operator(nx, ny, nt):
    D_spatial = gen_first_derivative_operator_2D(nx,ny)
    Lt = gen_first_derivative_operator(nt)
    ITLs = sparse.kron(sparse.identity(nt), D_spatial)
    LTIN = sparse.kron(Lt, sparse.identity(nx**2))
    L =  sparse.vstack((ITLs, LTIN))
    return L    

## Framelet operator
"""Framelet operators"""

def construct_H(l,n):

    e = np.ones((n,))
    # build H_0
    H_0 = sparse.spdiags(e, -1-l+1, n, n) + sparse.spdiags(2*e, 0, n, n) + sparse.spdiags(e, 1+l-1, n, n)
    H_0 = H_0.tocsr()
    for jj in range(0,l):
        H_0[jj, l-jj-1] += 1
        H_0[-jj-1, -l+jj] += 1
    H_0 /= 4
    # build H_1

    H_1 = sparse.spdiags(-e, -1-l+1, n, n) + sparse.spdiags(e, 1+l-1, n, n)
    H_1 = H_1.tocsr()

    for jj in range(0,l):
        H_1[jj, l-jj-1] -= 1
        H_1[-jj-1, -l+jj] += 1

    H_1 *= np.sqrt(2)/4

    # build H_2

    H_2 = sparse.spdiags(-e, -1-l+1, n, n) + sparse.spdiags(2*e, 0, n, n) + sparse.spdiags(-e, 1+l-1, n, n)
    H_2 = H_2.tocsr()

    for jj in range(0,l):
        H_2[jj, l-jj-1] -= 1
        H_2[-jj-1, -l+jj] -= 1

    H_2 /= 4

    return (H_0, H_1, H_2)


def create_analysis_operator_rec(n, level, l, w):

    if level == l:
        return sparse.vstack( construct_H(level, n) )

    else:
        (H_0, H_1, H_2) = construct_H(level, n)
        W_1 = create_analysis_operator_rec(n, level+1, l, H_0)

        return sparse.vstack( (W_1, H_1, H_2) ) * w


def create_analysis_operator(n, l):

    return create_analysis_operator_rec(n, 1, l, 1)


def create_framelet_operator(n,m,l):

    W_n = create_analysis_operator(n, l)
    W_m = create_analysis_operator(m, l)

    proj_forward = lambda x: (W_n @ (x.reshape(n,m, order='F') @ W_m.H)).reshape(-1,1, order='F')

    proj_backward = lambda x: (W_n.H @ (x.reshape( n*(2*l+1) , m*(2*l+1), order='F' ) @ W_m)).reshape(-1,1, order='F')

    W = pylops.FunctionOperator(proj_forward, proj_backward, n*(2*l+1) * m*(2*l+1), n*m)

    return W

""" 
Other operators
"""

def operator_qr(A):

    """
    Handles QR decomposition for an operator A of any form: dense or sparse array, or a pylops LinearOperator.
    """

    if isinstance(A, LinearOperator):
        return la.qr(A.todense(), mode='economic')
    else:
        return la.qr(A, mode='economic')
    

def operator_svd(A):

    """
    Handles QR decomposition for an operator A of any form: dense or sparse array, or a pylops LinearOperator.
    """

    if isinstance(A, LinearOperator):
        return la.svd(A.todense(), full_matrices=False)
    else:
        return la.svd(A, full_matrices=False)
    