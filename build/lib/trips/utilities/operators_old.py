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

"""regularization operators (derivatives)"""

def first_derivative_operator(n):
    """ 
    Description: Defines a matrix that is the discretization of the first derivative operator
    Inputs: 
        n: The dimension of the image

    Outputs: 
        L: An operator of dimensions (n x n) 
    """
    L = pylops.FirstDerivative(n, dtype="float32")

    return L

def first_derivative_operator_2d(nx, ny):

    D_x = first_derivative_operator(nx)
    D_y = first_derivative_operator(ny)

    IDx = pylops.Kronecker( pylops.Identity(nx, dtype='float32'), D_x )
    DyI = pylops.Kronecker( D_y, pylops.Identity(ny, dtype='float32') )

    D_spatial = pylops.VStack((IDx, DyI))

    return D_spatial

def spatial_derivative_operator(nx, ny, nt):

    D_spatial = first_derivative_operator_2d(nx,ny)

    ID_spatial = pylops.Kronecker( pylops.Identity(nt, dtype='float32'), D_spatial)

    return ID_spatial

def time_derivative_operator(nx, ny, nt):
    
    D_time = first_derivative_operator(nt)

    D_timeI = pylops.Kronecker( D_time, pylops.Identity(nx**2, dtype='float32') )

    return D_timeI


"""" Regularization operators as matrices here"""

def generate_first_derivative_operator_matrix(n):

    D = sparse.spdiags( data=np.ones(n-1) , diags=-1, m=n, n=n)
    L = sparse.identity(n)-D
    L = L[0:-1, :]

    return L


def generate_first_derivative_operator_2d_matrix(nx, ny):

    D_x = generate_first_derivative_operator_matrix(nx)
    D_y = generate_first_derivative_operator_matrix(ny)

    IDx = sparse.kron( sparse.identity(nx), D_x)
    DyI = sparse.kron(D_y, sparse.identity(ny))

    L = sparse.vstack((IDx, DyI))

    return L


def generate_spatial_derivative_operator_matrix(nx, ny, nt):
    
    D_spatial = generate_first_derivative_operator_2d_matrix(nx,ny)
    
    ID_spatial = sparse.kron( sparse.identity(nt), D_spatial)
    
    return ID_spatial


def generate_time_derivative_operator_matrix(nx, ny, nt):
    
    D_time = generate_first_derivative_operator_2d_matrix(nt)[:-1, :]
    
    D_timeI = sparse.kron(D_time, sparse.identity(nx**2))
    
    return D_timeI

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
    


if __name__ == "__main__":

    print(first_derivative_operator_2d(10,10))

    L_spatial = spatial_derivative_operator(10,10,10)

    L_time = time_derivative_operator(10,10,10)

    print(L_time)

    print(L_spatial)

    arr = np.random.rand(1000,1)

    arr2 = np.random.rand(1000,1000)

    print(L_spatial @ arr)

    thing = create_framelet_operator(10, 10, 2)

    out = thing @ np.eye(10).reshape(-1,1, order='F')

    back = thing.T @ out

    (H0, H1, H2) = construct_H(2,10)

    output = H2 @ np.eye(10)

    breakpoint()