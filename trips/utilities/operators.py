#!/usr/bin/env python
"""
Functions that define the regularization parameters
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, Connor Sanderford, and Ugochukwu Obinna Ugwu"
__affiliations__ = 'Tufts University, University of Bath, Arizona State University, and Tufts University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu; connorsanderford@gmail.com; Ugochukwu.Ugwu@tufts.edu"

from venv import create
import numpy as np
import pylops
from scipy.ndimage import convolve
from scipy import sparse
import scipy

"""regularization operators (derivatives)"""

def gen_first_derivative_operator(n):
    D = scipy.sparse.diags(diagonals=np.ones(n-1), offsets=1, shape=None, format=None, dtype=None)
    L = sparse.identity(n)-D
    Lx = L[0:-1, :]
    return Lx

def gen_first_derivative_operator_2D(nx, ny):
    D_x = gen_first_derivative_operator(nx)
    D_y = gen_first_derivative_operator(ny)
    IDx = sparse.kron( sparse.identity(nx), D_x)
    DyI = sparse.kron(D_y, sparse.identity(ny))
    L = sparse.vstack((IDx, DyI))
    return L

def gen_spacetime_derivative_operator(nx, ny, nt):
    D_spatial = gen_first_derivative_operator_2D(nx,ny)
    Lt = gen_first_derivative_operator(nt)
    ITLs = sparse.kron(sparse.identity(nt), D_spatial)
    LTIN = sparse.kron(Lt, sparse.identity(nx**2))
    L =  sparse.vstack((ITLs, LTIN))
    return L    

    