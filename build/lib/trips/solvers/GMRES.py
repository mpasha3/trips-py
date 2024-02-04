#!/usr/bin/env python
"""
Builds function for GMRES
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, Connor Sanderford, and Ugochukwu Obinna Ugwu"
__affiliations__ = 'Tufts University, University of Bath, Arizona State University, and Tufts University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu; connorsanderford@gmail.com; Ugochukwu.Ugwu@tufts.edu"

from ..utilities.decompositions import arnoldi
import numpy as np
from scipy import linalg as la
from tqdm import tqdm

def GMRES(A, b, n_iter = 3, dp_stop = 0, **kwargs):

    """
    Description: The basic GMRES (Generalized Minimum Residual Method) algorithm that solves a large linear system of equations
    Ax = b

    Inputs: 

    A: The matrix of the system to be solved; A square matrix (A is n x n)

    b: The right-randside of the system given as a vector
    
    n_iter: The dimension of the sybspace. The default value is 3

    dp_stop: A logical variable to define if the iterations (in Arnoldi algorithm) are stopped by the discrepancy principle
    The default value is 0.

    Outputs: 
    x_GMRES: Returns the approximate solution obtained

    Example of calling the method: 

    xx = GMRES(A, b, 10, dp_stop = 0)
    
    """
    if (A.shape[0] is not A.shape[1]):
        raise ValueError("Arnoldi can not be used. The operator is not square")
    
    b_vec = b.reshape((-1,1))
    (Vdp1,H) = arnoldi(A, b_vec, n_iter = 5)
    Vd= Vdp1[:, 0:-1]
    bhat = Vdp1.T@b_vec
    y = np.linalg.lstsq(H.T, H.T@bhat)[0]
    x_GMRES = Vdp1@y
    return (x_GMRES)