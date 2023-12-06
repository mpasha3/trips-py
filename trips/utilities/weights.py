#!/usr/bin/env python
"""
Defines weights for different methods
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, Connor Sanderford, and Ugochukwu Obinna Ugwu"
__affiliations__ = 'Tufts University, University of Bath, Arizona State University, and Tufts University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu; connorsanderford@gmail.com; Ugochukwu.Ugwu@tufts.edu"


from pylops import LinearOperator
from trips.utilities.operators import *
from scipy import linalg as la
from scipy.sparse._arrays import _sparray
from scipy import sparse

import numpy as np

from pylops import Identity, LinearOperator

"""
Utility functions.
"""

# Define IsoTV weights      
def iso_TV_weights(x, nx, ny, epsilon, qnorm):
    nt = int((x.reshape((-1,1)).shape[0])/(nx*ny))
    Ls = first_derivative_operator_2d(nx, ny)
    utemp = np.reshape(x, (nx*ny, nt))
    Dutemp = Ls.dot(utemp)
    wr = np.exp(2) * np.ones((2*nx*(ny), 1))
    for i in range(2*nx*(ny)):
        wr[i] = (np.linalg.norm(Dutemp[i,:])**2 + wr[i])**(qnorm/2-1)
    wr = np.kron(np.ones((nt, 1)), wr)
    return wr

# Define GS weights
def GS_weights(x, nx, ny, epsilon, qnorm):
    nt = int((x.reshape((-1,1)).shape[0])/(nx*ny))
    utemp = np.reshape(x, (nx*ny, nt))
    Ls = first_derivative_operator_2d(nx, ny)
    Dutemp = Ls.dot(utemp)
    wr = np.exp(2) * np.ones((2*nx*(ny), 1))
    for i in range(2*nx*(ny)):
        wr[i] = (np.linalg.norm(Dutemp[i,:])**2 + wr[i])**(qnorm/2-1)
    wr = np.kron(np.ones((nt, 1)), wr)
    return wr

# Define regular weights for MMGKS
def smoothed_holder_weights(x, epsilon, p):
    z = (x**2 + epsilon**2)**(p/2 - 1)
    return z