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
__maintainer__ = "Mirjeta Pasha and Connor Sanderford"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com and csanderf@asu.edu; connorsanderford@gmail.com"


from pylops import LinearOperator
from operators import *
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
    

def soft_thresh(x, mu):
    #y = np.sign(x)*np.max([np.abs(x)-mu], 0)
    y = np.abs(x) - mu
    y[y < 0] = 0
    y = y * np.sign(x)
    return y

def generate_noise(shape, noise_level, dist='normal'):
    """
    Produces noise at the desired noise level.
    """

    if dist == 'normal':
        noise = np.random.randn(shape)
    elif dist == 'poisson':
        noise = np.random.poisson
    e = noise_level * noise / la.norm(noise)


def is_identity(A):
    """
    Checks whether the operator A is identity.
    """

    if isinstance(A, Identity): # check if A is a pylops identity operator
        return True

    elif (not isinstance(A, LinearOperator)) and ( A.shape[0] == A.shape[1] ) and ( np.allclose(A, np.eye(A.shape[0])) ): # check if A is an array resembling the identity matrix
        return True
    
    elif isinstance(A, _sparray) and ( A.shape[0] == A.shape[1] ) and ( A - sparse.eye(A.shape[0]) ).sum() < 10**(-6):
        return True
    
    else:
        return False

def check_noise_type(noise_type):
    if noise_type in ['g', 'p', 'l', 'gaussian', 'Gaussian', 'Poisson', 'poisson', 'Laplace', 'laplace']:
        valid = True
    else:
        valid = False
    if not valid:
       raise TypeError('You must enter a valid name for the noise. For Gaussian noise input g or Gaussian or gaussian. For Poisson noise input p or Poisson or poisson. For Laplace noise input l or laplace or laplace.')

def check_noise_level(noise_level):
    valid  = False
    if (isinstance(noise_level, float) or isinstance(noise_level, int)):
        if int(noise_level) > 0 or int(noise_level) == 0:
            valid = True
    if not valid:
        raise TypeError('You must enter a valid noise level! Choose 0 for 0 %, 1 for 1%, or other valid values acordingly.')

def check_Regparam(Regparam = 1):
    valid = False
    case1 = False
    # if str(Regparam).isnumeric():
    if (isinstance(Regparam, float) or isinstance(Regparam, int)):
        if int(Regparam) > 0:
            valid = True
        else:
            valid = False
            case1 = True
    elif Regparam in ['gcv', 'GCV', 'Gcv', 'DP', 'dp', 'Dp', 'Discrepancy Principle', 'Discrepancy principle', 'discrepancy principle']:
        valid = True
    if not valid and case1 == True:
        raise TypeError("You must specify a valid regularization parameter. Input a positive number!")
    elif not valid:
        raise TypeError("You must specify a valid regularization parameter. For Generalized Cross Validation type 'gcv'. For 'Discrepancy Principle type 'dp'.")

def check_Positivescalar(value):
    if int(value) > 0:
        valid = True
    else:
        valid = False

def check_operator_type(A):
    aa = str(type(A))
    if 'array' in aa:
        A = A
    # elif 'sparse' in aa:
    else:
        A = A.todense()
    return A

### TODO: Add a general reweighting function