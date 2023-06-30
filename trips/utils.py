from pylops import LinearOperator

from scipy import linalg as la
from scipy.sparse._arrays import _sparray
from scipy import sparse

import numpy as np

from pylops import Identity, LinearOperator

"""
Utility functions.
"""

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


def add_noise(self, x, noise_level, distribution='normal'):
        
    """
    Adds noise at the desired noise level.
    """

    distribution = distribution.lower()
    if (distribution in ['gaussian', 'normal']):

        e = np.random.randn(shape=x.shape)
        delta = np.linalg.norm(e)

        x_with_noise = x + e * (noise_level * np.linalg.norm(x)/delta)
        
    if (distribution = 'poisson'):

        gamma = 1 # background counts assumed known
        x_with_noise = np.random.poisson(lam=x+gamma) 

        e = 0
        delta = np.linalg.norm(e)

    if (distribution in ['laplace', 'laplacean']):

        e = np.random.laplace(shape=x.shape)
        delta = np.linalg.norm(e)

        x_with_noise = x + e * (noise_level * np.linalg.norm(x)/delta)
        
        return x_with_noise


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
    


### TODO: Add a general reweighting function