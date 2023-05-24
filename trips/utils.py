from pylops import LinearOperator

from scipy import linalg as la
import numpy as np

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

def generate_noise(shape, noise_level, dist='normal'):
    """
    Produces noise at the desired noise level.
    """

    if dist == 'normal':
        noise = np.random.randn(shape)
    elif dist == 'poisson':
        noise = np.random.poisson
    e = noise_level * noise / la.norm(noise)
    


### TODO: Add a general reweighting function