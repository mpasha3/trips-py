from pylops import LinearOperator

from scipy import linalg as la

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


### TODO: Add a general reweighting function