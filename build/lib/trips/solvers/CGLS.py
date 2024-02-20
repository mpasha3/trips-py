#!/usr/bin/env python
"""
Builds function for CGLS
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__developers__ = "Mirjeta Pasha"
__affiliations__ = 'MIT and Tufts University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk;"

import numpy as np
import sys
def CGLS(A, b, x0, max_iter, tol,  x_true = None, **kwargs):
    """
    Description: Conjugate Gradient Least Squares method

    Inputs: 
    A: the matrix of the system to be solved

    b: the available data (righthandside)

    max_iter: the maximal number of iterations
    
    tol: a tolerance set by the user on the accuracy of the computed solution

    Outputs: 

    x: The computed solution

    x_history: A list of all the computed approximate solutions through iterations

    k: the number of iterations where GCLS converged

    Example of calling the method: 

    (x, x_history, k) = TP_cgls(A, b, x_0, max_iter = 30, tol = 0.001)

    """
    info = {}
    b = b.reshape((-1,1))
    x = x0
    r = b - A@x
    t = A.T@r
    p = t
    x_history = []
    norms_t0 = np.linalg.norm(t)
    normx = np.linalg.norm(x)
    gamma, xmax = norms_t0**2, normx
    k, check = 0, 0
    x_old = x
    rel_residual = []
    rel_error = []
    regularization_parameter = []
    while (k < max_iter) and (check == 0):
        x_old = x
        k += 1  
        w = A@p
        delta = np.linalg.norm(w)**2
        if (delta == 0):
            delta = np.eps
        beta = gamma / delta
        x = x + beta*p  
        x_history.append(x)  
        r = r - beta*w
        t  = A.T@r
        gamma_old = gamma
        norm_t = np.linalg.norm(t)
        gamma = norm_t**2
        p = t + (gamma/gamma_old)*p
        norm_x = np.linalg.norm(x)
        xmax = max(xmax, norm_x)
        check = (norm_t <= norms_t0*tol) or (norm_x*tol >= 1)
        tmp_rel_res = np.linalg.norm(x - x_old)/np.linalg.norm(x)
        rel_residual.append(tmp_rel_res)
        if x_true is not None:
           tmp_rel_err = np.linalg.norm(x - x_true)/np.linalg.norm(x)
           rel_error.append(tmp_rel_err)
           info = {'xHistory': x_history, 'regParam': regularization_parameter, 'relError': rel_error, 'relResidual': rel_residual, 'its': k}
        else:
           info = {'xHistory': x_history, 'regParam': regularization_parameter, 'relResidual': rel_residual, 'its': k}
    shrink = norm_x/xmax
    # info = {'xHistory': x_history, 'regParam': regularization_parameter, 'relError': rel_error, 'relResidual': rel_residual, 'its': k}
    return (x, info)