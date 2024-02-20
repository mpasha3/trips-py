#!/usr/bin/env python
"""
Builds function for GKS
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, and Connor Sanderford"
__affiliations__ = 'MIT and Tufts University, University of Bath, Arizona State University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu;"

from ..utilities.decompositions import golub_kahan, arnoldi
from ..utilities.reg_param.gcv import *
from ..utilities.reg_param.discrepancy_principle import *
from ..utilities.utils import *#smoothed_holder_weights, operator_qr, operator_svd, is_identity
from scipy import sparse
import numpy as np
from scipy import linalg as la
from pylops import Identity
from ..utilities.weights import *
from tqdm import tqdm
from collections.abc import Iterable

def GKS(A, b, L, projection_dim=3, n_iter=50, regparam = 'gcv', x_true=None, **kwargs):

    delta = kwargs['delta'] if ('delta' in kwargs) else None
    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False

    if (regparam == 'dp' or dp_stop != False) and delta == None:
        raise Exception("""A value for the noise level delta was not provided and the discrepancy principle cannot be applied. 
                    Please supply a value of delta based on the estimated noise level of the problem, or choose the regularization parameter according to gcv or a different stopping criterion.""")

    (U, B, V) = golub_kahan(A, b, projection_dim, dp_stop, **kwargs)
    AV = A@V
    LV = L@V
    x_history = []
    lambda_history = []
    residual_history = []
    for ii in tqdm(range(n_iter), 'running GKS...'):
        
        if is_identity(L):

            Q_A, R_A, _ = la.svd(AV, full_matrices=False)

            R_A = np.diag(R_A)

            R_L = Identity(V.shape[1])

        else:

            (Q_A, R_A) = la.qr(AV, mode='economic') # Project A into V, separate into Q and R
        
            _, R_L = la.qr(LV, mode='economic') # Project L into V, separate into Q and R

        bhat = (Q_A.T@b).reshape(-1,1) 

        if regparam == 'gcv':
            # lambdah = generalized_crossvalidation(Q_A, R_A, R_L, Q_A@bhat, **kwargs)#['x'].item() # find ideal lambda by crossvalidation
            # called in this way to have GCV work in a general framework
            lambdah = generalized_crossvalidation(Q_A, R_A, R_L, b, **kwargs)#['x'].item() # find ideal lambda by crossvalidation

        elif regparam == 'dp':
            lambdah = discrepancy_principle(Q_A, R_A, R_L, b, **kwargs)#['x'].item() # find ideal lambdas by crossvalidation

        elif isinstance(regparam, Iterable):
            lambdah = regparam[ii]
        else:
            lambdah = regparam

        lambda_history.append(lambdah)

        y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)

        x = V @ y # project y back
        x_history.append(x)
        v = AV@y
        v = v - b
        u = LV @ y
        ra = AV @ y - b
        ra = A.T @ ra
        rb = (LV @ y)
        rb = L.T @ rb
        r = ra + lambdah * rb
        r = r - V @ (V.T @ r)
        r = r - V @ (V.T @ r)
        r = r - V @ (V.T @ r)
        residual_history.append(la.norm(r))
        vn = r / np.linalg.norm(r)
        V = np.column_stack((V, vn))
        Avn = A @ vn
        AV = np.column_stack((AV, Avn))
        Lvn = vn
        Lvn = L*vn
        LV = np.column_stack((LV, Lvn))
    if (x_true is not None):
        if x_true.shape[1] is not 1:
            x_true = x_true.reshape(-1,1)
        x_true_norm = la.norm(x_true)
        rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relError': rre_history, 'Residual': residual_history, 'its': ii}
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'Residual': residual_history, 'its': ii}
    return (x, info)
