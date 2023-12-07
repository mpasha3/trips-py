#!/usr/bin/env python
"""
Builds function for MMGKS
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, Connor Sanderford, and Ugochukwu Obinna Ugwu"
__affiliations__ = 'Tufts University, University of Bath, Arizona State University, and Tufts University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu; connorsanderford@gmail.com; Ugochukwu.Ugwu@tufts.edu"

from ..utilities.decompositions import golub_kahan, arnoldi
from ..parameter_selection.gcv import generalized_crossvalidation
from ..parameter_selection.discrepancy_principle import discrepancy_principle
from ..utilities.utils import *#smoothed_holder_weights, operator_qr, operator_svd, is_identity
from scipy import sparse
import numpy as np
from scipy import linalg as la
from pylops import Identity
from trips.utilities.weights import *
from tqdm import tqdm
from collections.abc import Iterable

def MMGKS(A, b, L, pnorm=2, qnorm=1, projection_dim=3, n_iter=5, regparam='gcv', x_true=None, **kwargs):

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    isoTV_option = kwargs['isoTV'] if ('isoTV' in kwargs) else False
    GS_option = kwargs['GS'] if ('GS' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    prob_dims = kwargs['prob_dims'] if ('prob_dims' in kwargs) else False
    regparam_sequence = kwargs['regparam_sequence'] if ('regparam_sequence' in kwargs) else [0.1*(0.5**(x)) for x in range(0,n_iter)]
    (U, B, V) = golub_kahan(A, b, projection_dim, dp_stop, **kwargs)
    
    x_history = []
    lambda_history = []
    e = 1
    x = A.T @ b # initialize x for reweighting
    AV = A@V
    LV = L@V
    for ii in tqdm(range(n_iter), desc='running MMGKS...'):
        # compute reweighting for p-norm approximation
        v = A @ x - b
        wf = (v**2 + epsilon**2)**(pnorm/2 - 1)
        AA = AV*wf
        # z = smoothed_holder_weights(v, epsilon=epsilon, p=pnorm).reshape((-1,1))**(1/2)
        # p = sparse.spdiags(data = z.flatten() , diags=0, m=z.shape[0], n=z.shape[0])
        # temp = p @ (A @ V)
        (Q_A, R_A) = la.qr(AA, mode='economic') # Project A into V, separate into Q and R
        # Compute reweighting for q-norm approximation
        u = L @ x
        if isoTV_option in ['isoTV', 'ISOTV', 'IsoTV']:
            if prob_dims == False:
                raise TypeError("For Isotropic TV you must enter the dimension of the dynamic problem! Example: (x_mmgks, info_mmgks) = MMGKS(A, data_vec, L, pnorm=2, qnorm=1, projection_dim=2, n_iter =3, regparam = 0.005, x_true = None, isoTV = 'isoTV', prob_dims = (nx,ny, nt))")
            else:
                nx = prob_dims[0]
                ny = prob_dims[1]
            #### This are the same weights as in utilities.weights
            nt = int((x.reshape((-1,1)).shape[0])/(nx*ny))
            Ls = first_derivative_operator_2d(nx, ny)
            spacen = int(Ls.shape[0] / 2)
            spacent = spacen * nt
            X = x.reshape(nx**2, nt)
            LsX = Ls @ X
            LsX1 = LsX[:spacen, :]
            LsX2 = LsX[spacen:2*spacen, :]
            weightx = (LsX1**2 + LsX2**2 + epsilon**2)**((qnorm-2) / 4)
            weightx = np.concatenate((weightx.flatten(), weightx.flatten()))
            weightt = (u[2*spacent:]**2 + epsilon**2)**((qnorm-2) / 4)
            wr = np.concatenate((weightx.reshape(-1,1), weightt))
            ######
        elif GS_option in  ['GS', 'gs', 'Gs']:
            if prob_dims == False:
                raise TypeError("For Isotropic TV you must enter the dimension of the dynamic problem. (x_mmgks, info_mmgks) = MMGKS(A, data_vec, L, pnorm=2, qnorm=1, projection_dim=2, n_iter =3, regparam = 0.005, x_true = None, isoTV = 'isoTV', prob_dims = (nx,ny, nt))")
            else:
                nx = prob_dims[0]
                ny = prob_dims[1]
            wr = GS_weights(x, nx, ny, epsilon, qnorm)
        else:
            wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))#**(1/2)
        # q = sparse.spdiags(data = z.flatten() , diags=0, m=z.shape[0], n=z.shape[0])
        # temp = q @ (L @ V)
        # wr = (u**2 + epsilon**2)**(qnorm/2 - 1)
        LL = LV * wr
        (Q_L, R_L) = la.qr(LL, mode='economic') # Project L into V, separate into Q and R

        # Compute the projected rhs
        bhat = (Q_A.T @ b).reshape(-1,1)
       
        if regparam == 'gcv':
            lambdah = generalized_crossvalidation(Q_A, R_A, R_L, b, **kwargs)#['x'].item() # find ideal lambda by crossvalidation
        elif regparam == 'dp':
            lambdah = discrepancy_principle(Q_A, R_A, R_L, b, **kwargs)#['x'].item() # find ideal lambdas by crossvalidation

        else:
            lambdah = regparam
        
        lambda_history.append(lambdah)
        # R_stacked = np.vstack( [R_A]+ [lambdah*R_L] ) # Stack projected operators
        # b_stacked = np.vstack([bhat] + [np.zeros(shape=(R_L.shape[0], 1))]) # pad with zeros
        # y, _,_,_ = la.lstsq(R_stacked, b_stacked) # get least squares solution
        y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)
        x = V @ y # project y back
        x_history.append(x)
        if ii >= R_L.shape[0]:
            break
        v = AV@y
        v = v - b
        u = LV @ y
        ra = wf * (AV @ y - b)
        ra = A.T @ ra
        rb = wr * (LV @ y)
        rb = L.T @ rb
        r = ra + lambdah * rb
        r = r - V @ (V.T @ r)
        r = r - V @ (V.T @ r)
        normed_r = r / la.norm(r) # normalize residual
        vn = r / np.linalg.norm(r)
        V = np.column_stack((V, vn))
        Avn = A @ vn
        AV = np.column_stack((AV, Avn))
        Lvn = vn
        Lvn = L*vn
        LV = np.column_stack((LV, Lvn))
        residual_history = [A@x - b for x in x_history]
    if x_true is not None:
        x_true_norm = la.norm(x_true)
        rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relError': rre_history, 'Residual': residual_history, 'its': ii}
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'Residual': residual_history, 'its': ii}
    
    return (x, info)
