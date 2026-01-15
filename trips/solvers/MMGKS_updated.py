#!/usr/bin/env python
"""
Builds function for MMGKS
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, Connor Sanderford, and Ugochukwu Obinna Ugwu"
__affiliations__ = 'MIT and Tufts University, University of Bath, Arizona State University, and Tufts University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu; connorsanderford@gmail.com; Ugochukwu.Ugwu@tufts.edu"

from ..utilities.decompositions import *
from ..utilities.reg_param.gcv import *
from ..utilities.reg_param.discrepancy_principle import *
from ..utilities.reg_param.l_curve import *
from ..utilities.utils import *
from scipy import sparse
import numpy as np
from scipy import linalg as la
from pylops import Identity
from trips.utilities.weights import *
from tqdm import tqdm
from collections.abc import Iterable

def MMGKS(A, b, L, pnorm=2, qnorm=1, projection_dim=3, n_iter=5, regparam='gcv', x_true=None, exp = 0.5, **kwargs):

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    isoTV_option = kwargs['isoTV'] if ('isoTV' in kwargs) else False
    GS_option = kwargs['GS'] if ('GS' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    prob_dims = kwargs['prob_dims'] if ('prob_dims' in kwargs) else False
    non_neg = kwargs['non_neg'] if ('non_neg' in kwargs) else False
    regparam_sequence = kwargs['regparam_sequence'] if ('regparam_sequence' in kwargs) else [0.1*(0.5**(x)) for x in range(0,n_iter)]
    (U, B, V) = golub_kahan(A, b, projection_dim, dp_stop, **kwargs)
    
    x_history = []
    lambda_history = []
    residual_history = []
    e = 1
    x = A.T @ b 
    AV = A@V
    if GS_option in  ['GS', 'gs', 'Gs']:
        nx = prob_dims[0]
        ny = prob_dims[1]
        nt = prob_dims[2]
        Ls = generate_first_derivative_operator_2d_matrix(nx, ny)
        # Ls = first_derivative_operator_2d(nx, ny)
        L = sparse.kron(sparse.identity(nt), Ls)
        LV = L@V
    else:
        LV = L@V
    for ii in tqdm(range(n_iter), desc='running MMGKS...'):
        v = A @ x - b
        wf = (v**2 + epsilon**2)**(pnorm/2 - 1)
        AA = AV*(wf**exp)
        (Q_A, R_A) = la.qr(AA, mode='economic') 
        u = L @ x
        if isoTV_option in ['isoTV', 'ISOTV', 'IsoTV']:
            if prob_dims == False:
                raise TypeError("For Isotropic TV you must enter the dimension of the dynamic problem! Example: (x_mmgks, info_mmgks) = MMGKS(A, data_vec, L, pnorm=2, qnorm=1, projection_dim=2, n_iter =3, regparam = 'gcv', x_true = None, isoTV = 'isoTV', prob_dims = (nx,ny, nt))")
            else:
                nx = prob_dims[0]
                ny = prob_dims[1]
            Ls = first_derivative_operator_2d(nx, ny)
            nt = int((x.reshape((-1,1)).shape[0])/(nx*ny))
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
        elif GS_option in  ['GS', 'gs', 'Gs']:
            if prob_dims == False:
                raise TypeError("For Isotropic Group Sparsity you must enter the dimension of the dynamic problem. (x_mmgks, info_mmgks) = MMGKS(A, data_vec, L, pnorm=2, qnorm=1, projection_dim=2, n_iter =3, regparam = 'gcv', x_true = None, GS = 'GS', prob_dims = (nx,ny, nt))")
            else:
                nx = prob_dims[0]
                ny = prob_dims[1]
            nt = int((x.reshape((-1,1)).shape[0])/(nx*ny))
            utemp = np.reshape(x, (nx*ny, nt))
            Dutemp = Ls.dot(utemp)
            wr = np.exp(2) * np.ones((2*nx*(ny-1), 1))
            for i in range(2*nx*(ny-1)):
                wr[i] = (np.linalg.norm(Dutemp[i,:])**2 + wr[i])**(qnorm/2-1)
            wr = np.kron(np.ones((nt, 1)), wr)
        else:
            wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))
        LL = LV * (wr**exp)
        (Q_L, R_L) = la.qr(LL, mode='economic') 
        if regparam == 'gcv':
            lambdah = generalized_crossvalidation(Q_A, R_A, R_L, (wf**exp) *b, **kwargs)
        elif regparam == 'dp':
            lambdah = discrepancy_principle(Q_A, R_A, R_L, (wf**exp) *b, **kwargs)
        elif regparam == 'l_curve':
            lambdah = l_curve(R_A, R_L,Q_A.T@ ((wf**exp)*b))
        else:
            lambdah = regparam
        
        lambda_history.append(lambdah)
        y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ ((wf**exp)*b), np.zeros((R_L.shape[0],1)))),rcond=None)
        x = V @ y
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
        normed_r = r / la.norm(r) 
        vn = r / np.linalg.norm(r)
        V = np.column_stack((V, vn))
        Avn = A @ vn
        AV = np.column_stack((AV, Avn))
        Lvn = vn
        Lvn = L*vn
        LV = np.column_stack((LV, Lvn))
        residual_history.append(la.norm(r))
    if x_true is not None:
        x_true_norm = la.norm(x_true)
        rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relError': rre_history, 'Residual': residual_history, 'its': ii}
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'Residual': residual_history, 'its': ii}
    
    return (x, info)
