#!/usr/bin/env python
"""
Definition of test problems
--------------------------------------------------------------------------
Created Jun 7, 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha and Connor Sanderford"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com and csanderf@asu.edu; connorsanderford@gmail.com"

from ..decompositions import golub_kahan, arnoldi
from ..parameter_selection.gcv import generalized_crossvalidation
from ..parameter_selection.discrepancy_principle import discrepancy_principle
from ..utils import smoothed_holder_weights, operator_qr, operator_svd, is_identity
from scipy import sparse
import numpy as np
from scipy import linalg as la
from pylops import Identity
from trips.utils import *
from tqdm import tqdm

from collections.abc import Iterable


def GKS(A, b, L, projection_dim=3, n_iter=50, regparam = 'gcv', x_true=None, **kwargs):

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False

    # do you ever use this below?
    regparam_sequence = kwargs['regparam_sequence'] if ('regparam_sequence' in kwargs) else [0.1*(0.5**(x)) for x in range(0,n_iter)]

    (U, B, V) = golub_kahan(A, b, projection_dim, dp_stop, **kwargs)

    x_history = []
    lambda_history = []
    residual_history = []
    rel_error = []
    for ii in tqdm(range(n_iter), 'running GKS...'):
        
        if is_identity(L):

            Q_A, R_A, _ = la.svd(A @ V, full_matrices=False)

            R_A = np.diag(R_A)

            R_L = Identity(V.shape[1])

        else:

            # should we use operator_qr ?
            (Q_A, R_A) = la.qr(A @ V, mode='economic') # Project A into V, separate into Q and R
        
            _, R_L = la.qr(L @ V, mode='economic') # Project L into V, separate into Q and R

        bhat = (Q_A.T@b).reshape(-1,1) 

        if regparam == 'gcv':
            # lambdah = generalized_crossvalidation(Q_A, R_A, R_L, Q_A@bhat, **kwargs)#['x'].item() # find ideal lambda by crossvalidation
            # called in this way to have GCV work in a general framework
            lambdah = generalized_crossvalidation(Q_A, R_A, R_L, b, **kwargs)#['x'].item() # find ideal lambda by crossvalidation

        elif regparam == 'dp':
            ## THESE ARE NATURAL CONDITIONS, NOW EMBEDDED IN THE discrepancy_principle FCN
            # y = la.lstsq(R_A,bhat)[0]
            # nrmr = np.linalg.norm(bhat - R_A@y)
            # print(nrmr**2)
            # print(la.norm(b - Q_A@bhat)**2)
            # delta = kwargs['delta']
            # eta = kwargs['eta'] if ('eta' in kwargs) else 1.01
            # print((eta*delta)**2)
            lambdah = discrepancy_principle(Q_A, R_A, R_L, b, **kwargs)#['x'].item() # find ideal lambdas by crossvalidation

        elif isinstance(regparam, Iterable):
            lambdah = regparam[ii]
        else:
            lambdah = regparam

        lambda_history.append(lambdah)

        # bhat = (Q_A.T @ b).reshape(-1,1) # Project b

        R_stacked = np.vstack( [R_A]+ [lambdah*R_L] ) # Stack projected operators

        b_stacked = np.vstack([bhat] + [np.zeros(shape=(R_L.shape[0], 1))]) # pad with zeros

        y, _,_,_ = la.lstsq(R_stacked, b_stacked) # get least squares solution

        x = V @ y # project y back

        x_history.append(x)
        r = (A @ x).reshape(-1,1) - b.reshape(-1,1) # get residual
        residual_history.append(la.norm(r))
        ra = A.T@r
        rb = lambdah * L.T @ (L @ x)
        r = ra + rb
        r = r - V@(V.T@r)
        r = r - V@(V.T@r)
        normed_r = r / la.norm(r) # normalize residual
        V = np.hstack([V, normed_r]) # add residual to basis
        V, _ = la.qr(V, mode='economic') # orthonormalize basis using QR

    if (x_true != None).all():
        if (x_true.shape[1] != 1):
            x_true = x_true.reshape(-1,1)
        x_true_norm = la.norm(x_true)
        rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relError': rre_history, 'Residual': residual_history, 'its': ii}
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'Residual': residual_history, 'its': ii}
    return (x, info)


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

    x = A.T @ b # initialize x for reweighting

    for ii in tqdm(range(n_iter), desc='running MMGKS...'):

        # compute reweighting for p-norm approximation
        v = A @ x - b
        z = smoothed_holder_weights(v, epsilon=epsilon, p=pnorm).reshape((-1,1))**(1/2)
        p = sparse.spdiags(data = z.flatten() , diags=0, m=z.shape[0], n=z.shape[0])
        temp = p @ (A @ V)
        (Q_A, R_A) = la.qr(temp, mode='economic') # Project A into V, separate into Q and R
        # Compute reweighting for q-norm approximation
        u = L @ x
        if isoTV_option in ['isoTV', 'ISOTV', 'IsoTV']:
            if prob_dims == False:
                raise TypeError("For Isotropic TV you must enter the dimension of the dynamic problem! Example: (x_mmgks, info_mmgks) = MMGKS(A, data_vec, L, pnorm=2, qnorm=1, projection_dim=2, n_iter =3, regparam = 0.005, x_true = None, isoTV = 'isoTV', prob_dims = (nx,ny, nt))")
            else:
                nx = prob_dims[0]
                ny = prob_dims[1]
            z = iso_TV_weights(x, nx, ny, epsilon, qnorm)
        elif GS_option in  ['GS', 'gs', 'Gs']:
            if prob_dims == False:
                raise TypeError("For Isotropic TV you must enter the dimension of the dynamic problem. (x_mmgks, info_mmgks) = MMGKS(A, data_vec, L, pnorm=2, qnorm=1, projection_dim=2, n_iter =3, regparam = 0.005, x_true = None, isoTV = 'isoTV', prob_dims = (nx,ny, nt))")
            else:
                nx = prob_dims[0]
                ny = prob_dims[1]
            z = GS_weights(x, nx, ny, epsilon, qnorm)
        else:
            z = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))**(1/2)
        q = sparse.spdiags(data = z.flatten() , diags=0, m=z.shape[0], n=z.shape[0])
        temp = q @ (L @ V)
        (Q_L, R_L) = la.qr(temp, mode='economic') # Project L into V, separate into Q and R

        # Compute the projected rhs
        bhat = (Q_A.T @ b).reshape(-1,1)
        if regparam == 'gcv':
            # find ideal lambda by crossvalidation
            lambdah = generalized_crossvalidation(Q_A, R_A, R_L, b, **kwargs ) # should bhat be reweighted?
        elif regparam == 'dp':
            lambdah = discrepancy_principle((A @ V), b, (L @ V), **kwargs )#['x'].item()
        elif isinstance(regparam, Iterable):
            lambdah = regparam[ii]
        else:
            lambdah = regparam
        # if (regparam in ['gcv', 'dp']) and (ii > 1):
        #     if abs(lambdah - lambda_history[-1]) > (1)*lambda_history[-1]:
        #         lambdah = lambda_history[-1]
        lambda_history.append(lambdah)
        R_stacked = np.vstack( [R_A]+ [lambdah*R_L] ) # Stack projected operators
        b_stacked = np.vstack([bhat] + [np.zeros(shape=(R_L.shape[0], 1))]) # pad with zeros
        y, _,_,_ = la.lstsq(R_stacked, b_stacked) # get least squares solution
        x = V @ y # project y back
        x_history.append(x)
        r = p @ (A @ x).reshape(-1,1) - b.reshape(-1,1) # get residual
        ra = A.T @ r
        rb = lambdah * L.T @ (q @ (L @ x))
        r = ra  + rb
        r = r - V@(V.T@r)
        r = r - V@(V.T@r)
        normed_r = r / la.norm(r) # normalize residual
        V = np.hstack([V, normed_r]) # add residual to basis
        V, _ = la.qr(V, mode='economic') # orthonormalize basis using QR
    residual_history = [A@x - b for x in x_history]
    if x_true is not None:
        x_true_norm = la.norm(x_true)
        rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relError': rre_history, 'relResidual': residual_history, 'its': ii}
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relResidual': residual_history, 'its': ii}
    
    return (x, info)


"""
Classes which implement GKS. 
"""
class GKSClass:

    def __init__(self, projection_dim=3, regparam='gcv', projection_method='auto', **kwargs):

        self.projection_dim = projection_dim
        self.projection_method = projection_method
        self.regparam = regparam

        self.kwargs = kwargs

        self.dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False

        self.x_history = []
        self.lambda_history = []
        

    def change_regparam(self, regparam='gcv'):
        self.regparam = regparam


    def _project(self, A, b, projection_dim=None, **kwargs):
        
        if projection_dim is not None:

            if ((self.projection_method == 'auto') and (A.shape[0] == A.shape[1])) or (self.projection_method == 'arnoldi'):


                (basis,_) = arnoldi(A, b, projection_dim, self.dp_stop, **kwargs)

            else:
                (_, _, basis) = golub_kahan(A, b, projection_dim, self.dp_stop, **kwargs)
        
        else:
            
            if ((self.projection_method == 'auto') and (A.shape[0] == A.shape[1])) or (self.projection_method == 'arnoldi'):

                if A.shape[0] == A.shape[1]:
                    (basis,_) = arnoldi(A, b, projection_dim, self.dp_stop, **kwargs)

                else:
                    (_, _, basis) = golub_kahan(A, b, self.projection_dim, self.dp_stop, **kwargs)

        self.basis = basis

        return basis
    
    def restart(self):
        self.basis = None

    def run(self, A, b, L, n_iter=50, warm_start=False, x_true=None, **kwargs):

        self.regparam_sequence = kwargs['regparam_sequence'] if ('regparam_sequence' in kwargs) else [0.1*(0.5**(x)) for x in range(0,n_iter)]

        if warm_start == False:

            self._project(A, b, self.projection_dim)

            x = A.T @ b # initialize x to b for reweighting
            self.x = x

        x = self.x

        for ii in tqdm(range(n_iter), 'running GKS...'):

            if is_identity(L):

                Q_A, R_A, _ = la.svd(A @ self.basis, full_matrices=False)

                (Q_L, R_L) = (Identity(L.shape[0]) @ self.basis, Identity(L.shape[0]) @ self.basis)

                R_A = np.diag(R_A)

            else:

                (Q_A, R_A) = la.qr(A @ self.basis, mode='economic') # Project A into V, separate into Q and R
        
                (Q_L, R_L) = la.qr(L @ self.basis, mode='economic') # Project L into V, separate into Q and R
            
            if self.regparam == 'gcv':
                lambdah = generalized_crossvalidation(A @ self.basis, b, L @ self.basis, **self.kwargs)['x'].item() # find ideal lambda by crossvalidation
            
            elif self.regparam == 'dp':
                lambdah = discrepancy_principle(A @ self.basis, b, L @ self.basis, **self.kwargs)['x'].item() # find ideal lambdas by crossvalidation
            
            elif self.regparam == 'gcv+sequence':
                if ii == 0:
                    lambdah = generalized_crossvalidation(A @ V, b, L @ V, **self.kwargs)['x'].item() # find ideal lambda by crossvalidation
                else:
                    lambdah = self.lambda_history[0] * self.regparam_sequence[ii]

            elif isinstance(self.regparam, Iterable):
                lambdah = self.regparam[ii]
            
            else:
                lambdah = self.regparam

            if (self.regparam in ['gcv', 'dp']) and (ii > 1):

                if abs(lambdah - self.lambda_history[-1]) > (1)*self.lambda_history[-1]:
                    lambdah = self.lambda_history[-1]


            self.lambda_history.append(lambdah)

            bhat = (Q_A.T @ b).reshape(-1,1) # Project b

            R_stacked = np.vstack( [R_A]+ [lambdah*R_L] ) # Stack projected operators

            b_stacked = np.vstack([bhat] + [np.zeros(shape=(R_L.shape[0], 1))]) # pad with zeros

            y, _,_,_ = la.lstsq(R_stacked, b_stacked) # get least squares solution

            x = self.basis @ y # project y back

            self.x_history.append(x)

            r = (A @ x).reshape(-1,1) - b.reshape(-1,1) # get residual
            ra = A.T@r

            rb = lambdah * L.T @ (L @ x)
            r = ra + rb


            normed_r = r / la.norm(r) # normalize residual

            self.basis = np.hstack([self.basis, normed_r]) # add residual to basis

            self.basis, _ = la.qr(self.basis, mode='economic') # orthonormalize basis using QR

            self.x = x


        if x_true is not None:
            x_true_norm = la.norm(x_true)
            residual_history = [A@x - b for x in self.x_history]
            rre_history = [la.norm(x - x_true)/x_true_norm for x in self.x_history]

            self.residual_history = residual_history
            self.rre_history = rre_history

            return x
        
        else:
            return x


class MMGKSClass:

    def __init__(self, pnorm=2, qnorm=1, projection_dim=3, regparam='gcv', projection_method='auto', **kwargs):

        self.pnorm = pnorm
        self.qnorm = qnorm

        self.projection_dim = projection_dim
        self.projection_method = projection_method
        self.regparam = regparam

        self.kwargs = kwargs

        self.dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
        self.epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.001

        self.x_history = []
        self.lambda_history = []

    def change_regparam(self, regparam='gcv'):
        self.regparam = regparam

    def _project(self, A, b, projection_dim=None, **kwargs):
        
        if projection_dim is not None:

            if ((self.projection_method == 'auto') and (A.shape[0] == A.shape[1])) or (self.projection_method == 'arnoldi'):

                (basis,_) = arnoldi(A, b, projection_dim, self.dp_stop, **kwargs)

            else:
                (_, _, basis) = golub_kahan(A, b, projection_dim, self.dp_stop, **kwargs)
        
        else:
            
            if ((self.projection_method == 'auto') and (A.shape[0] == A.shape[1])) or (self.projection_method == 'arnoldi'):

                if A.shape[0] == A.shape[1]:
                    (basis,_) = arnoldi(A, b, projection_dim, self.dp_stop, **kwargs)

                else:
                    (_, _, basis) = golub_kahan(A, b, self.projection_dim, self.dp_stop, **kwargs)

        self.basis = basis

        return basis

    def restart(self):
        self.basis = None

    def run(self, A, b, L, n_iter=50, warm_start=False, x_true=None, **kwargs):

        self.regparam_sequence = kwargs['regparam_sequence'] if ('regparam_sequence' in kwargs) else [0.1*(0.5**(x)) for x in range(0,n_iter)]

        if warm_start == False:

            self._project(A, b, self.projection_dim)

            x = A.T @ b # initialize x for reweighting
            self.x = x

        x = self.x

        for ii in tqdm(range(n_iter), 'running MMGKS...'):

            v = A @ x - b
            z = smoothed_holder_weights(v, epsilon=self.epsilon, p=self.pnorm).reshape((-1,1))**(1/2)
            p = z[:, np.newaxis]
            temp = p * (A @ self.basis)

            (Q_A, R_A) = la.qr(temp, mode='economic') # Project A into V, separate into Q and R
            
            u = L @ x
            z = smoothed_holder_weights(u, epsilon=self.epsilon, p=self.qnorm).reshape((-1,1))**(1/2)
            q = z[:, np.newaxis]
            temp = q * (L @ self.basis) 
        
            (Q_L, R_L) = la.qr(temp, mode='economic') # Project L into V, separate into Q and R

            
            if self.regparam == 'gcv':
                lambdah = generalized_crossvalidation(p * (A @ self.basis), b, q * (L @ self.basis), **self.kwargs)['x'].item() # find ideal lambda by crossvalidation
            
            elif self.regparam == 'dp':
                lambdah = discrepancy_principle(p * (A @ self.basis), b, q * (L @ self.basis), **self.kwargs)['x'].item() # find ideal lambdas by crossvalidation
            
            elif self.regparam == 'gcv+sequence':
                if ii == 0:
                    lambdah = generalized_crossvalidation(A @ V, b, L @ V, **self.kwargs)['x'].item() # find ideal lambda by crossvalidation
                else:
                    lambdah = self.lambda_history[0] * self.regparam_sequence[ii]
            
            elif isinstance(self.regparam, Iterable):
                lambdah = self.regparam[ii]
            
            else:
                lambdah = self.regparam

            if (self.regparam in ['gcv', 'dp']) and (ii > 1):

                if abs(lambdah - self.lambda_history[-1]) > (1)*self.lambda_history[-1]:
                    lambdah = self.lambda_history[-1]

            self.lambda_history.append(lambdah)

            bhat = (Q_A.T @ b).reshape(-1,1) # Project b

            R_stacked = np.vstack( [R_A]+ [lambdah*R_L] ) # Stack projected operators

            b_stacked = np.vstack([bhat] + [np.zeros(shape=(R_L.shape[0], 1))]) # pad with zeros

            y, _,_,_ = la.lstsq(R_stacked, b_stacked) # get least squares solution

            x = self.basis @ y # project y back

            self.x_history.append(x)

            r = p * (A @ x).reshape(-1,1) - b.reshape(-1,1) # get residual
            ra = A.T @ r

            rb = lambdah * L.T @ (q * (L @ x))# this likely needs to include information from the pnorm weighting
            r = ra  + rb


            normed_r = r / la.norm(r) # normalize residual

            self.basis = np.hstack([self.basis, normed_r]) # add residual to basis

            self.basis, _ = la.qr(self.basis, mode='economic') # orthonormalize basis using QR

            self.x = x

        if x_true is not None:
            x_true_norm = la.norm(x_true)
            residual_history = [A@x - b for x in self.x_history]
            rre_history = [la.norm(x - x_true)/x_true_norm for x in self.x_history]

            self.residual_history = residual_history
            self.rre_history = rre_history

            return x
        
        else:
            return x