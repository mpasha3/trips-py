from select import select
from .decompositions import generalized_golub_kahan
from .parameter_selection import generalized_crossvalidation, discrepancy_principle
from .utils import smoothed_holder_weights

import numpy as np
from scipy import linalg as la

from tqdm import tqdm

"""
Functions which implement variants of GKS.
"""

def GKS(A, b, L, projection_dim=3, iter=50, selection_method = 'gcv', **kwargs):

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False

    (U, betas, alphas, V) = generalized_golub_kahan(A, b, projection_dim, dp_stop, **kwargs) # Find a small basis V
    
    x_history = []
    lambda_history = []

    for ii in tqdm(range(iter), 'running GKS...'):

        (Q_A, R_A) = la.qr(A @ V, mode='economic') # Project A into V, separate into Q and R
        
        (Q_L, R_L) = la.qr(L @ V, mode='economic') # Project L into V, separate into Q and R
        
        if selection_method == 'gcv':
            lambdah = generalized_crossvalidation(A @ V, b, L @ V, **kwargs)['x'] # find ideal lambda by crossvalidation
        else:
            lambdah = discrepancy_principle(A @ V, b, L @ V, **kwargs)['x'] # find ideal lambdas by crossvalidation


        lambda_history.append(lambdah)

        bhat = (Q_A.T @ b).reshape(-1,1) # Project b

        R_stacked = np.vstack( [R_A]+ [lambdah*R_L] ) # Stack projected operators

        b_stacked = np.vstack([bhat] + [np.zeros(shape=(R_L.shape[0], 1))]) # pad with zeros

        y, _,_,_ = la.lstsq(R_stacked, b_stacked) # get least squares solution

        x = V @ y # project y back

        x_history.append(x)

        r = (A @ x).reshape(-1,1) - b.reshape(-1,1) # get residual
        ra = A.T@r

        rb = lambdah[0] * L.T @ (L @ x)
        r = ra + rb

        #r = r - V@(V.T@r)
        #r = r - V@(V.T@r)

        normed_r = r / la.norm(r) # normalize residual

        V = np.hstack([V, normed_r]) # add residual to basis

        V, _ = la.qr(V, mode='economic') # orthonormalize basis using QR


    return (x, x_history, lambdah, lambda_history)



def MMGKS(A, b, L, pnorm=2, qnorm=2, projection_dim=3, iter=50, selection_method='gcv', **kwargs):

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False

    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.001

    (U, betas, alphas, V) = generalized_golub_kahan(A, b, projection_dim, dp_stop, **kwargs) # Find a small basis V
    
    x_history = []
    lambda_history = []

    x = A.T @ b # initialize x to b for reweighting

    for ii in tqdm(range(iter), desc='running MMGKS...'):

        # compute reweighting for p-norm approximation
        v = A @ x - b
        z = smoothed_holder_weights(v, epsilon=epsilon, p=pnorm).flatten()**(1/2)
        p = z[:, np.newaxis]
        temp = p * (A @ V)

        (Q_A, R_A) = la.qr(temp, mode='economic') # Project A into V, separate into Q and R
        

        # Compute reweighting for q-norm approximation
        u = L @ x
        z = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).flatten()**(1/2)
        q = z[:, np.newaxis]
        temp = q * (L @ V)  
        (Q_L, R_L) = la.qr(temp, mode='economic') # Project L into V, separate into Q and R

        if selection_method == 'gcv':
            lambdah = generalized_crossvalidation(p * (A @ V), b, q * (L @ V), **kwargs )['x'] # find ideal lambda by crossvalidation
        else:
            lambdah = discrepancy_principle(p * (A @ V), b, q * (L @ V), **kwargs )['x']
        
        lambda_history.append(lambdah)

        bhat = (Q_A.T @ b).reshape(-1,1) # Project b

        R_stacked = np.vstack( [R_A]+ [lambdah*R_L] ) # Stack projected operators

        b_stacked = np.vstack([bhat] + [np.zeros(shape=(R_L.shape[0], 1))]) # pad with zeros

        y, _,_,_ = la.lstsq(R_stacked, b_stacked) # get least squares solution

        x = V @ y # project y back
        
        x_history.append(x)

        r = p * (A @ x).reshape(-1,1) - b.reshape(-1,1) # get residual
        ra = A.T @ r

        rb = lambdah[0] * L.T @ (q * (L @ x))# this likely needs to include information from the pnorm weighting
        r = ra  + rb

        #r = r - V@(V.T@r)
        #r = r - V@(V.T@r)


        normed_r = r / la.norm(r) # normalize residual


        V = np.hstack([V, normed_r]) # add residual to basis

        V, _ = la.qr(V, mode='economic') # orthonormalize basis using QR


    return (x, x_history, lambdah, lambda_history)


"""
Classes which implement GKS. 
"""

class GKSClass:

    def __init__(self, projection_dim=3, selection_method='gcv', **kwargs):

        self.projection_dim = projection_dim
        self.selection_method = selection_method

        self.kwargs = kwargs

        self.dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
        epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.001

        self.x_history = []
        self.lambda_history = []

    def _project(self, A, b, projection_dim=None, **kwargs):
        
        if projection_dim is not None:

            (_,_,_,basis) = generalized_golub_kahan(A, b, projection_dim, self.dp_stop, **kwargs)
        
        else:
            (_,_,_,basis) = generalized_golub_kahan(A, b, self.projection_dim, self.dp_stop, **kwargs)

        self.basis = basis

        return basis
    
    def restart(self):
        self.basis = None

    def run(self, A, b, L, iter=50, warm_start=False):

        if warm_start == False:

            self._project(A, b, self.projection_dim)

            x = A.T @ b # initialize x to b for reweighting
            self.x = x

        x = self.x

        for ii in tqdm(range(iter), 'running GKS...'):

            (Q_A, R_A) = la.qr(A @ self.basis, mode='economic') # Project A into V, separate into Q and R
            
            (Q_L, R_L) = la.qr(L @ self.basis, mode='economic') # Project L into V, separate into Q and R
            
            if self.selection_method == 'gcv':
                lambdah = generalized_crossvalidation(A @ self.basis, b, L @ self.basis, **self.kwargs)['x'] # find ideal lambda by crossvalidation
            else:
                lambdah = discrepancy_principle(A @ self.basis, b, L @ self.basis, **self.kwargs)['x'] # find ideal lambdas by crossvalidation


            self.lambda_history.append(lambdah)

            bhat = (Q_A.T @ b).reshape(-1,1) # Project b

            R_stacked = np.vstack( [R_A]+ [lambdah*R_L] ) # Stack projected operators

            b_stacked = np.vstack([bhat] + [np.zeros(shape=(R_L.shape[0], 1))]) # pad with zeros

            y, _,_,_ = la.lstsq(R_stacked, b_stacked) # get least squares solution

            x = self.basis @ y # project y back

            self.x_history.append(x)

            r = (A @ x).reshape(-1,1) - b.reshape(-1,1) # get residual
            ra = A.T@r

            rb = lambdah[0] * L.T @ (L @ x)
            r = ra + rb


            normed_r = r / la.norm(r) # normalize residual

            self.basis = np.hstack([self.basis, normed_r]) # add residual to basis

            self.basis, _ = la.qr(self.basis, mode='economic') # orthonormalize basis using QR

            self.x = x


        return x
        

    
class GKSClass:

    def __init__(self, projection_dim=3, selection_method='gcv', **kwargs):

        self.projection_dim = projection_dim
        self.selection_method = selection_method

        self.kwargs = kwargs

        self.dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False

        self.x_history = []
        self.lambda_history = []

    def _project(self, A, b, projection_dim=None, **kwargs):
        
        if projection_dim is not None:

            (_,_,_,basis) = generalized_golub_kahan(A, b, projection_dim, self.dp_stop, **kwargs)
        
        else:
            (_,_,_,basis) = generalized_golub_kahan(A, b, self.projection_dim, self.dp_stop, **kwargs)

        self.basis = basis

        return basis
    
    def restart(self):
        self.basis = None

    def run(self, A, b, L, iter=50, warm_start=False):

        if warm_start == False:

            self._project(A, b, self.projection_dim)

            x = A.T @ b # initialize x to b for reweighting
            self.x = x

        x = self.x

        for ii in tqdm(range(iter), 'running GKS...'):

            (Q_A, R_A) = la.qr(A @ self.basis, mode='economic') # Project A into V, separate into Q and R
            
            (Q_L, R_L) = la.qr(L @ self.basis, mode='economic') # Project L into V, separate into Q and R
            
            if self.selection_method == 'gcv':
                lambdah = generalized_crossvalidation(A @ self.basis, b, L @ self.basis, **self.kwargs)['x'] # find ideal lambda by crossvalidation
            else:
                lambdah = discrepancy_principle(A @ self.basis, b, L @ self.basis, **self.kwargs)['x'] # find ideal lambdas by crossvalidation


            self.lambda_history.append(lambdah)

            bhat = (Q_A.T @ b).reshape(-1,1) # Project b

            R_stacked = np.vstack( [R_A]+ [lambdah*R_L] ) # Stack projected operators

            b_stacked = np.vstack([bhat] + [np.zeros(shape=(R_L.shape[0], 1))]) # pad with zeros

            y, _,_,_ = la.lstsq(R_stacked, b_stacked) # get least squares solution

            x = self.basis @ y # project y back

            self.x_history.append(x)

            r = (A @ x).reshape(-1,1) - b.reshape(-1,1) # get residual
            ra = A.T@r

            rb = lambdah[0] * L.T @ (L @ x)
            r = ra + rb


            normed_r = r / la.norm(r) # normalize residual

            self.basis = np.hstack([self.basis, normed_r]) # add residual to basis

            self.basis, _ = la.qr(self.basis, mode='economic') # orthonormalize basis using QR

            self.x = x


        return x



class MMGKSClass:

    def __init__(self, pnorm=1, qnorm=1, projection_dim=3, selection_method='gcv', **kwargs):

        self.pnorm = pnorm
        self.qnorm=qnorm

        self.projection_dim = projection_dim
        self.selection_method = selection_method

        self.kwargs = kwargs

        self.dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
        self.epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.001

        self.x_history = []
        self.lambda_history = []

    def _project(self, A, b, projection_dim=None, **kwargs):
        
        if projection_dim is not None:

            (_,_,_,basis) = generalized_golub_kahan(A, b, projection_dim, self.dp_stop, **kwargs)
        
        else:
            (_,_,_,basis) = generalized_golub_kahan(A, b, self.projection_dim, self.dp_stop, **kwargs)

        self.basis = basis

        return basis

    def restart(self):
        self.basis = None

    def run(self, A, b, L, iter=50, warm_start=False):

        if warm_start == False:

            self._project(A, b, self.projection_dim)

            x = A.T @ b # initialize x to b for reweighting
            self.x = x

        x = self.x

        for ii in tqdm(range(iter), 'running GKS...'):

            v = A @ x - b
            z = smoothed_holder_weights(v, epsilon=self.epsilon, p=self.pnorm).flatten()**(1/2)
            p = z[:, np.newaxis]
            temp = p * (A @ self.basis)

            (Q_A, R_A) = la.qr(temp, mode='economic') # Project A into V, separate into Q and R

            
            u = L @ x
            z = smoothed_holder_weights(u, epsilon=self.epsilon, p=self.qnorm).flatten()**(1/2)
            q = z[:, np.newaxis]
            temp = q * (L @ self.basis)  
            (Q_L, R_L) = la.qr(temp, mode='economic') # Project L into V, separate into Q and R
            
            if self.selection_method == 'gcv':
                lambdah = generalized_crossvalidation(p * (A @ self.basis), b, p * (L @ self.basis), **self.kwargs)['x'] # find ideal lambda by crossvalidation
            else:
                lambdah = discrepancy_principle(p * (A @ self.basis), b, p * (L @ self.basis), **self.kwargs)['x'] # find ideal lambdas by crossvalidation


            self.lambda_history.append(lambdah)

            bhat = (Q_A.T @ b).reshape(-1,1) # Project b

            R_stacked = np.vstack( [R_A]+ [lambdah*R_L] ) # Stack projected operators

            b_stacked = np.vstack([bhat] + [np.zeros(shape=(R_L.shape[0], 1))]) # pad with zeros

            y, _,_,_ = la.lstsq(R_stacked, b_stacked) # get least squares solution

            x = self.basis @ y # project y back

            self.x_history.append(x)

            r = p * (A @ x).reshape(-1,1) - b.reshape(-1,1) # get residual
            ra = A.T @ r

            rb = lambdah[0] * L.T @ (q * (L @ x))# this likely needs to include information from the pnorm weighting
            r = ra  + rb


            normed_r = r / la.norm(r) # normalize residual

            self.basis = np.hstack([self.basis, normed_r]) # add residual to basis

            self.basis, _ = la.qr(self.basis, mode='economic') # orthonormalize basis using QR

            self.x = x


        return x