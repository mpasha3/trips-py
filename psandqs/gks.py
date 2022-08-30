from .decompositions import generalized_golub_kahan
from .parameter_selection import generalized_crossvalidation
from .utils import smoothed_holder_weights

import numpy as np
from scipy import linalg as la

from tqdm import tqdm

"""
Functions which implement variants of GKS.
"""

def GKS(A, b, L, projection_dim, iter):

    (U, betas, alphas, V) = generalized_golub_kahan(A, b, projection_dim) # Find a small basis V
    
    x_history = []
    lambda_history = []

    for ii in tqdm(range(iter)):

        (Q_A, R_A) = la.qr(A @ V, mode='economic') # Project A into V, separate into Q and R
        
        (Q_L, R_L) = la.qr(L @ V, mode='economic') # Project L into V, separate into Q and R
        

        lambdah = generalized_crossvalidation(A @ V, b, L @ V)['x'] # find ideal lambdas by crossvalidation
        
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



def MMGKS(A, b, L, pnorm, projection_dim, iter):

    (U, betas, alphas, V) = generalized_golub_kahan(A, b, projection_dim) # Find a small basis V
    
    x_history = []
    lambda_history = []

    x = A.T @ b # initialize x to b for reweighting

    for ii in tqdm(range(iter)):

        (Q_A, R_A) = np.linalg.qr(A @ V) # Project A into V, separate into Q and R
        

        # Compute reweighting for p-norm approximation
        u = L @ x
        z = smoothed_holder_weights(u, epsilon=0.001, p=pnorm).flatten()**(1/2)
        p = z[:, np.newaxis]
        temp = p * (L @ V)  
        (Q_L, R_L) = la.qr(temp, mode='economic') # Project L into V, separate into Q and R


        lambdah = generalized_crossvalidation(A @ V, b, p * (L @ V) )['x'] # find ideal lambdas by crossvalidation
        
        lambda_history.append(lambdah)

        bhat = (Q_A.T @ b).reshape(-1,1) # Project b

        R_stacked = np.vstack( [R_A]+ [lambdah*R_L] ) # Stack projected operators

        b_stacked = np.vstack([bhat] + [np.zeros(shape=(R_L.shape[0], 1))]) # pad with zeros

        y, _,_,_ = la.lstsq(R_stacked, b_stacked) # get least squares solution

        x = V @ y # project y back
        
        x_history.append(x)

        r = (A @ x).reshape(-1,1) - b.reshape(-1,1) # get residual
        ra = A.T @ r

        rb = lambdah[0] * L.T @ (p * (L @ x))# this likely needs to include information from the pnorm weighting
        r = ra  + rb

        #r = r - V@(V.T@r)
        #r = r - V@(V.T@r)


        normed_r = r / la.norm(r) # normalize residual


        V = np.hstack([V, normed_r]) # add residual to basis

        V, _ = la.qr(V, mode='economic') # orthonormalize basis using QR


    return (x, x_history, lambdah, lambda_history)