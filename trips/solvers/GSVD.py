
import numpy as np
from scipy.linalg import toeplitz
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.optimize import fminbound
from trips.utilities.decompositions import gsvd

def differential_operators(ncol, order):
    if order == 1:
        mn = [np.ones(ncol + 1), -np.ones(ncol)]
        offset = [0, 1]
        Ds = 1/2 * diags(mn, offset).toarray()
    elif order == 2:
        mn = [-np.ones(ncol), 2 * np.ones(ncol + 1), -np.ones(ncol)]
        offset = [-1, 0, 1]
        Ds = 1/4 * diags(mn, offset).toarray() 
    elif order == 'sq1':
        mn = [np.ones(ncol + 1), -np.ones(ncol)]
        offset = [0, 1]
        Ds = 1/2 * diags(mn, offset).toarray()
    elif order == 'sq2':
        mn = [-np.ones(ncol), 2 * np.ones(ncol + 1), -np.ones(ncol)]
        offset = [-1, 0, 1]
        Ds = 1/4 * diags(mn, offset).toarray()
    return Ds[:-1, :-1]  

def basic_blurring_operators(N,sigma,band,type):
    z = np.concatenate([np.exp(-(np.arange(band) ** 2) / (2 * sigma ** 2)), np.zeros(N - band)])
    if type == 'bccb':
        z_flipped = np.flip(z[1:])
        A = (1 / (sigma * np.sqrt(2 * np.pi))) * toeplitz(np.concatenate([z[:1], z_flipped]), z)
    elif type == 'bctb':
        zz = np.concatenate([z[:1], np.flip(z[-len(z) + 1:])])
        A = (1 / np.sqrt(2 * np.pi * sigma)) * toeplitz(z, zz)
    elif type == 'blur':
        A = (1 / np.sqrt(2 * np.pi * sigma)) * toeplitz(z)
    return np.kron(A,A)

def simple_plot(R,C,S):
    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1), plt.title('Generalized Singular Values of $A$ and $L$'),
    plt.plot(R,'-*r', lw=2,ms=10), plt.xlabel('$\ell$',fontsize=16), plt.ylabel('$\gamma_\ell$',fontsize=16),
    plt.minorticks_on(), plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')

    plt.subplot(1,3,2), plt.title('Diagonal entries of $C$'),
    plt.plot(np.diag(C),'-*r', lw=2,ms=10), plt.xlabel('$\ell$',fontsize=16), plt.ylabel('$\sigma_\ell$',fontsize=16),
    plt.minorticks_on(), plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')

    plt.subplot(1,3,3), plt.title('Diagonal entries of $S$'),
    plt.plot(np.diag(S),'-*r', lw=2,ms=10), plt.xlabel('$\ell$',fontsize=16), plt.ylabel('$\mu_\ell$',fontsize=16),
    plt.minorticks_on(), plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')

def gsvd_plot(x_true,b_true,b,x_gsvd,x_tgsvd,N):
    plt.figure(figsize=(15,4))
    plt.subplot(1,5,1), plt.title('x_true'), plt.axis('off'), plt.imshow(x_true)
    plt.subplot(1,5,2), plt.title('b_true'), plt.axis('off'), plt.imshow(b_true.reshape((N,N)))
    plt.subplot(1,5,3), plt.title('b'), plt.axis('off'), plt.imshow(b.reshape((N,N)))
    plt.subplot(1,5,4), plt.title('x_gsvd'), plt.axis('off'), plt.imshow(x_gsvd.reshape((N,N)))
    plt.subplot(1,5,5), plt.title('x_tgsvd'), plt.axis('off'), plt.imshow(x_tgsvd.reshape((N,N)))
        

def tgsvd_tik_sol(A,L,b_vec,mu,k):
    U, _, Z, C, S = gsvd(A,L) 
    Y = np.linalg.inv(Z.T)[:,0:k]
    CC = C[0:k,0:k]
    SS = S[0:k,0:k]
    UU = U.T[0:k,:]
    xsol = Y@np.linalg.inv(CC.T@CC+1/mu*SS.T@SS)@CC.T@(UU@b_vec)
    return xsol

def gsvd_tik_sol(A,L,b_vec,mu):
    U, _, Z, C, S = gsvd(A,L) 
    Y = np.linalg.inv(Z.T)
    xsol = Y@np.linalg.inv(C.T@C+1/mu*S.T@S)@C.T@(U.T@b_vec)
    return xsol

def gcvd(A,L,b):
    U, _, _, C, S = gsvd(A,L)
    bhat = U.T @ b
    c = np.diag(C)
    s = np.diag(S)
    mu = fminbound(gcv, 0, 100, args=(c, s, bhat))
    # mu = op.fminbound(func = gcv, x1 = 0, x2 = 1e2, args=(c,s,bhat),
    # xtol=1e-12, maxfun=1000, full_output=0, disp=0)
    return mu

def gcv(mu, c, s, bhat):
    num = (s**2 * bhat / (c**2 + mu * s**2))**2
    num = np.sum(num)
    den = (s**2 / (c**2 + mu * s**2))**2
    den = np.sum(den)
    G = num / den
    return G


