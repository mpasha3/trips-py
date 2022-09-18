"""
Functions which implement operators for measurement or regularization.
"""

from venv import create
import numpy as np
import pylops
from scipy.ndimage import convolve

from scipy import sparse



"""derivative operators"""

def first_derivative_operator(n):

    L = pylops.FirstDerivative(n, dtype="float32")

    return L



def first_derivative_operator_2d(nx, ny):

    D_x = first_derivative_operator(nx)
    D_y = first_derivative_operator(ny)

    IDx = pylops.Kronecker( pylops.Identity(nx, dtype='float32'), D_x )
    DyI = pylops.Kronecker( D_y, pylops.Identity(ny, dtype='float32') )

    D_spatial = pylops.VStack((IDx, DyI))

    return D_spatial



def spatial_derivative_operator(nx, ny, nt):

    D_spatial = first_derivative_operator_2d(nx,ny)

    ID_spatial = pylops.Kronecker( pylops.Identity(nt, dtype='float32'), D_spatial)

    return ID_spatial

def time_derivative_operator(nx, ny, nt):
    
    D_time = first_derivative_operator(nt)

    D_timeI = pylops.Kronecker( D_time, pylops.Identity(nx**2, dtype='float32') )

    return D_timeI
    


"""blur operators"""

def Gauss(dim, s): # Dr. Pasha's Gaussian PSF code

    if hasattr(dim, "__len__"):
        m, n = dim[0], dim[1]
    else:
        m, n = dim, dim
    s1, s2 = s, s
    
    # Set up grid points to evaluate the Gaussian function
    x = np.arange(-np.fix(n/2), np.ceil(n/2))
    y = np.arange(-np.fix(m/2), np.ceil(m/2))
    X, Y = np.meshgrid(x, y)

    # Compute the Gaussian, and normalize the PSF.
    PSF = np.exp( -0.5* ((X**2)/(s1**2) + (Y**2)/(s2**2)) )
    PSF /= PSF.sum()

    # find the center
    mm, nn = np.where(PSF == PSF.max())
    center = np.array([mm[0], nn[0]])

    return PSF, center.astype(int)


def gaussian_blur_operator(dim, spread, nx, ny):



    PSF, center = Gauss(dim, spread)

    proj_forward = lambda X: convolve(X.reshape([nx,ny]), PSF, mode='constant').flatten()

    proj_backward = lambda B: convolve(B.reshape([nx,ny]), np.flipud(np.fliplr(PSF)), mode='constant' ).flatten()
    
    blur = pylops.FunctionOperator(proj_forward, proj_backward, nx*ny)

    return blur


"""Framelet operators"""

def construct_H(l,n):

    e = np.ones((n,))

    # build H_0

    H_0 = sparse.spdiags(e, -1-l+1, n, n) + sparse.spdiags(2*e, 0, n, n) + sparse.spdiags(e, 1+l-1, n, n)
    H_0 = H_0.tocsr()

    for jj in range(0,l):
        H_0[jj, l-jj-1] += 1
        H_0[-jj-1, -l+jj] += 1

    H_0 /= 4

    # build H_1

    H_1 = sparse.spdiags(-e, -1-l+1, n, n) + sparse.spdiags(e, 1+l-1, n, n)
    H_1 = H_1.tocsr()

    for jj in range(0,l):
        H_1[jj, l-jj-1] -= 1
        H_1[-jj-1, -l+jj] += 1

    H_1 *= np.sqrt(2)/4

    # build H_2

    H_2 = sparse.spdiags(-e, -1-l+1, n, n) + sparse.spdiags(2*e, 0, n, n) + sparse.spdiags(-e, 1+l-1, n, n)
    H_2 = H_2.tocsr()

    for jj in range(0,l):
        H_2[jj, l-jj-1] -= 1
        H_2[-jj-1, -l+jj] -= 1

    H_2 /= 4

    return (H_0, H_1, H_2)


def create_analysis_operator_rec(n, level, l, w):

    if level == l:
        return sparse.vstack( construct_H(level, n) )

    else:
        (H_0, H_1, H_2) = construct_H(level, n)
        W_1 = create_analysis_operator_rec(n, level+1, l, H_0)

        return sparse.vstack( (W_1, H_1, H_2) ) * w


def create_analysis_operator(n, l):

    return create_analysis_operator_rec(n, 1, l, 1)


def create_framelet_operator(n,m,l):

    W_n = create_analysis_operator(n, l)
    W_m = create_analysis_operator(m, l)

    proj_forward = lambda x: (W_n @ (x.reshape(n,m, order='F') @ W_m.H)).reshape(-1,1, order='F')

    proj_backward = lambda x: (W_n.H @ (x.reshape( n*(2*l+1) , m*(2*l+1), order='F' ) @ W_m)).reshape(-1,1, order='F')

    W = pylops.FunctionOperator(proj_forward, proj_backward, n*(2*l+1) * m*(2*l+1), n*m)

    return W



if __name__ == "__main__":

    print(first_derivative_operator_2d(10,10))

    L_spatial = spatial_derivative_operator(10,10,10)

    L_time = time_derivative_operator(10,10,10)

    print(L_time)

    print(L_spatial)

    arr = np.random.rand(1000,1)

    arr2 = np.random.rand(1000,1000)

    print(L_spatial @ arr)

    blur = gaussian_blur_operator([5,5], 2, 1000, 1000)

    thing = create_framelet_operator(10, 10, 2)

    out = thing @ np.eye(10).reshape(-1,1, order='F')

    back = thing.T @ out

    (H0, H1, H2) = construct_H(2,10)

    output = H2 @ np.eye(10)


    breakpoint()