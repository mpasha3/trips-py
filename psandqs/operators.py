"""
Functions which implement operators for measurement or regularization.
"""

import numpy as np

import pylops

from scipy.ndimage import convolve

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


    breakpoint()