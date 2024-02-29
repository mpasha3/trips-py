#!/usr/bin/env python
""" 
Utility functions
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha"
__affiliations__ = 'MIT and Tufts University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com;"

from pylops import LinearOperator
from trips.utilities.operators_old import *
from scipy import linalg as la
from scipy.sparse._arrays import _sparray
from scipy import sparse

import numpy as np

from pylops import Identity, LinearOperator

"""
Utility functions.
"""

def soft_thresh(x, mu):
    #y = np.sign(x)*np.max([np.abs(x)-mu], 0)
    y = np.abs(x) - mu
    y[y < 0] = 0
    y = y * np.sign(x)
    return y

def generate_noise(shape, noise_level, dist='normal'):
    """
    Produces noise at the desired noise level.
    """

    if dist == 'normal':
        noise = np.random.randn(shape)
    elif dist == 'poisson':
        noise = np.random.poisson
    e = noise_level * noise / la.norm(noise)


def is_identity(A):
    """
    Checks whether the operator A is identity.
    """

    if isinstance(A, Identity): # check if A is a pylops identity operator
        return True

    elif (not isinstance(A, LinearOperator)) and ( A.shape[0] == A.shape[1] ) and ( np.allclose(A, np.eye(A.shape[0])) ): # check if A is an array resembling the identity matrix
        return True
    
    elif isinstance(A, _sparray) and ( A.shape[0] == A.shape[1] ) and ( A - sparse.eye(A.shape[0]) ).sum() < 10**(-6):
        return True
    
    else:
        return False

def check_noise_type(noise_type):
    if noise_type in ['g', 'p', 'l', 'gaussian', 'Gaussian', 'Poisson', 'poisson', 'Laplace', 'laplace']:
        valid = True
    else:
        valid = False
    if not valid:
       raise TypeError('You must enter a valid name for the noise. For Gaussian noise input g or Gaussian or gaussian. For Poisson noise input p or Poisson or poisson. For Laplace noise input l or laplace or laplace.')

def check_noise_level(noise_level):
    valid  = False
    if (isinstance(noise_level, float) or isinstance(noise_level, int)):
        if int(noise_level) > 0 or int(noise_level) == 0:
            valid = True
    if not valid:
        raise TypeError('You must enter a valid noise level! Choose 0 for 0 %, 1 for 1%, or other valid values acordingly.')

def check_Regparam(Regparam = 1):
    valid = False
    case1 = False
    # if str(Regparam).isnumeric():
    if (isinstance(Regparam, float) or isinstance(Regparam, int)):
        if int(Regparam) > 0:
            valid = True
        else:
            valid = False
            case1 = True
    elif Regparam in ['gcv', 'GCV', 'Gcv', 'DP', 'dp', 'Dp', 'Discrepancy Principle', 'Discrepancy principle', 'discrepancy principle']:
        valid = True
    if not valid and case1 == True:
        raise TypeError("You must specify a valid regularization parameter. Input a positive number!")
    elif not valid:
        raise TypeError("You must specify a valid regularization parameter. For Generalized Cross Validation type 'gcv'. For 'Discrepancy Principle type 'dp'.")

def check_Positivescalar(value):
    if int(value) > 0:
        valid = True
    else:
        valid = False

def check_operator_type(A):
    aa = str(type(A))
    if 'array' in aa:
        A = A
    # elif 'sparse' in aa:
    else:
        A = A.todense()
    return A

# def check_imagesize_toreshape(existingimage, chooseimage, old_size, newsize):
#     path_package = '/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package'
#     if (old_size[0] != newsize[0] or old_size[1] != newsize[1]):
#         Deblur.plot_rec(existingimage.reshape((shape), order = 'F'), save_imgs = False)
#         temp_im = Image.open(path_package + '/demos/data/images/'+chooseimage+'_'+str(newsize[0])+'.jpg')
#         image_new =  np.array(temp_im.resize((newsize[0], newsize[1])))
#         spio.savemat(path_package + '/demos/data/images/'+chooseimage+'_'+str(newsize[0])+'.mat', mdict={'x_true': image_new})
#     return image_new


def get_input_image_size(image):
    imshape = image.shape
    if imshape[1] == 1:
        nx = int(np.sqrt(imshape[0]))
        ny = int(np.sqrt(imshape[0]))
    else:
        nx = imshape[0]
        ny = imshape[1]
    newshape = (nx, ny)
    return newshape


def check_if_vector(im, nx, ny):
    if im.shape[1] == 1:
        im_vec = im
    else:
        im_vec = im.reshape((nx*ny, 1)) 
    return im_vec

def image_to_new_size(image, n):
    X, Y = np.meshgrid(np.linspace(1, image.shape[1], n[0]), np.linspace(1, image.shape[0], n[1]))
    im = interp2linear(image, X, Y, extrapval=np.nan)
    return im

def interp2linear(z, xi, yi, extrapval=np.nan):
    """
    This function is obtained from this github repository: https://github.com/serge-m/pyinterp2 to be used for automatically reshaping the images
    __author__ = 'Sergey Matyunin'
    Linear interpolation equivalent to interp2(z, xi, yi,'linear') in MATLAB
    @param z: function defined on square lattice [0..width(z))X[0..height(z))
    @param xi: matrix of x coordinates where interpolation is required
    @param yi: matrix of y coordinates where interpolation is required
    @param extrapval: value for out of range positions. default is numpy.nan
    @return: interpolated values in [xi,yi] points
    @raise Exception:
    """
    x = xi.copy()
    y = yi.copy()
    nrows, ncols = z.shape
    if nrows < 2 or ncols < 2:
        raise Exception("z shape is too small")
    if not x.shape == y.shape:
        raise Exception("sizes of X indexes and Y-indexes must match")
    # find x values out of range
    x_bad = ( (x < 0) | (x > ncols - 1))
    if x_bad.any():
        x[x_bad] = 0
    # find y values out of range
    y_bad = ((y < 0) | (y > nrows - 1))
    if y_bad.any():
        y[y_bad] = 0
    # linear indexing. z must be in 'C' order
    ndx = np.floor(y) * ncols + np.floor(x)
    ndx = ndx.astype('int32')
    # fix parameters on x border
    d = (x == ncols - 1)
    x = (x - np.floor(x))
    if d.any():
        x[d] += 1
        ndx[d] -= 1
    # fix parameters on y border
    d = (y == nrows - 1)
    y = (y - np.floor(y))
    if d.any():
        y[d] += 1
        ndx[d] -= ncols
    # interpolate
    one_minus_t = 1 - y
    z = z.ravel()
    f = (z[ndx] * one_minus_t + z[ndx + ncols] * y ) * (1 - x) + (
        z[ndx + 1] * one_minus_t + z[ndx + ncols + 1] * y) * x
    # Set out of range positions to extrapval
    if x_bad.any():
        f[x_bad] = extrapval
    if y_bad.any():
        f[y_bad] = extrapval
    return f

### TODO: Add a general reweighting function