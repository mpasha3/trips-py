#!/usr/bin/env python
""" 
Builds a deblurring class for 2D problems
--------------------------------------------------------------------------
Created in January 2024 for TRIPs-Py library
"""
__developers__ = "Mirjeta Pasha and Silvia Gazzola"
__affiliations__ = 'MIT and Tufts University, University of Bath'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk;"

import time
import numpy as np
import scipy as sp
import scipy.stats as sps
import scipy.io as spio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import astra
import trips.utilities.phantoms as phantom
from venv import create
import pylops
from scipy.ndimage import convolve
from scipy import sparse
import scipy.special as spe
from trips.utilities.operators_old import *
from PIL import Image
from resizeimage import resizeimage
import requests
from os import mkdir
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from trips.utilities.utils import *
import scipy.linalg as la

class Deblurring2D():
    def __init__(self,**kwargs):
        seed = kwargs.pop('seed',2022)
        self.nx = None
        self.ny = None
        self.CommitCrime = kwargs['CommitCrime'] if ('CommitCrime' in kwargs) else False

    def Gauss(self, PSFdim, PSFspread):
        self.m = PSFdim[0]
        self.n = PSFdim[1]
        self.dim = PSFdim
        self.spread = PSFspread
        if type(PSFspread) in [int]:
            self.s1, self.s2 = PSFspread, PSFspread
        else:
            self.s1, self.s2 = PSFspread[0], PSFspread[1]
        x = np.arange(-np.fix(self.n/2), np.ceil(self.n/2))
        y = np.arange(-np.fix(self.m/2), np.ceil(self.m/2))
        X, Y = np.meshgrid(x, y)
        PSF = np.exp( -0.5* ((X**2)/(self.s1**2) + (Y**2)/(self.s2**2)) )
        PSF /= PSF.sum()
        mm, nn = np.where(PSF == PSF.max())
        center = np.array([mm[0], nn[0]])   
        return PSF, center.astype(int)
    
    def forward_Op(self, dim, spread, nx, ny):
        self.nx = nx
        self.ny = ny
        PSF, center = self.Gauss(dim, spread)
        proj_forward = lambda X: convolve(X.reshape([nx,ny]), PSF, mode='reflect').reshape((-1,1))
        proj_backward = lambda B: convolve(B.reshape([nx,ny]), np.flipud(np.fliplr(PSF)), mode='reflect' ).reshape((-1,1))
        blur = pylops.FunctionOperator(proj_forward, proj_backward, nx*ny)
        return blur
    
    def im_image_dat(self, im):
        if exists(f'./data/image_data/{im}.mat'):
            print('data already in the path.')
        else:
            print("Please make sure your data are on the data folder!")
        f = spio.loadmat(f'./data/image_data/{im}.mat')
        X = f['x_true']
        im_shape = X.shape
        if len(im_shape) == 3:
             X = 0.4*X[:, :, 0] + 0.4*X[:, :, 1] + 0.1*X[:, :, 2]
        return X  

    def gen_true_mydata(self, im, **kwargs):
        if (self.nx is None or self.ny is None):
            if (('nx' in kwargs) and ('ny' in kwargs)):
                self.nx = kwargs['nx'] 
                self.ny = kwargs['ny'] 
            else:
                raise TypeError("The dimension of the image is not specified. You can input nx and ny as gen_true(im, nx, ny) or first define the forward operator through A = Deblur.forward_Op_matrix([11,11], nx, ny) or A = Deblur.forward_Op([11,11], 0.7, nx, ny) ")
        image = self.im_image_dat(im)
        current_shape = get_input_image_size(image)
        if ((current_shape[0] is not self.nx) and (current_shape[1] is not self.ny)):
            newimage = image_to_new_size(image, (self.nx, self.ny))
            newimage[np.isnan(newimage)] = 0
        return newimage

    def gen_true(self, im, **kwargs):
        if (self.nx is None or self.ny is None):
            if (('nx' in kwargs) and ('ny' in kwargs)):
                self.nx = kwargs['nx'] 
                self.ny = kwargs['ny'] 
            else:
                raise TypeError("The dimension of the image is not specified. You can input nx and ny as gen_true(im, nx, ny) or first define the forward operator through A = Deblur.forward_Op_matrix([11,11], nx, ny) or A = Deblur.forward_Op([11,11], 0.7, nx, ny) ")
        if im in ['satellite', 'hubble', 'h_im','shape']:
            image = self.im_image_dat(im)
            newimage = image
            current_shape = get_input_image_size(image)
            if ((current_shape[0] is not self.nx) and (current_shape[1] is not self.ny)):
                newimage = image_to_new_size(image, (self.nx, self.ny))
                newimage[np.isnan(newimage)] = 0
        else:
            raise ValueError("The image you requested does not exist! Specify the right name. Options are 'satellite', 'hubble', 'h_im")
        return newimage

    def gen_data(self, x):
        im = x.reshape((self.nx, self.ny))
        if self.CommitCrime == False:
            nxbig = 2*self.nx
            nybig = 2*self.ny
            im = x.reshape((self.nx, self.ny))
            padim = np.zeros((nxbig, nybig))
            putidx = self.nx//2
            putidy = self.ny//2
            padim[putidx:(putidx+self.nx), putidy:(putidy+self.ny)] = im
            PSF, _ = self.Gauss(self.dim, self.spread)
            A0 = lambda X: convolve(X.reshape([nxbig,nybig]), PSF, mode='constant')
            b = A0(padim)
            x = padim[putidx:(putidx+self.nx), putidy:(putidy+self.ny)].reshape((-1,1))
            b = b[putidx:(putidx+self.nx), putidy:(putidy+self.ny)].reshape((-1,1))
        else:
            PSF, _ = self.Gauss(self.dim, self.spread)
            A = lambda X: convolve(X, PSF, mode='reflect')
            b = A(im)
            b = b.reshape((-1,1))
        return b
        
    def add_noise(self, b_true, opt, noise_level):
        if (opt == 'Gaussian'):
            e = np.random.randn(self.nx*self.ny, 1)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
            delta = np.linalg.norm(sig_obs*e)
            b_meas_im = b_meas.reshape((self.nx, self.ny))
        if (opt == 'Poisson'):
            gamma = 1 
            b_meas = np.random.poisson(lam=b_true+gamma) 
            b_meas_im = b_meas.reshape((self.nx, self.ny))
            e = 0
            delta = np.linalg.norm(e)
        if (opt == 'Laplace'):
            e = np.random.laplace(self.nx*self.ny, 1)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
            delta = np.linalg.norm(sig_obs*e)
            b_meas_im = b_meas.reshape((self.nx, self.ny))
        return (b_meas_im, delta)
    
    def vec(self, image):
        sh = image.shape
        return image.reshape((sh[0]*sh[1]))
    ## convert a 1-d vector into a 2-d image of the given shape
    def im(self, x, shape):
        return x.reshape(shape)
    ## display a 1-d vector as a 2-d image
    def display_vec(self, vec, shape, scale = 1):
        image = self.im(vec, shape)
        plt.imshow(image, vmin=0, vmax=scale * np.max(vec), cmap='gray')
        plt.axis('off')
        plt.show()
        ## a helper function for creating the blurring operator
    def get_column_sum(self, spread):
        length = 40
        raw = np.array([np.exp(-(((i-length/2)/spread[0])**2 + ((j-length/2)/spread[1])**2)/2) 
                        for i in range(length) for j in range(length)])
        return np.sum(raw[raw > 0.0001])
    ## blurs a single pixel at center with a specified Gaussian spread
    def P(self, spread, center, shape):
        image = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                v = np.exp(-(((i-center[0])/spread[0])**2 + ((j-center[1])/spread[1])**2)/2)
                if v < 0.0001:
                    continue
                image[i,j] = v
        return image
    
    def plot_rec(self, img, save_imgs = False, save_path='./saveImagesDeblurringReconstructions'):
            plt.set_cmap('inferno')
            if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
            plt.imshow(img.reshape((self.nx, self.ny)))
            plt.axis('off')
            if save_imgs:  plt.savefig(save_path+'/rec'+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw()

    def plot_data(self, img, save_imgs = False, save_path='./saveImagesDeblurringData'):
            plt.set_cmap('inferno')
            if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
            plt.imshow(img.reshape((self.nx, self.ny)))
            plt.axis('off')
            if save_imgs:  plt.savefig(save_path+'/rec'+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw() 

if __name__ == '__main__':
    Deblur = Deblurring2D()

    

   
