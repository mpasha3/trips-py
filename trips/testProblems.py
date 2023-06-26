#!/usr/bin/env python
"""
Definition of test problems
--------------------------------------------------------------------------
Created december 10, 2022 for TRIPs-Py library
"""
__author__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com"
# import sys, os
# sys.path.insert(0,'/Users/mirjetapasha/Documents/Research_Projects/TRIPSpy/TRIPSpy')
import time
import numpy as np
import scipy as sp
import scipy.stats as sps
import scipy.io as spio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import astra
# from data.phantoms import *
from venv import create
import pylops
from scipy.ndimage import convolve
from scipy import sparse
from scipy.ndimage import convolve
import scipy.special as spe
from trips.operators import *
from PIL import Image
from resizeimage import resizeimage
class Deblurring:
    def __init__(self,**kwargs):
        seed = kwargs.pop('seed',2022)
    def Gauss(self, PSFdim, PSFspread):
        self.m = PSFdim[0]
        self.n = PSFdim[1]
        self.dim = PSFdim
        self.spread = PSFspread
        self.s1, self.s2 = PSFspread, PSFspread
        # Set up grid points to evaluate the Gaussian function
        x = np.arange(-np.fix(self.n/2), np.ceil(self.n/2))
        y = np.arange(-np.fix(self.m/2), np.ceil(self.m/2))
        X, Y = np.meshgrid(x, y)
        # Compute the Gaussian, and normalize the PSF.
        PSF = np.exp( -0.5* ((X**2)/(self.s1**2) + (Y**2)/(self.s2**2)) )
        PSF /= PSF.sum()
        # find the center
        mm, nn = np.where(PSF == PSF.max())
        center = np.array([mm[0], nn[0]])
        return PSF, center.astype(int)

    def forward_Op(self, dim, spread, nx, ny):
        PSF, center = self.Gauss(dim, spread)
        proj_forward = lambda X: convolve(X.reshape([nx,ny]), PSF, mode='constant').flatten()
        proj_backward = lambda B: convolve(B.reshape([nx,ny]), np.flipud(np.fliplr(PSF)), mode='constant' ).flatten()
        blur = pylops.FunctionOperator(proj_forward, proj_backward, nx*ny)
        return blur

    def generate_true(self, choose_image):
        # Specify the path
        path_package = '/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package'
        if choose_image == 'satellite128':
            dataa = spio.loadmat(path_package + '/demos/data/images/satellite128.mat')
            X = dataa['x_true']
            X_true = X/X.max()
            self.nx, self.ny = X_true.shape  
            x_truef = X_true.flatten(order = 'F')
        elif choose_image == 'satellite64':
            dataa = spio.loadmat(path_package + '/demos/data/images/satellite64.mat')
            X = dataa['x_new']
            X_true = X/X.max()
            self.nx, self.ny = X_true.shape  
            x_truef = X_true.flatten(order = 'F')
        elif choose_image == 'edges':
            dataa = spio.loadmat(path_package + '/demos/data/images/edges.mat')
            X = dataa['x']
            X_true = X/X.max()
            self.nx, self.ny = X_true.shape  
            x_truef = X_true.flatten(order = 'F')
        elif choose_image == 'pattern1':
            dataa = spio.loadmat(path_package + '/demos/data/images/shape1.mat')
            X = dataa['xtrue']
            X_true = X/X.max()
            self.nx, self.ny = X_true.shape  
            x_truef = X_true.flatten(order = 'F')
        elif choose_image == 'Himage':
            dx = 10
            dy = 10
            up_width = 10
            bar_width= 5
            size = 64
            self.nx, self.ny = 64, 64
            h_im = np.zeros((size, size))
            for i in range(size):
                if i < dy or i > size-dy:
                    continue
                for j in range(size):
                    if j < dx or j > size - dx:
                        continue
                    if j < dx + up_width or j > size - dx - up_width:
                        h_im[i, j] = 1
                    if abs(i - size/2) < bar_width:
                        h_im[i, j] = 1
            x_truef = self.vec(h_im)
            # X_true = h_im
        else:
            raise ValueError("The image you requested does not exist! Specify the right name.")
        return (x_truef, self.nx, self.ny)
        ## convert a 2-d image into a 1-d vector
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

    def forward_Op_matrix(self, spread, shape, nx, ny):
        ## construct our blurring matrix with a Gaussian spread and zero boundary conditions
        #normalize = get_column_sum(spread)
        m = shape[0]
        n = shape[1]
        A = np.zeros((m*n, m*n))
        count = 0
        self.spread = spread
        self.shape = shape
        for i in range(m):
            for j in range(n):
                column = self.vec(self.P(spread, [i, j],  shape))
                A[:, count] = column
                count += 1
        normalize = np.sum(A[:, int(m*n/2 + n/2)])
        A = 1/normalize * A
        return A

    def generate_data(self, x, matrix):
        if matrix == 'False':
            A = self.forward_Op(self.dim, self.spread, self.nx, self.ny)
            b = A*x
        else:
            A = self.forward_Op_matrix(self.spread, self.shape, self.nx, self.ny)
            b = A@x
        return b
        
    def add_noise(self, b_true, opt, noise_level):
        if (opt == 'Gaussian'):
            mu_obs = np.zeros(self.nx*self.ny)      # mean of noise
            e = np.random.randn(self.nx*self.ny)
            delta = np.linalg.norm(e)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
            b_meas_im = b_meas.reshape((self.nx, self.ny), order='F')
        if (opt == 'Poisson'):
            gamma = 1 # background counts assumed known
            b_meas = np.random.poisson(lam=b_true+gamma) 
            b_meas_im = b_meas.reshape((self.nx, self.ny), order='F')
            e = 0
            delta = np.linalg.norm(e)
        if (opt == 'Laplace'):
            mu_obs = np.zeros(self.nx*self.ny)      # mean of noise
            e = np.random.laplace(self.nx*self.ny)
            delta = np.linalg.norm(e)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
            b_meas_im = b_meas.reshape((self.nx, self.ny), order='F')
        return (b_meas_im, delta)
          
    def Tikh_sol(A, b_vec, L, x_true):
        lamb = 1/5
        minerror = 100
        while lamb > 1e-04:
            xTik = np.linalg.solve(A.T@A + lamb**2*L.T@L, A.T@b_vec)
            error = np.linalg.norm(xTik - x_true)/np.linalg.norm(x_true)
            if error < minerror:
                minerror = error
                minlamb = lamb
                mingTik = xTik
            lamb /= 2
        return mingTik  
      
    def plot_rec(self, img, save_imgs = False, save_path='./saveImagesDeblurring'):
        import matplotlib.pyplot as plt
        import os, sys
        plt.set_cmap('inferno')
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        # plt.imshow(img.reshape((self.nx, self.ny), order = 'F'))
        plt.imshow(img.reshape((self.nx, self.ny)))
        plt.axis('off')
        if save_imgs:  plt.savefig(save_path+'/rec'+'.png',bbox_inches='tight')
        plt.pause(.1)
        plt.draw()

class Tomography:
    def __init__(self,**kwargs):
        seed = kwargs.pop('seed',2022)
        # self.setup(seed,**kwargs)
    def set_up(self, sizex, sizey):
        # self.nx = 64          # object size nx-by-nx pixels
        # self.ny = 64
        self.nx = sizex
        self.ny = sizey
        self.p = int(np.sqrt(2)*self.nx)    # number of detector pixels
        self.q = 180            # number of projection angles
        self.theta = np.linspace(0, 2*np.pi, self.q, endpoint=False)   # in rad
        self.source_origin = 3*self.nx                     # source origin distance [cm]
        self.detector_origin = self.nx                      # origin detector distance [cm]
        self.detector_pixel_size = (self.source_origin + self.detector_origin)/self.source_origin
        self.detector_length = self.detector_pixel_size*self.p   # detector length
        self.vol_geom = astra.create_vol_geom(self.nx,self.nx)
        self.proj_geom = astra.create_proj_geom('fanflat', self.detector_pixel_size, self.p, self.theta, self.source_origin, self.detector_origin)
        self.proj_id = astra.create_projector('line_fanflat', self.proj_geom, self.vol_geom)
        return self.proj_geom, self.proj_id

    def forward_Op_mat(self, sizex, sizey):
        proj_geom, proj_id = self.set_up(sizex, sizey)
        self.mat_id = astra.projector.matrix(proj_id)
        return astra.matrix.get(self.mat_id) 

    def forward_Op(self, sizex, sizey): 
        proj_geom, proj_id = self.set_up(sizex, sizey)       
        return astra.OpTomo(self.proj_id)

    def generate_true(self, test_problem):
        if test_problem == 'grains':
            N_fine = self.nx
            numGrains = int(round(4*np.sqrt(N_fine)))
            x_true = grains(N_fine, numGrains) 
            idd = np.where(x_true<0)
            x_true[idd] = 0
            x_truef = x_true.flatten(order='F') 
            xx_norm = np.linalg.norm(x_truef)
        else:
            data = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPSpy/TRIPSpy/data/xx.mat')
            X = data['xx']
            X_true = X/X.max()
            # X_true = resizeimage.resize_cover(X_true, [64, 64], validate=False)
            # X_true = X_true.resize(64, 64)
            self.nx, self.ny = X_true.shape 
            x_truef = X_true.flatten(order='F') 
        return x_truef
    def generate_data(self, x, proj_type, flag):
        if flag == 'simulated':
            if proj_type == 1:
                # forward projection
                _, b = astra.create_sino(x.reshape((self.nx,self.nx), order='F'), self.proj_id)
                b = np.fliplr(b)
                out = b.flatten()
            elif proj_type == 2:
                # backward projection   
                b = np.fliplr(x.reshape((self.q, self.p)))
                _, ATb = astra.create_backprojection(b, self.proj_id)
                b = ATb.flatten(order='F')
        else: ##MP: Will develop the true data case here
            if proj_type == 1:
                # forward projection
                _, Ax = astra.create_sino(x.reshape((self.nx,self.nx), order='F'), self.proj_id)
                Ax = np.fliplr(Ax)
                out = Ax.flatten()
            elif proj_type == 2:
                # backward projection   
                b = np.fliplr(x.reshape((self.q, self.p)))
                _, ATb = astra.create_backprojection(b, self.proj_id)
                b = ATb.flatten(order='F')
        return out
    def add_noise(self, b_true, opt, noise_level):
        if (opt == 'Gaussian'):
            mu_obs = np.zeros(self.p*self.q)      # mean of noise
            # # e=e/norm(e)*norm(b)*sigma
            # e = np.random.randn(self.p*self.q)
            # e = e/np.linalg.norm(e)*np.linalg.norm(b_true)*noise_level
            noise = np.random.randn(b_true.shape[0]).reshape(-1,1)
            e = noise_level * np.linalg.norm(b_true) / np.linalg.norm(noise) * noise
            e = e.reshape(-1,1)
            b_true = b_true.reshape(-1,1)
            b = b_true + e # add noise
            # sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            # sig_obs = err_lev * np.linalg.norm(b_true)/np.sqrt(m)             # std of noise
            # e_true = np.random.normal(loc=mu_obs, scale=sig_obs*np.ones(m))   # noise
            # b_meas = b_true + e_true                                          # 'measured' data
            b_meas = b_true + e
            b_meas_i = b_meas.reshape((self.p, self.q), order='F')
        elif (opt == 'Poisson'):
            # Add Poisson Noise 
            gamma = 1 # background counts assumed known
            b_meas = np.random.poisson(lam=b_true+gamma) 
            b_meas_i = b_meas.reshape((self.p, self.q), order='F')
        else:
            mu_obs = np.zeros(self.p*self.q)      # mean of noise
            e = np.random.laplace(self.p*self.q)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
            b_meas_i = b_meas.reshape((self.p, self.q), order='F')
        return (b_meas_i , e)

    def plot_rec(self, img, save_imgs=True, save_path='./saveImagesTomo'):
            import matplotlib.pyplot as plt
            import os, sys
            plt.set_cmap('inferno')
            if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
            plt.imshow(img.reshape((self.nx, self.ny), order = 'F'))
            plt.axis('off')
            if save_imgs:  plt.savefig(save_path+'/rec'+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw()
        
    def plot_sino(self, img, save_imgs = True, save_path='./saveImagesTomo'):
        import matplotlib.pyplot as plt
        import os, sys
        plt.set_cmap('inferno')
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        plt.imshow(img.reshape((self.p, self.q), order = 'F'))
        plt.axis('off')
        if save_imgs:  plt.savefig(save_path+'/sino'+'.png',bbox_inches='tight')
        plt.pause(.1)
        plt.draw()

if __name__ == '__main__':
    # Test Deblurring class
    from solvers.gks import *
    Deblur = Deblurring()
    generate_matrix = True
    imagesize_x = 64
    imagesize_y = 64
    spread = 1.5
    choose_image = 'Himage'
    if generate_matrix == True:
        # spread = [2,2]
        size = imagesize_x
        shape = (size, size)
        spreadnew = (spread, spread)
        A = Deblur.forward_Op_matrix(spreadnew, shape, imagesize_x, imagesize_y)
    x_true = Deblur.generate_true(choose_image)
    b_true = Deblur.generate_data(x_true, generate_matrix)
    b = Deblur.add_noise(b_true, 'Gaussian', noise_level = 0.01)
    Deblur.plot_rec(x_true, save_imgs = True, save_path='./saveImagesDeblurring')
    b_vec = b.reshape((-1,1))
    L = spatial_derivative_operator(imagesize_x, imagesize_y, 1)
    xhat = GKS(A, b_vec, L, 3, 4)
    # GKS(A, b, L, projection_dim=3, iter=50, selection_method = 'gcv', **kwargs):
    # Test Tomography class
    Tomo = Tomography()
    Amat = Tomo.forward_Op_mat()
    A = Tomo.forward_Op
    x_true = Tomo.generate_true()
    b_true = Tomo.generate_data(x_true, 1, 'simulated')
    b = Tomo.add_noise(b_true, 'Gaussian', noise_level= 0.01)
    Tomo.plot_rec(x_true)
    Tomo.plot_sino(b)
    

   
