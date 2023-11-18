#!/usr/bin/env python
"""
Definition of test problems
--------------------------------------------------------------------------
Created December 10, 2022 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha and Connor Sanderford"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com and csanderf@asu.edu; connorsanderford@gmail.com"
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
import trips.phantoms as phantom
from venv import create
import pylops
from scipy.ndimage import convolve
from scipy import sparse
import scipy.special as spe
from trips.operators import *
from PIL import Image
from resizeimage import resizeimage
import requests
from os import mkdir
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from trips.utils import *

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
        self.nx = nx
        self.ny = ny
        PSF, center = self.Gauss(dim, spread)
        proj_forward = lambda X: convolve(X.reshape([nx,ny]), PSF, mode='constant').reshape((-1,1))
        proj_backward = lambda B: convolve(B.reshape([nx,ny]), np.flipud(np.fliplr(PSF)), mode='constant' ).reshape((-1,1))
        blur = pylops.FunctionOperator(proj_forward, proj_backward, nx*ny)
        return blur
    
    def im_image_dat(self, im):
        assert im in ['satellite', 'hubble', 'star', 'h_im']
        if exists(f'./data/image_data/{im}.mat'):
            print('data already in the path.')
        else:
            print("Please make sure your data are on the data folder!")
        f = spio.loadmat(f'./data/image_data/{im}.mat')
        X = f['x_true']
        return X

    # def image_to_new_size(self, image, n):
    #     X, Y = np.meshgrid(np.linspace(1, image.shape[1], n[0]), np.linspace(1, image.shape[0], n[1]))
    #     im = interp2linear(image, X, Y, extrapval=np.nan)
    #     return im

    def gen_true(self, im):
        if im in ['satellite', 'hubble', 'h_im']:
            image = self.im_image_dat(im)
            current_shape = get_input_image_size(image)
            if ((current_shape[0] is not self.nx) and (current_shape[1] is not self.ny)):
                newimage = image_to_new_size(image, (self.nx, self.ny))
        else:
            raise ValueError("The image you requested does not exist! Specify the right name. Options are 'satellite', 'hubble', 'h_im")
        return newimage
   
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
        self.nx = nx
        self.ny = ny
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

    def gen_data(self, x, matrix):
        if matrix == False:
            A = self.forward_Op(self.dim, self.spread, self.nx, self.ny)
            x = check_if_vector(x_true, self.nx, self.ny)
            b = A*x
        else:
            A = self.forward_Op_matrix(self.spread, self.shape, self.nx, self.ny)
            x = check_if_vector(x_true, self.nx, self.ny)
            b = A@x
        
    def add_noise(self, b_true, opt, noise_level):
        if (opt == 'Gaussian'):
            e = np.random.randn(self.nx*self.ny, 1)
            delta = np.linalg.norm(e)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
            b_meas_im = b_meas.reshape((self.nx, self.ny))
        if (opt == 'Poisson'):
            gamma = 1 # background counts assumed known
            b_meas = np.random.poisson(lam=b_true+gamma) 
            b_meas_im = b_meas.reshape((self.nx, self.ny))
            e = 0
            delta = np.linalg.norm(e)
        if (opt == 'Laplace'):
            e = np.random.laplace(self.nx*self.ny, 1)
            delta = np.linalg.norm(e)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
            b_meas_im = b_meas.reshape((self.nx, self.ny), order='F')
        return (b_meas_im, delta)
    def plot_rec(self, img, save_imgs = False, save_path='./saveImagesDeblurringReconstructions'):
            plt.set_cmap('inferno')
            if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
            # plt.imshow(img.reshape((self.nx, self.ny), order = 'F'))
            plt.imshow(img.reshape((self.nx, self.ny)))
            plt.axis('off')
            if save_imgs:  plt.savefig(save_path+'/rec'+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw()

    def plot_data(self, img, save_imgs = False, save_path='./saveImagesDeblurringData'):
            plt.set_cmap('inferno')
            if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
            # plt.imshow(img.reshape((self.nx, self.ny), order = 'F'))
            plt.imshow(img.reshape((self.nx, self.ny)))
            plt.axis('off')
            if save_imgs:  plt.savefig(save_path+'/rec'+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw() 

class Tomography():
    def __init__(self,**kwargs):
        seed = kwargs.pop('seed',2022)
    def define_proj_id(self, sizex, sizey, views, **kwargs):
        self.dataset = kwargs['dataset'] if ('dataset' in kwargs) else False
        self.nx = sizex
        self.ny = sizey
        self.p = int(np.sqrt(2)*self.nx)    # number of detector pixels
        self.q = views           # number of projection angles
        self.views = views
        self.theta = np.linspace(0, 2*np.pi, self.q, endpoint=False)   # in rad
        self.source_origin = 3*self.nx                     # source origin distance [cm]
        self.detector_origin = self.nx                      # origin detector distance [cm]
        self.detector_pixel_size = (self.source_origin + self.detector_origin)/self.source_origin
        self.detector_length = self.detector_pixel_size*self.p   # detector length
        self.vol_geom = astra.create_vol_geom(self.nx,self.nx)
        self.proj_geom = astra.create_proj_geom('fanflat', self.detector_pixel_size, self.p, self.theta, self.source_origin, self.detector_origin)
        self.proj_id = astra.create_projector('line_fanflat', self.proj_geom, self.vol_geom)
        return self.proj_id

    def define_A(self, sizex, sizey, views): 
            proj_id = self.define_proj_id(sizex, sizey, views)  
            self.A = astra.OpTomo(self.proj_id)     
            return self.A

    def forward_Op(self, x, sizex, sizey, views):
        A = self.define_A(sizex, sizey, views)
        operatorf = lambda X: (A*X.reshape((sizex, sizey))).reshape(-1,1)
        operatorb = lambda B: A.T*B.reshape((self.p, self.q))
        OP = pylops.FunctionOperator(operatorf, operatorb, self.p*self.q, sizex*sizey)
        return OP, A

    def forward_Op_mat(self, sizex, sizey, views):
        proj_id = self.define_proj_id(sizex, sizey, views)
        self.mat_id = astra.projector.matrix(proj_id)
        self.Amat = astra.matrix.get(self.mat_id) 
        return self.Amat
    
    def gen_true(self, sizex, sizey, test_problem):
        if test_problem == 'grains':
            N_fine = sizex
            numGrains = int(round(4*np.sqrt(N_fine)))
            x_true = phantom.grains(N_fine, numGrains) 
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'smooth':
            N_fine = sizex
            x_true = phantom.smooth(N_fine) 
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'tectonic':
            N_fine = sizex
            x_true = phantom.tectonic(N_fine)
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1] 
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'threephases':
            N_fine = sizex
            x_true = phantom.threephases(N_fine) 
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'ppower':
            N_fine = sizex
            x_true = phantom.ppower(N_fine) 
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'CT60':
            data = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/Shepp-Logan_proj60_SNR100.mat')
            x_true = data['x_true']
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'CT90':
            data = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/Shepp-Logan_proj90_SNR100.mat')
            x_true = data['x_true']
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'head':
            dataname = 'head'
            data = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/'+ dataname +'.mat')
            x_true = data['x_true']
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1))  
        else:
            raise TypeError("You must enter a valid test problem! Options are: grains, smooth, tectonic, threephases, ppower, CT60, CT90, head.")
        return (x_truef, self.nx, self.ny)

    def gen_saved_data(self, dataset):
        if dataset == 60:
            CT = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/CT_x128_proj60_loc100.mat')
            A, phi, s = CT['A'],CT['phi'],CT['s']
            data = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/Shepp-Logan_proj60_SNR100.mat')
            x_true = data['x_true']
            b = data['b'].T#A*x_true.reshape((-1,1))
            self.q = phi.shape[1]
            self.p = s.shape[1]
        elif dataset == 90:
            CT = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/CT_x128_proj90_loc100.mat')
            A, phi, s = CT['A'],CT['phi'],CT['s']
            data = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/Shepp-Logan_proj90_SNR100.mat')
            x_true = data['x_true']
            b = data['b'].T#A*x_true.reshape((-1,1))
            self.q = phi.shape[1]
            self.p = s.shape[1]
        elif dataset == 'head':
            CT = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/CT_x512_proj200_loc512.mat')
            A,phi,s = CT['A'],CT['phi'],CT['s']
            data = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/'+'head'+'.mat')
            x_true,b = data['x_true'], data['b']
            b = b.T
        return (A, x_true, b)

    def gen_data(self, x, matrix):
        proj_id = self.define_proj_id(self.nx, self.ny, self.views)
        if matrix == True:
            A = self.forward_Op_mat(self.nx, self.ny, self.views)
            b = A@x.reshape((-1,1))
            bshape = b.shape
            self.p = self.views
            self.q = int(bshape[0]/self.views)
            bimage = b.reshape((self.p, self.q))
        elif matrix == False:
            A = self.forward_Op(self.nx, self.ny, self.views)
            b = A@x.reshape((-1,1))
            bshape = b.shape
            self.p = self.views
            self.q = int(bshape[0]/self.views)
            bimage = b.reshape((self.p, self.q))
        return (A, b, self.p, self.q)

    def add_noise(self, b_true, opt, noise_level):
        if (opt == 'Gaussian'):
            # mu_obs = np.zeros((self.p*self.q,1))      # mean of noise
            noise = np.random.randn(b_true.shape[0]).reshape((-1,1))
            e = noise_level * np.linalg.norm(b_true) / np.linalg.norm(noise) * noise
            e = e.reshape((-1,1))
            b_true = b_true.reshape((-1,1))
            b = b_true + e # add noise
            b_meas = b_true + e
            b_meas_i = b_meas.reshape((self.p, self.q))
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
            b_meas_i = b_meas.reshape((self.p, self.q))
        return (b_meas_i , e)

    def plot_rec(self, img, save_imgs=True, save_path='./saveImagesTomo'):
            plt.set_cmap('inferno')
            if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
            plt.imshow(img.reshape((self.nx, self.ny), order = 'F'))
            plt.axis('off')
            if save_imgs:  plt.savefig(save_path+'/rec'+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw()
        
    def plot_data(self, img, save_imgs = False, save_path='./saveImagesData'):
        plt.set_cmap('inferno')
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        plt.imshow(img.reshape((self.p, self.q), order = 'F'))
        plt.axis('off')
        if save_imgs:  plt.savefig(save_path+'/sino'+'.png',bbox_inches='tight')
        plt.pause(.1)
        plt.draw()

class Tomography():
    def __init__(self,**kwargs):
        seed = kwargs.pop('seed',2022)
    def define_proj_id(self, sizex, sizey, views, **kwargs):
        self.dataset = kwargs['dataset'] if ('dataset' in kwargs) else False
        self.nx = sizex
        self.ny = sizey
        self.p = int(np.sqrt(2)*self.nx)    # number of detector pixels
        self.q = views           # number of projection angles
        self.views = views
        self.theta = np.linspace(0, 2*np.pi, self.q, endpoint=False)   # in rad
        self.source_origin = 3*self.nx                     # source origin distance [cm]
        self.detector_origin = self.nx                      # origin detector distance [cm]
        self.detector_pixel_size = (self.source_origin + self.detector_origin)/self.source_origin
        self.detector_length = self.detector_pixel_size*self.p   # detector length
        self.vol_geom = astra.create_vol_geom(self.nx,self.nx)
        self.proj_geom = astra.create_proj_geom('fanflat', self.detector_pixel_size, self.p, self.theta, self.source_origin, self.detector_origin)
        self.proj_id = astra.create_projector('line_fanflat', self.proj_geom, self.vol_geom)
        return self.proj_id

    def define_A(self, sizex, sizey, views): 
            proj_id = self.define_proj_id(sizex, sizey, views)  
            self.A = astra.OpTomo(self.proj_id)     
            return self.A

    def forward_Op(self, x, sizex, sizey, views):
        A = self.define_A(sizex, sizey, views)
        operatorf = lambda X: (A*X.reshape((sizex, sizey))).reshape(-1,1)
        operatorb = lambda B: A.T*B.reshape((self.p, self.q))
        OP = pylops.FunctionOperator(operatorf, operatorb, self.p*self.q, sizex*sizey)
        return OP, A

    def forward_Op_mat(self, sizex, sizey, views):
        proj_id = self.define_proj_id(sizex, sizey, views)
        self.mat_id = astra.projector.matrix(proj_id)
        self.Amat = astra.matrix.get(self.mat_id) 
        return self.Amat

    def gen_true(self, sizex, sizey, test_problem):
        if test_problem == 'grains':
            N_fine = sizex
            numGrains = int(round(4*np.sqrt(N_fine)))
            x_true = phantom.grains(N_fine, numGrains) 
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'smooth':
            N_fine = sizex
            x_true = phantom.smooth(N_fine) 
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'tectonic':
            N_fine = sizex
            x_true = phantom.tectonic(N_fine)
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1] 
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'threephases':
            N_fine = sizex
            x_true = phantom.threephases(N_fine) 
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'ppower':
            N_fine = sizex
            x_true = phantom.ppower(N_fine) 
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'CT60':
            data = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/Shepp-Logan_proj60_SNR100.mat')
            x_true = data['x_true']
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'CT90':
            data = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/Shepp-Logan_proj90_SNR100.mat')
            x_true = data['x_true']
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'head':
            dataname = 'head'
            data = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/'+ dataname +'.mat')
            x_true = data['x_true']
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1))  
        else:
            raise TypeError("You must enter a valid test problem! Options are: grains, smooth, tectonic, threephases, ppower, CT60, CT90, head.")
        return (x_truef, self.nx, self.ny)

    def gen_saved_data(self, dataset):
        if dataset == 60:
            CT = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/CT_x128_proj60_loc100.mat')
            A, phi, s = CT['A'],CT['phi'],CT['s']
            data = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/Shepp-Logan_proj60_SNR100.mat')
            x_true = data['x_true']
            b = data['b'].T#A*x_true.reshape((-1,1))
            self.q = phi.shape[1]
            self.p = s.shape[1]
        elif dataset == 90:
            CT = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/CT_x128_proj90_loc100.mat')
            A, phi, s = CT['A'],CT['phi'],CT['s']
            data = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/Shepp-Logan_proj90_SNR100.mat')
            x_true = data['x_true']
            b = data['b'].T#A*x_true.reshape((-1,1))
            self.q = phi.shape[1]
            self.p = s.shape[1]
        elif dataset == 'head':
            CT = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/CT_x512_proj200_loc512.mat')
            A,phi,s = CT['A'],CT['phi'],CT['s']
            data = spio.loadmat('/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos/data/CT/'+'head'+'.mat')
            x_true,b = data['x_true'], data['b']
            b = b.T
        return (A, x_true, b)

    def gen_data(self, x, matrix, nx, ny, views):
        self.nx = nx
        self.ny = ny
        self.views = views
        proj_id = self.define_proj_id(self.nx, self.ny, self.views)
        if matrix == True:
            A = self.forward_Op_mat(self.nx, self.ny, self.views)
            b = A@x.reshape((-1,1))
            bshape = b.shape
            self.p = self.views
            self.q = int(bshape[0]/self.views)
            bimage = b.reshape((self.p, self.q))
            AforMatrixOperation = []
        elif matrix == False:
            # A = self.forward_Op(self.nx, self.ny, self.views)
            (A, AforMatrixOperation) = self.forward_Op(x, self.nx, self.ny, self.views)
            b = A@x.reshape((-1,1))
            bshape = b.shape
            self.p = self.views
            self.q = int(bshape[0]/self.views)
            bimage = b.reshape((self.p, self.q))
        return (A, b, self.p, self.q, AforMatrixOperation)
    
    def add_noise(self, b_true, opt, noise_level):
        if (opt == 'Gaussian'):
            # mu_obs = np.zeros((self.p*self.q,1))      # mean of noise
            noise = np.random.randn(b_true.shape[0]).reshape((-1,1))
            e = noise_level * np.linalg.norm(b_true) / np.linalg.norm(noise) * noise
            e = e.reshape((-1,1))
            b_true = b_true.reshape((-1,1))
            b = b_true + e # add noise
            b_meas = b_true + e
            b_meas_i = b_meas.reshape((self.p, self.q))
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
            b_meas_i = b_meas.reshape((self.p, self.q))
        return (b_meas_i , e)

    def plot_rec(self, img, save_imgs=True, save_path='./saveImagesTomo'):
            plt.set_cmap('inferno')
            if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
            plt.imshow(img.reshape((self.nx, self.ny), order = 'F'))
            plt.axis('off')
            if save_imgs:  plt.savefig(save_path+'/rec'+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw()
        
    def plot_data(self, img, save_imgs = False, save_path='./saveImagesData'):
        plt.set_cmap('inferno')
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        plt.imshow(img.reshape((self.p, self.q), order = 'F'))
        plt.axis('off')
        if save_imgs:  plt.savefig(save_path+'/sino'+'.png',bbox_inches='tight')
        plt.pause(.1)
        plt.draw()

class Deblurring1D:
    def __init__(self,**kwargs):
        seed = kwargs.pop('seed',2022)
    def operator(self, x, projection, PSF, boundary_condition):
        if projection == 'forward':
            return self.forward_p(x, PSF, boundary_condition)
        elif projection == 'backward':
            return self.backward_p(x, PSF, boundary_condition)
    def forward_p(self, x, PSF, boundary_conditions):
        A_times_x = convolve1d(x, PSF, mode = boundary_conditions) 
        return A_times_x
    def backward_p(self, b, PSF, boundary_conditions):
        PSF = PSF[::-1]
        A_transpose_b = convolve1d(b, PSF, mode = boundary_conditions)
        return A_transpose_b
    def Gauss1D(self, grid_points, parameter):  
        self.grid_points = grid_points  
        x = np.arange(-np.fix(grid_points/2), np.ceil(grid_points/2))
        PSF = np.exp(-0.5*((x**2)/(parameter**2)))
        PSF /= PSF.sum()
        center = np.int0(np.where(PSF == PSF.max())[0][0])
        return PSF, center
    def Defocus1D(self, grid_points, parameter):  
        self.grid_points = grid_points  
        center = np.int0(np.fix(int(grid_points/2)))
        if (parameter == 0):    
            PSF = np.zeros(grid_points)
            PSF[center] = 1
            self.PSF = PSF
        else:
            PSF = np.ones(grid_points) / (np.pi * parameter**2)
            temp = np.array(((np.arange(1, grid_points+1)-center)**2 > (parameter**2)))
            PSF[temp] = 0
            self.PSF = PSF / PSF.sum()
        return PSF, center

    def P1D(self, spread, center, shape):
        image = np.zeros(shape)
        for i in range(shape):
                v = np.exp(-(((i-center)/spread)**2 )/2)
                if v < 0.0001:
                    continue
                image[i] = v
        return image

    def forward_Op_matrix_1D(self, spread, shape):
        m = shape
        n = 1
        A = np.zeros((m*n, m*n))
        count = 0
        spread = spread
        shape = shape
        for i in range(m):
            for j in range(n):
                column = (self.P1D(spread, i,  shape))
                A[:, count] = column
                count += 1
        normalize = np.sum(A[:, int(m*n/2 + n/2)])
        A = 1/normalize * A
        return A
    
    def forward_Op_1D(self, x, blur_type, parameter, boundary_condition = 'reflect'):
        self.parameter = parameter
        self.PSF, self.center = self.Gauss1D(self.grid_points, self.parameter)
        proj_forward = lambda x: self.operator(x, 'forward', self.PSF, boundary_condition)
        proj_backward = lambda x: self.operator(x, 'backward', self.PSF, boundary_condition)
        blur = pylops.FunctionOperator(proj_forward, proj_backward, self.grid_points)
        return blur
    
    def gen_data(self, x, blur_type, parameter, boundary_condition = 'reflect'):
        self.parameter = parameter
        self.PSF, self.center = self.Gauss1D(self.grid_points, self.parameter)
        proj_forward = lambda x: self.operator(x, 'forward', self.PSF, boundary_condition)
        proj_backward = lambda x: self.operator(x, 'backward', self.PSF, boundary_condition)
        blur = pylops.FunctionOperator(proj_forward, proj_backward, self.grid_points)
        b = self.operator(x, 'forward', self.PSF, boundary_condition)
        return b
    
    def gen_xtrue(self, N, test):
        self.grid_points = N
        if test == 'sigma':
            x = np.linspace(-2.5, 2.5, N)
            x_true = np.piecewise(x, [x < 0, x >= 0], [-1, 1]) 
        if test == 'piecewise':
            x_min = 0
            x_max = 1
            values = np.array([0, 1, 0, 0, 0, 0, 0, 0.25, 0, 1, 0])
            conditions = lambda x: [(x_min <= x) & (x < 0.10), (0.10 <= x) & (x < 0.15), (0.15 <= x) & (x < 0.20),  \
                   (0.20  <= x) & (x < 0.25), (0.25 <= x) & (x < 0.35), (0.35 <= x) & (x < 0.38),\
                   (0.38  <= x) & (x < 0.45), (0.45 <= x) & (x < 0.55), \
                   (0.55  <= x) & (x < 0.75), (0.75 <= x) & (x < 0.8), (0.8 <= x) & (x <= x_max)]
            f = lambda x: np.piecewise(x, conditions(x), values)
            xx  = np.linspace(x_min, x_max, N)
            x_true = f(xx)
        if test == 'shaw':
            h = np.pi/N
            a1 = 2; c1 = 6; t1 =  .8
            a2 = 1; c2 = 2; t2 = -.5
            x1 = a1*np.exp(-c1*(-np.pi/2 + np.arange(0.5,N,1)*h - t1)**2)
            x2 = a2*np.exp(-c2*(-np.pi/2 + np.arange(0.5,N,1)*h - t2)**2)
            x_true = x1 + x2
        elif test == 'curve1':
            x_true = np.zeros((N,1))
            h = 1/N; sqh = np.sqrt(h)
            h32 = h*sqh
            h2 = h**2 
            sqhi = 1/sqh
            t = 2/3; 
            for i in range(0, N):
                x_true[i] = h32*(i+1-0.5)
        elif test == 'curve2':
            x_true = np.zeros((N,1))
            h = 1/N; sqh = np.sqrt(h)
            h32 = h*sqh
            h2 = h**2 
            sqhi = 1/sqh
            t = 2/3; 
            for i in range(0, N):
                x_true[i] = sqhi*(np.exp((i+1)*h) - np.exp(i*h))
        elif test == 'curve3':
            x_true = np.zeros((N,1))
            h = 1/N; sqh = np.sqrt(h)
            h32 = h*sqh
            h2 = h**2 
            sqhi = 1/sqh
            t = 2/3; 
            for i in range(0, np.int0(N/2+1)):
                x_true[i] = sqhi*(((i+1)*h)**2 - ((i)*h)**2)/2
            for i in range(np.int0(N/2+1), N):
                x_true[i] = sqhi*(h - (((i+1)*h)**2 - ((i)*h)**2)/2); 
        return x_true 
    
    def add_noise(self, b_true, opt, noise_level):
        if (opt == 'Gaussian'):
            mu_obs = np.zeros(self.grid_points)      # mean of noise
            e = np.random.randn(self.grid_points)
            delta = noise_level * np.linalg.norm(b_true) #np.linalg.norm(e)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
        if (opt == 'Poisson'):
            gamma = 1 # background counts assumed known
            b_meas = np.random.poisson(lam=b_true+gamma) 
            e = 0
            delta = np.linalg.norm(e)
        if (opt == 'Laplace'):
            mu_obs = np.zeros(self.grid_points)      # mean of noise
            e = np.random.laplace(self.grid_points)
            delta = np.linalg.norm(e)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
        return (b_meas, delta)
    
    def plot_rec(self, img, save_imgs = False, save_path='./saveImagesDeblurring1DReconstructions'):
        plt.set_cmap('inferno')
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        plt.plot(img)
        plt.axis('off')
        if save_imgs:  plt.savefig(save_path+'/rec'+'.png',bbox_inches='tight')
        plt.pause(.1)
        plt.draw()
    def plot_data(self, img, save_imgs = False, save_path='./saveImagesDeblurring1DData'):
        plt.set_cmap('inferno')
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        # plt.imshow(img.reshape((self.nx, self.ny), order = 'F'))
        plt.plot(img)
        plt.axis('off')
        if save_imgs:  plt.savefig(save_path+'/rec'+'.png',bbox_inches='tight')
        plt.pause(.1)
        plt.draw()    


if __name__ == '__main__':
    # Test Deblurring class
    from solvers.gks_all import *
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
    # xhat = GKS(A, b_vec, L, 3, 4)
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
    

   
