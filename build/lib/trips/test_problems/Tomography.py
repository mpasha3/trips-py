#!/usr/bin/env python
""" 
Builds a Tomography class
--------------------------------------------------------------------------
Created in January 2024 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola"
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
    
class Tomography():
    def __init__(self,**kwargs):
        seed = kwargs.pop('seed',2022)
        self.nx = None
        self.ny = None
        self.CommitCrime = kwargs['CommitCrime'] if ('CommitCrime' in kwargs) else False
        print('settings')
    
    def define_proj_id(self, nx, ny, views, **kwargs):
        self.dataset = kwargs['dataset'] if ('dataset' in kwargs) else False
        self.nx = nx
        self.ny = ny
        self.p = int(np.sqrt(2)*self.nx)    # number of detector pixels
        self.q = views           # number of projection angles
        self.views = views
        self.theta = np.linspace(0, np.pi, self.q, endpoint=False)   # in rad
        self.source_origin = 3*self.nx                     # source origin distance [cm]
        self.detector_origin = self.nx                      # origin detector distance [cm]
        self.detector_pixel_size = (self.source_origin + self.detector_origin)/self.source_origin
        self.detector_length = self.detector_pixel_size*self.p   # detector length
        self.vol_geom = astra.create_vol_geom(self.nx,self.nx)
        if self. CommitCrime == False:
            self.theta_mis = self.theta + 1e-8
            self.proj_geom_mis = astra.create_proj_geom('fanflat', self.detector_pixel_size, self.p, self.theta_mis, self.source_origin, self.detector_origin)
            self.proj_id_mis = astra.create_projector('line_fanflat', self.proj_geom_mis, self.vol_geom)
        self.proj_geom = astra.create_proj_geom('fanflat', self.detector_pixel_size, self.p, self.theta, self.source_origin, self.detector_origin)
        self.proj_id = astra.create_projector('line_fanflat', self.proj_geom, self.vol_geom)
        # return self.proj_id

    def define_A(self, nx, ny, views): 
            # proj_id = self.define_proj_id(nx, ny, views)
            self.define_proj_id(nx, ny, views)  
            self.A = astra.OpTomo(self.proj_id)
            if self. CommitCrime == False:
                self.A_mis = astra.OpTomo(self.proj_id_mis)
            # return self.A

    def forward_Op(self, nx, ny, views):
        A = self.define_A(nx, ny, views)
        self.define_A(nx, ny, views)
        operatorf = lambda X: (self.A*X.reshape((nx, ny))).reshape(-1,1)
        operatorb = lambda B: self.A.T*B.reshape((self.p, self.q))
        OP = pylops.FunctionOperator(operatorf, operatorb, self.p*self.q, nx*ny)
        if self. CommitCrime == False:
            A_mis = self.A_mis
            return OP, A, A_mis
        else:
            return OP, A

    def gen_true(self, test_problem, **kwargs):
        if (self.nx is None or self.ny is None):
            if (('nx' in kwargs) and ('ny' in kwargs)):
                self.nx = kwargs['nx'] 
                self.ny = kwargs['ny'] 
            else:
                raise TypeError("The dimension of the image is not specified. You can input nx and ny as (x_true, nx, ny) = Tomo.gen_true(testproblem, nx = nx, ny = ny) or first define the forward operator")
        
        if test_problem in ['SL60', 'SL90', 'head']:
            image = self.im_image_dat(test_problem)
            newimage = image
            current_shape = get_input_image_size(image)
            if ((current_shape[0] is not self.nx) and (current_shape[1] is not self.ny)):
                newimage = image_to_new_size(image, (self.nx, self.ny))
                newimage[np.isnan(newimage)] = 0
                x_truef = newimage
                self.nx = newimage.shape[0]
                self.ny = newimage.shape[1]
            else:
                self.nx = current_shape[0]
                self.ny = current_shape[1]
        elif test_problem in ['grains', 'smooth', 'tectonic', 'threephases', 'ppower']:
            if test_problem == 'grains':
                N_fine = self.nx
                numGrains = int(round(4*np.sqrt(N_fine)))
                x_true = phantom.grains(N_fine, numGrains) 
                tmp_shape = x_true.shape
                self.nx = tmp_shape[0]
                self.ny = tmp_shape[1]
                x_truef = x_true.reshape((-1,1)) 
            elif test_problem == 'smooth':
                N_fine = self.nx
                x_true = phantom.smooth(N_fine) 
                tmp_shape = x_true.shape
                self.nx = tmp_shape[0]
                self.ny = tmp_shape[1]
                x_truef = x_true.reshape((-1,1)) 
            elif test_problem == 'tectonic':
                N_fine = self.nx
                x_true = phantom.tectonic(N_fine)
                tmp_shape = x_true.shape
                self.nx = tmp_shape[0]
                self.ny = tmp_shape[1] 
                x_truef = x_true.reshape((-1,1)) 
            elif test_problem == 'threephases':
                N_fine = self.nx
                x_true = phantom.threephases(N_fine) 
                tmp_shape = x_true.shape
                self.nx = tmp_shape[0]
                self.ny = tmp_shape[1]
                x_truef = x_true.reshape((-1,1)) 
            elif test_problem == 'ppower':
                N_fine = self.nx
                x_true = phantom.ppower(N_fine) 
                tmp_shape = x_true.shape
                self.nx = tmp_shape[0]
                self.ny = tmp_shape[1]
                x_truef = x_true.reshape((-1,1)) 
        else:
            raise TypeError("You must enter a valid test problem! Options are: grains, smooth, tectonic, threephases, ppower, CT60, CT90, head.")
        
        return (x_truef, self.nx, self.ny)
    
    def gen_data(self, x, nx, ny, views):
        self.nx = nx
        self.ny = ny
        self.views = views
        if self. CommitCrime == False:
            (A, AforMatrixOperation, A_mis) = self.forward_Op(self.nx, self.ny, self.views)
            b = (A_mis*x.reshape((nx,ny))).reshape((-1,1))
            print('no crime')
        else:
            (A, AforMatrixOperation) = self.forward_Op(self.nx, self.ny, self.views)
            b = A@x.reshape((-1,1))
            print('crime')
        bshape = b.shape
        self.p = self.views
        self.q = int(bshape[0]/self.views)
        return A, b, self.p, self.q, AforMatrixOperation
    
    def gen_saved_data(self, dataset):
        if dataset == 60:
           test_problem = 'SL60'
           otherdata = 'CT60'
           data = self.im_other_dat(test_problem)
           CT = self.im_other_dat(otherdata)
           A, phi, s = CT['A'],CT['phi'],CT['s']
           x_true = data['x_true']
           b = data['b'].T#A*x_true.reshape((-1,1))
           self.q = phi.shape[1]
           self.p = s.shape[1]
        elif dataset == 90:
           test_problem = 'SL90'
           otherdata = 'CT90'
           data = self.im_other_dat(test_problem)
           CT = self.im_other_dat(otherdata)
           A, phi, s = CT['A'],CT['phi'],CT['s']
           x_true = data['x_true']
           b = data['x_true']['b'].T#A*x_true.reshape((-1,1))
           b = b[0][0]
           self.q = phi.shape[1]
           self.p = s.shape[1]
        elif dataset == 'head':
            test_problem = 'head'
            otherdata = 'CT200'
            data = self.im_other_dat(test_problem)
            CT = self.im_other_dat(otherdata)
            A,phi,s = CT['A'],CT['phi'],CT['s']
            x_true, b = data['x_true'], data['b']
            b = b.T
            b = b
        return (A, x_true, b)
    
    def add_noise(self, b_true, opt, noise_level):
        if (opt == 'Gaussian'):
            # mu_obs = np.zeros((self.p*self.q,1))      # mean of noise
            noise = np.random.randn(b_true.shape[0]).reshape((-1,1))
            e = noise_level * np.linalg.norm(b_true) / np.linalg.norm(noise) * noise
            e = e.reshape((-1,1))
            b_true = b_true.reshape((-1,1))
            delta = la.norm(e)
            b = b_true + e # add noise
            b_meas = b_true + e
            b_meas_i = b_meas.reshape((self.p, self.q))
        elif (opt == 'Poisson'):
            # Add Poisson Noise 
            gamma = 1 # background counts assumed known
            b_meas = np.random.poisson(lam=b_true+gamma) 
            b_meas_i = b_meas.reshape((self.p, self.q))
            delta = 0
        else:
            mu_obs = np.zeros(self.p*self.q)      # mean of noise
            e = np.random.laplace(self.p*self.q)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
            delta = la.norm(sig_obs*e)
            b_meas_i = b_meas.reshape((self.p, self.q))
        return (b_meas_i , delta)

    def plot_rec(self, img, save_imgs=True, save_path='./saveImagesTomo'):
            plt.set_cmap('inferno')
            if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
            plt.imshow(img.reshape((self.nx, self.ny)))
            plt.axis('off')
            if save_imgs:  plt.savefig(save_path+'/rec'+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw()

    def im_other_dat(self, im):
        if exists(f'./data/image_data/{im}.mat'):
            print('data already in the path.')
        else:
            print("Please make sure your data are on the data folder!")
        f = spio.loadmat(f'./data/image_data/{im}.mat')
        return f
    
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
    
    def plot_data(self, img, save_imgs = False, save_path='./saveImagesData'):
        plt.set_cmap('inferno')
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        plt.imshow(img.reshape((self.p, self.q)))
        plt.axis('off')
        if save_imgs:  plt.savefig(save_path+'/sino'+'.png',bbox_inches='tight')
        plt.pause(.1)
        plt.draw()


   
