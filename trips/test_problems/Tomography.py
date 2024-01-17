#!/usr/bin/env python
""" 
Builds a Tomography class
--------------------------------------------------------------------------
Created in January 2024 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, Connor Sanderford, and Ugochukwu Obinna Ugwu"
__affiliations__ = 'Tufts University, University of Bath, Arizona State University, and Tufts University'
__copyright__ = "Copyright 2024, TRIPs-Py library"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu; connorsanderford@gmail.com; Ugochukwu.Ugwu@tufts.edu"

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

    def gen_data(self, x, nx, ny, views):
        self.nx = nx
        self.ny = ny
        self.views = views
        proj_id = self.define_proj_id(self.nx, self.ny, self.views)
        (A, AforMatrixOperation) = self.forward_Op(x, self.nx, self.ny, self.views)
        b = A@x.reshape((-1,1))
        bshape = b.shape
        self.p = self.views
        self.q = int(bshape[0]/self.views)
        bimage = b.reshape((self.p, self.q))
        return A, b, self.p, self.q, AforMatrixOperation
    
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
        
    def plot_data(self, img, save_imgs = False, save_path='./saveImagesData'):
        plt.set_cmap('inferno')
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        plt.imshow(img.reshape((self.p, self.q)))
        plt.axis('off')
        if save_imgs:  plt.savefig(save_path+'/sino'+'.png',bbox_inches='tight')
        plt.pause(.1)
        plt.draw()

if __name__ == '__main__':

    Tomo = Tomography()
    Amat = Tomo.forward_Op_mat()
    A = Tomo.forward_Op
    x_true = Tomo.generate_true()
    b_true = Tomo.generate_data(x_true, 1, 'simulated')
    b = Tomo.add_noise(b_true, 'Gaussian', noise_level= 0.01)
    Tomo.plot_rec(x_true)
    Tomo.plot_sino(b)
    

   
