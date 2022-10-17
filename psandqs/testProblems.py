import time
import numpy as np
import scipy as sp
import scipy.stats as sps
import scipy.io as spio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import astra
import phantoms as phantom
from venv import create
import pylops
from scipy.ndimage import convolve
from scipy import sparse
from scipy.ndimage import convolve
import scipy.special as spe

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

    def generate_true(self):
        data = spio.loadmat('satellite.mat')
        X = data['x_true']
        X_true = X/X.max()
        self.nx, self.ny = X_true.shape  
        x_truef = X_true.flatten(order = 'F')
        return x_truef

    def generate_data(self, x):
        A = self.forward_Op(self.dim, self.spread, self.nx, self.ny)
        b = A*x
        return b
        
    def add_noise(self, b_true, opt, noise_level):
        if (opt == 'Gaussian'):
            mu_obs = np.zeros(self.nx*self.ny)      # mean of noise
            e = np.random.randn(self.nx*self.ny)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
            b_meas_i = b_meas.reshape((self.nx, self.ny), order='F')
        if (opt == 'Poisson'):
            gamma = 1 # background counts assumed known
            b_meas = np.random.poisson(lam=b_true+gamma) 
            b_meas_i = b_meas.reshape((self.nx, self.ny), order='F')
           
        if (opt == 'Laplace'):
            mu_obs = np.zeros(self.nx*self.ny)      # mean of noise
            e = np.random.laplace(self.nx*self.ny)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
            b_meas_i = b_meas.reshape((self.nx, self.ny), order='F')
        return b_meas_i
          
    def plot_rec(self, img, save_imgs = False, save_path='./saveImagesDeblurring'):
        import matplotlib.pyplot as plt
        import os, sys
        plt.set_cmap('inferno')
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        plt.imshow(img.reshape((self.nx, self.ny), order = 'F'))
        if save_imgs:  plt.savefig(save_path+'/rec'+'.png',bbox_inches='tight')
        plt.pause(.1)
        plt.draw()

class Tomography:
    def __init__(self,**kwargs):
        seed = kwargs.pop('seed',2022)
        # self.setup(seed,**kwargs)
    def set_up(self):
        self.nx = 256          # object size nx-by-nx pixels
        self.ny = 256
        self.p = int(np.sqrt(2)*self.nx)    # number of detector pixels
        self.q = 90            # number of projection angles
        self.theta = np.linspace(0, 2*np.pi, self.q, endpoint=False)   # in rad
        self.source_origin = 3*self.nx                     # source origin distance [cm]
        self.detector_origin = self.nx                      # origin detector distance [cm]
        self.detector_pixel_size = (self.source_origin + self.detector_origin)/self.source_origin
        self.detector_length = self.detector_pixel_size*self.p   # detector length
        self.vol_geom = astra.create_vol_geom(self.nx,self.nx)
        self.proj_geom = astra.create_proj_geom('fanflat', self.detector_pixel_size, self.p, self.theta, self.source_origin, self.detector_origin)
        self.proj_id = astra.create_projector('line_fanflat', self.proj_geom, self.vol_geom)
        return self.proj_geom, self.proj_id

    def forward_Op_mat(self):
        proj_geom, proj_id = self.set_up()
        self.mat_id = astra.projector.matrix(proj_id)
        return astra.matrix.get(self.mat_id) 

    def forward_Op(self):        
        return astra.OpTomo(self.proj_id)

    def generate_true(self):
        N_fine = self.nx
        numGrains = int(round(4*np.sqrt(N_fine)))
        x_true = phantom.grains(N_fine, numGrains) 
        idd = np.where(x_true<0)
        x_true[idd] = 0
        x_truef = x_true.flatten(order='F') 
        xx_norm = np.linalg.norm(x_truef)
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
            e = np.random.randn(self.p*self.q)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            # sig_obs = err_lev * np.linalg.norm(b_true)/np.sqrt(m)             # std of noise
            # e_true = np.random.normal(loc=mu_obs, scale=sig_obs*np.ones(m))   # noise
            # b_meas = b_true + e_true                                          # 'measured' data
            b_meas = b_true + sig_obs*e
            b_meas_i = b_meas.reshape((self.p, self.q), order='F')
        if (opt == 'Poisson'):
            # Add Poisson Noise 
            gamma = 1 # background counts assumed known
            b_meas = np.random.poisson(lam=b_true+gamma) 
            b_meas_i = b_meas.reshape((self.p, self.q), order='F')
        if (opt == 'Laplace'):
            mu_obs = np.zeros(self.p*self.q)      # mean of noise
            e = np.random.laplace(self.p*self.q)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
            b_meas_i = b_meas.reshape((self.p, self.q), order='F')
        return b_meas_i  

    def plot_rec(self, img, save_imgs=True, save_path='./saveImagesTomo'):
            import matplotlib.pyplot as plt
            import os, sys
            plt.set_cmap('inferno')
            if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
            plt.imshow(img.reshape((self.nx, self.ny), order = 'F'))
            if save_imgs:  plt.savefig(save_path+'/rec'+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw()
        
    def plot_sino(self, img, save_imgs = True, save_path='./saveImagesTomo'):
        import matplotlib.pyplot as plt
        import os, sys
        plt.set_cmap('inferno')
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        plt.imshow(img.reshape((self.p, self.q), order = 'F'))
        if save_imgs:  plt.savefig(save_path+'/sino'+'.png',bbox_inches='tight')
        plt.pause(.1)
        plt.draw()

if __name__ == '__main__':
    # Test Deblurring class
    Deblur = Deblurring()
    A = Deblur.forward_Op((21,21), 1.5, 128, 128)
    x_true = Deblur.generate_true()
    b_true = Deblur.generate_data(x_true)
    b = Deblur.add_noise(b_true, 'Gaussian', noise_level = 0.01)
    Deblur.plot_rec(x_true, save_imgs = True, save_path='./saveImagesDeblurring')

    # Test Tomography class
    Tomo = Tomography()
    Amat = Tomo.forward_Op_mat()
    A = Tomo.forward_Op
    x_true = Tomo.generate_true()
    b_true = Tomo.generate_data(x_true, 1, 'simulated')
    b = Tomo.add_noise(b_true, 'Gaussian', noise_level= 0.01)
    Tomo.plot_rec(x_true)
    Tomo.plot_sino(b)

   
