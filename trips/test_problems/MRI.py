#!/usr/bin/env python
""" 
Builds an MRI class
--------------------------------------------------------------------------
Created in August 2025 for TRIPs-Py library
Code on this file is adapted by the repositori developed by Tao Hong: https://github.com/hongtao-argmin/CQNPCS_MRIReco
Before running, get data from https://github.com/hongtao-argmin/CQNPCS_MRIReco/tree/main/data and place them in the folder MRI under data
"""
__authors__ = "Mirjeta Pasha and Tao Hong; Code on this file is adapted for TRIPs-Py from the repositori developed by Tao Hong: https://github.com/hongtao-argmin/CQNPCS_MRIReco"
__affiliations__ = 'Virginia Tech and University of Texas at Austin'
__copyright__ = "Copyright 2025, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@vt.edu; and tao.hong@austin.utexas.edu;"

# demo for the spiral and radial trj. Reco.
import scipy.io
import numpy as np
import sigpy as sp
# import optalgTao as opt #import optimization algorithms
import matplotlib.pyplot as plt


class MRIClass():
    def __init__(self,**kwargs):
        seed = kwargs.pop('seed',2022)
        self.nx = None
        self.ny = None
        self.im = None
        print('settings')
    
    def forward_Op(self, waveletChoice = 'NW', trajectoryType = 'Spiral'): 
            
            if trajectoryType in ['Spiral', 'Radial']:
                if trajectoryType == 'Spiral':
                   self.trajectoryType = 'Spiral'
                   trj_file = "data/MRI/spiral/trj.npy" # trajectory
                   mps_file = "data/MRI/spiral/mps.npy" # sensitivity map
                elif trajectoryType == 'Radial':
                   self.trajectoryType = 'Radial'
                   trj_file = 'data/MRI/radial/trj.npy'
                   mps_file = 'data/MRI/radial/mps.npy'
            else:
                raise TypeError("You must enter a valid trajectory type! Options are: Spiral, Radial.")
            
            trj = np.load(trj_file)
            mps = np.load(mps_file)
            (nc, sy, sx) = mps.shape
            S = sp.linop.Multiply(mps.shape[1:], mps)
            print('Here')
            print(S)
            F = sp.linop.NUFFT(mps.shape, coord=trj)#, toeplitz=True

            if waveletChoice == 'W':
                self.waveletChoice = 'W'
                W = sp.linop.Wavelet(S.ishape)
                A_w = F * S * W.H #  includes wavelet transform for only wavelet regularization
                LL_w = sp.app.MaxEig(A_w.H*A_w,dtype=np.complex128).run()
                # normalize A and A_w
                A = np.sqrt(1/LL_w)*A_w
            else:
                self.waveletChoice = 'NW'
                A = F * S         # exclude the wavelet transform
                LL = sp.app.MaxEig(A.H*A,dtype=np.complex128).run()# * 1.01#A.N
                A = np.sqrt(1/LL)*A
                W = 0
            return A, F, W, S

    def gen_true(self, test_problem, **kwargs):
        if (self.nx is None or self.ny is None):
            if (('nx' in kwargs) and ('ny' in kwargs)):
                self.nx = kwargs['nx'] 
                self.ny = kwargs['ny'] 
            else:
                raise TypeError("The dimension of the image is not specified.")
        
        if test_problem in ['brain', 'knee']:
            if test_problem == 'knee':
                data = scipy.io.loadmat('data/MRI/knee_GT.mat')
            elif test_problem == 'brain':
                data = scipy.io.loadmat('data/MRI/brain_GT.mat')
        else:
            raise TypeError("You must enter a valid test problem! Options are: brain, knee.")
        im_real = data['im_real']
        im_imag = data['im_imag']
        im = im_real+1j*im_imag
        im = im/np.max(np.abs(im))
        self.im = im
        self.nx = im.shape[0]
        self.ny = im.shape[1]
        # np.save("%s/GTReal.npy" % folderName, np.real(im_original))
        # np.save("%s/GTImag.npy" % folderName, np.imag(im_original))
        return im_imag, self.nx, self.ny
    
    def gen_data(self):
        # self.nx = nx
        # self.ny = ny
        im = self.im
        # forward_Op(self, waveletChoice, trajectoryType)
        A, F, W, S  = self.forward_Op(self.waveletChoice, self.trajectoryType)
        b = A*im
        return A, b
    
    def add_noise(self, b, noise_level):
        b_m,b_n,b_k = b.shape
        np.random.seed(2)
        noise_real = np.random.randn(b_m,b_n,b_k)
        np.random.seed(5)
        noise_imag = np.random.randn(b_m,b_n,b_k)
        b_noise  = b+noise_level*(noise_real+1j*noise_imag)
  
        return b_noise

    def plot_rec(self, img, save_imgs=True, save_path='./saveImagesTomo'):
            plt.set_cmap('inferno')
            if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
            plt.imshow(img.reshape((self.nx, self.ny)))
            plt.axis('off')
            if save_imgs:  plt.savefig(save_path+'/rec'+'.png',bbox_inches='tight')
            plt.pause(.1)
            plt.draw()