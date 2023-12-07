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


"""
Functions for downloading or generating data and measurement operators.
"""

import requests
from scipy import sparse
import scipy.io as spio
import numpy as np
import h5py
import astra
import requests
from os import mkdir
from os.path import exists
import scipy.sparse as sps
# split into test problems
# separate IO from test problem class implementation
import os,sys
"""
H image:
"""

def build_x_true(dx=10, dy=10, up_width=10, bar_width=5, size=64):

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

    x_exact = h_im.reshape(-1,1)
    return x_exact

"""
Images for 2D Deblurring
"""
def get_image_data(dataset = 'satellite'):

    assert dataset in ['satellite', 'hubble', 'star']

    if exists(f'./data/image_data/{dataset}.mat'):
        print('data already downloaded.')

    else:
        print("downloading...")

        if not exists('./data'):
            mkdir("./data")

        if not exists('./data/image_data'):
            mkdir("./data/image_data")

        r = requests.get(f'https://drive.google.com/drive/folders/1uexAAYKRnSy0YTXMisk6IxcmwQ66H2by?usp=share_link/{dataset}.mat')

        with open(f'./data/image_data/{dataset}.mat', "wb") as file:

            file.write(r.content)

        print("downloaded.")


"""
Generate image data
"""
def generate_image_dat(im):
    get_image_data(im)
    currentpath = os.getcwd()
    f = spio.loadmat(f'./data/image_data/{im}.mat')
    X = f['x_true']
    X_true = X/X.max()
    nx, ny = X_true.shape  
    x_truef = X_true.reshape((-1,1))
    return X_true


"""
Emoji data and operator:
"""
def get_emoji_data(dataset = 30):

    assert dataset in [30,60]

    if exists(f'./data/emoji_data/DataDynamic_128x{dataset}.mat'):
        print('data already downloaded.')

    else:
        print("downloading...")

        if not exists('./data'):
            mkdir("./data")

        if not exists('./data/emoji_data'):
            mkdir("./data/emoji_data")

        r = requests.get(f'https://zenodo.org/record/1183532/files/DataDynamic_128x{dataset}.mat')

        with open(f'./data/emoji_data/DataDynamic_128x{dataset}.mat', "wb") as file:

            file.write(r.content)

        print("downloaded.")



def generate_emoji(noise_level, dataset):

    get_emoji_data(dataset)

    with h5py.File(f'./data/emoji_data/DataDynamic_128x{dataset}.mat', 'r') as f:
        A = sparse.csc_matrix((f["A"]["data"], f["A"]["ir"], f["A"]["jc"]))
        normA = np.array(f['normA'])
        sinogram = np.array(f['sinogram']).T
    T = 33
    N = np.sqrt(A.shape[1] / T)
    [mm, nn] = sinogram.shape
    ind = []
    for ii in range(int(nn /3)):
        ind.extend( np.arange(0,mm) + (3*ii)*mm )
    m2 = sinogram[:, 0::3]
    A_small = A[ind, :]
    b = m2
    nt = int(T)
    nx = int(N)
    ny = int(N)
    b = b.reshape(-1, 1, order='F').squeeze()
    AA = list(range(T))
    B = list(range(T))
    e = np.random.randn(b.shape[0], 1)
    sig_obs = noise_level * np.linalg.norm(b)/np.linalg.norm(e)
    b = b + sig_obs*e
    delta = np.linalg.norm(sig_obs*e)
    print('MP')
    print(delta)
    # delta = 0 # np.linalg.norm() # no added noise for this dataset, change to allow added noise.
    for ii in range(T):
        AA[ii] = A_small[ 2170*(ii):2170*(ii+1), 16384*ii:16384*(ii+1) ]
        B[ii] = b[2170*(ii) : 2170*(ii+1)]
    return (A_small, b, AA, B, nx, ny, nt, delta)


"""Crossphantom data and operator"""

def get_crossPhantom(dataset):

    assert dataset in [15,60]

    if not exists('./data'):
        mkdir("./data")

    if not exists('./data/crossphantom_data'):
        mkdir("./data/crossphantom_data")

    if exists(f"./data/crossphantom_data/DataDynamic_128x{dataset}.mat"):
        print('Data already downloaded.')
    else:
        r = requests.get(f'https://zenodo.org/record/1341457/files/DataDynamic_128x{dataset}.mat')
        with open(f'./data/crossphantom_data/DataDynamic_128x{dataset}.mat', "wb") as file:
            file.write(r.content)
        print("CrossPhantom data downloaded.")
 

def generate_crossPhantom(noise_level, dataset): # use noise_level

    assert dataset in [15,60]

    get_crossPhantom(dataset)

    f = spio.loadmat(f'./data/crossphantom_data/DataDynamic_128x{dataset}.mat')

    A = f['A']
    sinogram = f['sinogram']

    T = 16
    N = np.sqrt(A.shape[1] / T)
    [mm, nn] = sinogram.shape
    ind = []
    for ii in range(int(nn /3)): # every 3 of 30 angles for 33 seconds
        ind.extend( np.arange(0,mm) + (3*ii)*mm )
    m2 = sinogram[:, ::3]
    A_small = A[ind, :]
    b = m2
    nt = int(T)
    nx = int(N)
    ny = int(N)
    b = b.reshape(-1, 1, order='F').squeeze()
    AA = list(range(T))
    B = list(range(T))
    e = np.random.randn(b.shape[0], 1)
    sig_obs = noise_level * np.linalg.norm(b)/np.linalg.norm(e)
    b = b + sig_obs*e
    delta = np.linalg.norm(sig_obs*e)
    for ii in range(T):
        AA[ii] = A_small[ 700*(ii):700*(ii+1), 16384*ii:16384*(ii+1) ] # 217 projections of size 128x128 at each of 10 selected angles
        B[ii] = b[700*(ii) : 700*(ii+1) ]
    return A_small, b, AA, B, nx, ny, nt, delta


if __name__ == "__main__":

    (A, b, AA, B, nx, ny, nt, z) = generate_emoji(noise_level = 0, dataset=60)

    print(A.shape)
    print(b.shape)
    print(nx,ny,nt)

    (A, b, AA, B, nx, ny, nt, z) = generate_emoji(noise_level = 0, dataset=30)

    print(A.shape)
    print(b.shape)
    print(nx,ny,nt)

    (A, b, AA, B, nx, ny, nt, z) = generate_crossPhantom(noise_level = 0, dataset=15)

    print(A.shape)
    print(b.shape)
    print(nx,ny,nt)

    breakpoint()