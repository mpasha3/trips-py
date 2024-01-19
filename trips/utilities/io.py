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
import pylops
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
    # b = b.reshape((-1,1))
    b = b.reshape(-1, 1, order='F').squeeze()
    AA = list(range(T))
    B = list(range(T))
    e = np.random.randn(b.shape[0],)
    sig_obs = noise_level * np.linalg.norm(b)/np.linalg.norm(e)
    b = b + sig_obs*e
    delta = np.linalg.norm(sig_obs*e)
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
    # b = b.reshape((-1,1))
    b = b.reshape(-1, 1, order='F').squeeze()
    AA = list(range(T))
    B = list(range(T))
    # e = np.random.randn(b.shape[0], 1)
    e = np.random.randn(b.shape[0],)
    sig_obs = noise_level * np.linalg.norm(b)/np.linalg.norm(e)
    b = b + sig_obs*e
    delta = np.linalg.norm(sig_obs*e)
    for ii in range(T):
        AA[ii] = A_small[ 700*(ii):700*(ii+1), 16384*ii:16384*(ii+1) ] # 217 projections of size 128x128 at each of 10 selected angles
        B[ii] = b[700*(ii) : 700*(ii+1) ]
    return A_small, b, AA, B, nx, ny, nt, delta


def get_stempo_data(data_set = 'real', data_thinning = 2):
        """
        Generate stempo observations
        """
        data_file = {'simulation':'stempo_ground_truth_2d_b4','real':'stempo_seq8x45_2d_b'+str(data_thinning)}[data_set]+'.mat'
        if not os.path.exists('./data/stempo_data'): os.makedirs('./data/stempo_data/')
        if not os.path.exists('./data/stempo_data'+data_file):
            import requests
            print("downloading...")
            r = requests.get('https://zenodo.org/record/7147139/files/'+data_file)
            with open('./data/stempo_data/'+data_file, "wb") as file:
                file.write(r.content)
            print("Stempo data downloaded.")
        if data_set=='simulation':
            truth = spio.loadmat('./data/stempo_data/'+data_file)
            image = truth['obj']
            nx, ny, nt = 560, 560, 20
            anglecount = 10
            rowshift = 5
            columnsshift = 14
            nt = 20
            angleVector = list(range(nt))
            for t in range(nt):
                angleVector[t] = np.linspace(rowshift*t, 14*anglecount+ rowshift*t, num = anglecount+1)
            angleVectorRad = np.deg2rad(angleVector)
                    # Generate matrix versions of the operators and a large bidiagonal sparse matrix
            N = nx         # object size N-by-N pixels
            # view angles
            theta = angleVectorRad#[0]#np.linspace(0, 2*np.pi, q, endpoint=False)   # in rad
            views = theta.shape[1]          # number of projection angles
            saveA = list(range(nt))
            saveb = np.zeros((views*N, nt))
            saveb_true = np.zeros((views*N, nt))
            savee = np.zeros((views*N, nt))
            savedelta = np.zeros((nt, 1))
            savex_true = np.zeros((nx*ny, nt))
            B = list(range(nt))
            count = int(360/nt)

            for i in range(nt):
                slice_geom = astra.create_vol_geom(N, N)
                sino_geom = astra.create_proj_geom('parallel', 1, N, theta[i])
                proj_id = astra.creators.create_projector('linear', sino_geom, slice_geom)
                A = astra.OpTomo(proj_id)
                operatorf = lambda X: (A*(X.reshape(N, N))).reshape(-1, 1) / N
                operatorb = lambda B: (A.T*(B.reshape(views, N))).reshape(-1, 1) / N
                A_n = pylops.FunctionOperator(operatorf, operatorb, views*N, N*N)
                x_true = image[:, :, count*i]
                x_truef_sino = x_true.flatten(order='F') 
                savex_true[:, i] = x_truef_sino
                sn = A_n@x_truef_sino
                b_i = sn.flatten(order='F') 
                # print('MP here')
                # tmp = A_n.T*b_i
                # print(tmp.shape)
                sigma = 0.01 # noise level
                e = np.random.normal(0, 1, b_i.shape[0])
                e = e/np.linalg.norm(e)*np.linalg.norm(b_i)*sigma
                delta = np.linalg.norm(e)
                b_m = b_i + e
                saveA[i] = A_n
                B[i] = b_m
                saveb_true[:, i] = sn
                saveb[:, i] = b_m
                savee[:, i] = e
                savedelta[i] = delta
            Afull = pylops.BlockDiag(saveA)
            # A = sps.block_diag((saveA))    
            b = saveb.flatten(order ='F') 
            # xf = savex_true.flatten(order = 'F')
            truth = savex_true.reshape((nx, ny, nt), order='F').transpose((2,0,1))
        elif data_set=='real':
            import h5py
            N = int(2240/data_thinning) # 140
            nx, ny, nt =  N, N, 8
            N_det = N
            N_theta = 45
            theta = np.linspace(0,360,N_theta,endpoint=False)
            with h5py.File(f'./data/stempo_data/'+data_file, 'r') as f:
                param = np.array(f['CtData']["parameters"])
                m = np.array(f['CtData']['sinogram']).T
            with h5py.File(f'./data/stempo_data/A_seqData.mat', 'r') as f:
                Adata = np.array(f["A"]["data"])
                print(Adata.shape)
                Arowind = np.array(f["A"]["ir"])
                print(Arowind.shape)
                Acolind = np.array(f["A"]["jc"])
                print(Acolind.shape)
            n_rows = N_det*N_theta 
            n_cols = N*N
            Aloaded = scipy.sparse.csc_matrix((Adata, Arowind, Acolind), shape=(n_rows, n_cols))
            # Aloaded = pylops.LinearOperator.tosparse((Adata, Arowind, Acolind), shape=(n_rows, n_cols))
            saveA = list(range(nt))
            saveb = np.zeros((n_rows, nt))
            savee = np.zeros((n_rows, nt))
            savedelta = np.zeros((nt, 1))
            B = list(range(nt))
            for i in range(nt):
                tmp = m[45*(i):45*(i+1), :]
                b_i = tmp.flatten()
                sigma = 0.01 # noise level
                e = np.random.normal(0, 1, b_i.shape[0])
                e = e/np.linalg.norm(e)*np.linalg.norm(b_i)*sigma
                delta = np.linalg.norm(e)
                b_m = b_i + e
                saveA[i] = Aloaded
                B[i] = b_m
                saveb[:, i] = b_m
                savee[:, i] = e
                savedelta[i] = delta  
            Afull = pylops.BlockDiag(saveA)
            b = saveb.flatten(order ='F')
            truth = None
        return Afull, b, saveA, B, nx, ny, nt, savedelta, truth


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