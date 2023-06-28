"""
Functions for downloading or generating data and measurement operators.
"""

import requests
from scipy import sparse
import scipy.io as spio
import numpy as np
import h5py
import astra
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

    # delta = np.linalg.norm() # no added noise for this dataset

    for ii in range(T):

        AA[ii] = A_small[ 2170*(ii):2170*(ii+1), 16384*ii:16384*(ii+1) ]
        B[ii] = b[ 2170*(ii) : 2170*(ii+1) ]

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
    delta = 0 # no added noise for this dataset
    for ii in range(T):
        AA[ii] = A_small[ 700*(ii):700*(ii+1), 16384*ii:16384*(ii+1) ] # 217 projections of size 128x128 at each of 10 selected angles
        B[ii] = b[ 700*(ii) : 700*(ii+1) ]
    return A_small, b, AA, B, nx, ny, nt, delta


def get_stempo_data(data_set = 'real', data_thinning = '2'):
        """
        Generate stempo observations
        """
        data_file = {'simulation':'stempo_ground_truth_2d_b4','real':'stempo_seq8x45_2d_b'+str(data_thinning)}[data_set]+'.mat'
        if not os.path.exists('./data/stempo_data'): os.makedirs('./data/stempo_data')
        if not os.path.exists('./data/stempo_data/'+data_file):
            import requests
            print("downloading...")
            r = requests.get('https://zenodo.org/record/7147139/files/'+data_file)
            with open('./data/stempo_data/'+data_file, "wb") as file:
                file.write(r.content)
            print("Stempo data downloaded.")
        if  data_set=='simulation':
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
            p = int(np.sqrt(2)*N)    # number of detector pixels
            # view angles
            theta = angleVectorRad#[0]#np.linspace(0, 2*np.pi, q, endpoint=False)   # in rad
            q = theta.shape[1]          # number of projection angles
            source_origin = 3*N                     # source origin distance [cm]
            detector_origin = N                       # origin detector distance [cm]
            detector_pixel_size = (source_origin + detector_origin)/source_origin
            detector_length = detector_pixel_size*p 
            saveA = list(range(nt))
            saveb = np.zeros((p*q, nt))
            saveb_true = np.zeros((p*q, nt))
            savee = np.zeros((p*q, nt))
            savedelta = np.zeros((nt, 1))
            savex_true = np.zeros((nx*ny, nt))
            B = list(range(nt))
            count = np.int_(360/nt)
            for i in range(nt):
                proj_geom = astra.create_proj_geom('fanflat', detector_pixel_size, p, theta[i], source_origin, detector_origin)
                vol_geom = astra.create_vol_geom(N, N)
                proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
                mat_id = astra.projector.matrix(proj_id)
                A_n = astra.matrix.get(mat_id)
                x_true = image[:, :, count*i]
                x_truef_sino = x_true.flatten(order='F') 
                savex_true[:, i] = x_truef_sino
                sn = A_n*x_truef_sino
                b_i = sn.flatten(order='F') 
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
                astra.projector.delete(proj_id)
                astra.matrix.delete(mat_id)
            A = sps.block_diag((saveA))    
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
            # Load measurement data as sinogram
            # data = spio.loadmat('./data/'+data_file) # scipy.io does not support Matlab struct
            data = h5py.File('./data/stempo_data/'+data_file,'r')
            CtData = data["CtData"]
            m = np.array(CtData["sinogram"]).T # strange: why is it transposed?
            # Load parameters
            param = CtData["parameters"]
            f = h5py.File('A_seqData.mat') # it is created by `create_ct_matrix_2d_fan_astra.m` separately
            fA = f["A"]
            # Extract information
            Adata = np.array(fA["data"])
            Arowind = np.array(fA["ir"])
            Acolind = np.array(fA["jc"])
            # Need to know matrix size (shape) somehow
            n_rows = N_det*N_theta # 6300
            n_cols = N*N
            Aloaded = sps.csc_matrix((Adata, Arowind, Acolind), shape=(n_rows, n_cols))
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
            A = sps.block_diag((saveA))    
            b = saveb.flatten(order ='F')
            truth = None
        return A, b, saveA, saveb, nx, ny, nt, savedelta, truth


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