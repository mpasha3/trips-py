"""
Functions for downloading or generating data and measurement operators.
"""

import requests
from scipy import sparse
import numpy as np
import h5py

from os.path import exists

"""
H image:
"""

def build_x_true():
    dx = 10
    dy = 10
    up_width = 10
    bar_width= 5
    size = 64

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

def get_emoji_data():

    if exists('./data/DataDynamic_128x30.mat'):
        print('data already downloaded.')

    else:
        print("downloading...")
        r = requests.get('https://zenodo.org/record/1183532/files/DataDynamic_128x30.mat')

        with open('./data/DataDynamic_128x30.mat', "wb") as file:

            file.write(r.content)

        print("downloaded.")



def generate_emoji(noise_level):

    get_emoji_data()

    with h5py.File('./data/DataDynamic_128x30.mat', 'r') as f:
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

    delta = 0 # no added noise for this dataset

    for ii in range(T):

        AA[ii] = A_small[ 2170*(ii):2170*(ii+1), 16384*ii:16384*(ii+1) ]
        B[ii] = b[ 2170*(ii) : 2170*(ii+1) ]

    return (A_small, b, AA, B, nx, ny, nt, 0)