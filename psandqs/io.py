"""
Functions for downloading or generating data and measurement operators.
"""

import requests
from scipy import sparse
import numpy as np
import h5py

from os import mkdir
from os.path import exists

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

def get_emoji_data(data_size = 30):

    assert data_size in [30,60]

    if exists(f'./data/DataDynamic_128x{data_size}.mat'):
        print('data already downloaded.')

    else:
        print("downloading...")

        if not exists('./data'):
            mkdir("./data")
        r = requests.get(f'https://zenodo.org/record/1183532/files/DataDynamic_128x{data_size}.mat')

        with open(f'./data/DataDynamic_128x{data_size}.mat', "wb") as file:

            file.write(r.content)

        print("downloaded.")



def generate_emoji(noise_level, data_size):

    get_emoji_data(data_size)

    with h5py.File(f'./data/DataDynamic_128x{data_size}.mat', 'r') as f:
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

if __name__ == "__main__":

    (A, b, AA, B, nx, ny, nt, z) = generate_emoji(noise_level = 0, data_size=60)

    print(A.shape)
    print(b.shape)
    print(nx,ny,nt)

    (A, b, AA, B, nx, ny, nt, z) = generate_emoji(noise_level = 0, data_size=30)

    print(A.shape)
    print(b.shape)
    print(nx,ny,nt)

    breakpoint()