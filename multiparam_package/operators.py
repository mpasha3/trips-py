import numpy as np

import pylops

def first_derivative_operator(n):

    L = pylops.FirstDerivative(n, dtype="float32")

    return L



def first_derivative_operator_2d(nx, ny):

    D_x = first_derivative_operator(nx)
    D_y = first_derivative_operator(ny)

    IDx = pylops.Kronecker( pylops.Identity(nx, dtype='float32'), D_x )
    DyI = pylops.Kronecker( D_y, pylops.Identity(ny, dtype='float32') )

    D_spatial = pylops.VStack((IDx, DyI))

    return D_spatial



def spatial_derivative_operator(nx, ny, nt):

    D_spatial = first_derivative_operator_2d(nx,ny)

    ID_spatial = pylops.Kronecker( pylops.Identity(nt, dtype='float32'), D_spatial)

    return ID_spatial

def time_derivative_operator(nx, ny, nt):
    
    D_time = first_derivative_operator(nt)

    D_timeI = pylops.Kronecker( D_time, pylops.Identity(nx**2, dtype='float32') )

    return D_timeI


if __name__ == "__main__":
    print(first_derivative_operator_2d(10,10)[:-1, :])

    D_spatial = first_derivative_operator_2d(nx,ny)

    ID_spatial = sparse.kron( sparse.identity(nt), D_spatial)