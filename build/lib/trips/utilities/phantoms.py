# ========================================================================
# Created by:
# Felipe Uribe @ DTU compute
# ========================================================================
# Adapted from 'phantomgallery' (AIRToolsII)
# https://github.com/jakobsj/AIRToolsII
# ========================================================================
# Version 2020-03
# ========================================================================
import numpy as np
# from scipy import sparse #, fftpack
# import itertools
# from numba import njit
# @njit
#=========================================================================
#=========================================================================
#=========================================================================
def shepp_logan(N):
    #=====================================================================
    # N : resolution
    #=====================================================================  
    #                  A      a      b     x0      y0    phi
    e = np.array( [ [  1,    .69,   .92,    0,       0,   0 ], 
                    [-.8,  .6624, .8740,    0,  -.0184,   0 ],
                    [-.2,  .1100, .3100,  .22,       0,  -18],
                    [-.2,  .1600, .4100, -.22,       0,   18],
                    [ .1,  .2100, .2500,    0,     .35,   0 ],
                    [ .1,  .0460, .0460,    0,      .1,   0 ],
                    [ .1,  .0460, .0460,    0,     -.1,   0 ],
                    [ .1,  .0460, .0230, -.08,   -.605,   0 ],
                    [ .1,  .0230, .0230,    0,   -.606,   0 ],
                    [ .1,  .0230, .0460,  .06,   -.605,   0 ] ] )
    #
    xn = ((np.arange(0,N)-(N-1)/2) / ((N-1)/2))#.reshape((1,N))
    Xn = np.tile(xn, (N,1))
    Yn = np.rot90(Xn)
    X  = np.zeros((N,N))

    # for each ellipse to be added     
    nn = len(e)   #e.shape[0]
    for i in range(nn):
        A   = e[i,0]
        a2  = e[i,1]**2
        b2  = e[i,2]**2
        x0  = e[i,3]
        y0  = e[i,4]
        phi = e[i,5]*np.pi/180        
        #
        x   = Xn-x0
        y   = Yn-y0
        idd = ((x*np.cos(phi) + y*np.sin(phi))**2)/a2 + ((y*np.cos(phi) - x*np.sin(phi))**2)/b2
        idx = np.where( idd <= 1 )

        # add the amplitude of the ellipse
        X[idx] += A
    #
    idx    = np.where( X < 0 )
    X[idx] = 0

    return X



#=========================================================================
#=========================================================================
#=========================================================================
def tectonic(N):
    # Creates a tectonic phantom of size N x N
    x   = np.zeros((N,N))
    N5  = round(N/5)
    N13 = round(N/13)
    N7  = round(N/7)
    N20 = round(N/20)

    # The right plate
    xr = np.arange(N5,N5+N7+1)-1
    yr = np.arange(5*N13,N+1)-1
    x[np.ix_(xr, yr)] = 0.75

    # The angle of the right plate
    i = N5-1
    for j in range(N20+1):
        if ((j+1)%2) != 0:
            i -= 1
            x[i, 5*N13+j:] = 0.75

    # The left plate before the break
    xr = np.arange(N5,N5+N5+1)-1
    yr = np.arange(1,5*N13+1)-1
    x[np.ix_(xr, yr)] = 1

    # The break from the left plate
    rang = np.arange(5*N13, min(12*N13,N)+1)-1
    for j in rang:
        if ((j+1)%2) != 0:
            xr += 1
        x[xr,j] = 1
    
    return x



#=========================================================================
#=========================================================================
#=========================================================================
def smooth(N, p=4):
    # SMOOTH Creates a 2D test image of a smooth function
    xx    = np.arange(1,N+1)-1
    I, J  = np.meshgrid(xx,xx, indexing='xy')
    sigma = 0.25*N
    #
    c = np.array([[0.6*N, 0.6*N], [0.5*N, 0.3*N], [0.2*N, 0.7*N], [0.8*N, 0.2*N]])
    a = np.array([1, 0.5, 0.7, 0.9])
    x = np.zeros((N,N))
    for i in range(p):
        x += a[i]*np.exp( -(I-c[i,0])**2/(1.2*sigma)**2 - (J-c[i,1])**2/sigma**2 )
    x = x/x.max()
    
    return x



#=========================================================================
#=========================================================================
#=========================================================================
def threephases(N, p=70):
    # THREEPHASES Creates a 2D test image with three different phases
    # 1st
    xx     = np.arange(1,N+1)-1
    I, J   = np.meshgrid(xx,xx, indexing='xy')
    sigma1 = 0.025*N
    c1     = np.random.rand(p,2)*N
    x1     = np.zeros((N,N))
    for i in range(p):
        x1 += np.exp(-abs(I-c1[i,0])**3/(2.5*sigma1)**3 - abs(J-c1[i,1])**3/sigma1**3)

    t1 = 0.35
    x1[x1 < t1]  = 0
    x1[x1 >= t1] = 2

    # 2nd
    sigma2 = 0.03*N
    c2     = np.random.rand(p,2)*N
    x2     = np.zeros((N,N))
    for i in range(p):
        x2 += np.exp(-(I-c2[i,0])**2/(2*sigma2)**2 - (J-c2[i,1])**2/sigma2**2)
    
    t2 = 0.55
    x2[x2 < t2]  = 0
    x2[x2 >= t2] = 1

    # combine the two images
    x = x1 + x2
    x[x == 3] = 1
    x = x/x.max()
    
    return x



#=========================================================================
#=========================================================================
#=========================================================================
def grains(N, numGrains):
    # numGrains = int(round(3*np.sqrt(N)))
    
    # GRAINS Creates a test image of Voronoi cells
    dN        = round(N/10)
    Nbig      = N + 2*dN
    total_dim = Nbig**2

    # random pixels whose coordinates (xG,yG,zG) are the "centre" of the grains
    xG = np.ceil(Nbig*np.random.rand(numGrains,1))
    yG = np.ceil(Nbig*np.random.rand(numGrains,1))

    # set up voxel coordinates for distance computation
    xx   = np.arange(1,Nbig+1)
    X, Y = np.meshgrid(xx,xx, indexing='xy')
    X    = X.flatten(order='F')
    Y    = Y.flatten(order='F')

    # for centre pixel k [xG(k),yG(k),zG(k)] compute the distance to all the 
    # voxels in the box and store the distances in column k.
    distArray = np.zeros((total_dim,numGrains))
    for k in range(numGrains):
        distArray[:,k] = (X-xG[k])**2 + (Y-yG[k])**2

    # determine to which grain each of the voxels belong. This is found as the
    # centre with minimal distance to the given voxel
    minIdx = np.argmin(distArray, axis=1)

    # reshape to 2D, subtract 1 to have 0 as minimal value, extract the
    # middle part of the image, and scale to have 1 as maximum value
    x   = minIdx.reshape(Nbig,Nbig) - 1
    x   = x[np.ix_(dN+np.arange(1,N+1)-1, dN+np.arange(1,N+1)-1)]
    x   = x/x.max()
    
    return x



#=========================================================================
#=========================================================================
#=========================================================================
def ppower(N, relnz=0.65, p=2.6):#relnz=0.65, p=2.3
    #PPOWER Creates a 2D test image with patterns of nonzero pixels
    if N/2 == round(N/2):
        Nodd = False
    else: 
        Nodd = True
        N += 1
    #
    P = np.random.randn(N,N)
    # idx = np.random.permutation(200)
    # idx = (idx[:150])
    # P = P[idx,idx]
    #
    xx   = np.arange(1,N+1)
    I, J = np.meshgrid(xx,xx, indexing='xy')
    #
    U = ( ( (2*I-1)/N - 1)**2 + ( (2*J-1)/N - 1)**2 )**(-p/2)
    F = U*np.exp(2*np.pi*np.sqrt(-1+0j)*P)
    F = abs(np.fft.ifft2(F))
    f = -np.sort(-F.flatten(order='F'))   # 'descend'
    k = round(relnz*N**2)-1
    #
    F[F < f[k]] = 0
    x = F/f[0]
    if Nodd:
        x = F[1:-1,1:-1]
    
    return x