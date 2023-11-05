import math
import numpy as np
import astra
import matplotlib.pyplot as plt
import mat73

# T H   2022
def createASTRAoperator_fanbeam(N, SOD, ODD, pixelSize, N_det, theta):
    """
    # Function for building the OpTomo forward operator

    N         : Reconstruction resolution (N x N image)
    SOD       : Source-to-Origin Distance
    ODD       : Origin-to-Detector Distance
    pixelSize : Width of the detector element after binning
    N_det     : Number of detector elements
    theta     : Projection angles (in degrees)
    """

    M = (SOD + ODD) / SOD # Geometric magnification
    effPixelSize = pixelSize / M # Effective pixelSize
    anglesRad = theta*(math.pi/180)
   
    projGeom = astra.create_proj_geom('fanflat', M, N_det, anglesRad, SOD/effPixelSize, ODD/effPixelSize)
    volGeom = astra.create_vol_geom(N,N)
    projId = astra.create_projector('strip_fanflat', projGeom, volGeom)
   
    return astra.OpTomo(projId)

def createASTRAmatrix_fanbeam(N, SOD, ODD, pixelSize, N_det, theta):
    """
    # Function for building the forward operator matrix

    N         : Reconstruction resolution (N x N image)
    SOD       : Source-to-Origin Distance
    ODD       : Origin-to-Detector Distance
    pixelSize : Width of the detector element after binning
    N_det     : Number of detector elements
    theta     : Projection angles (in degrees)
    """

    M = (SOD + ODD) / SOD # Geometric magnification
    effPixelSize = pixelSize / M # Effective pixelSize
    anglesRad = theta*(math.pi/180)
   
    projGeom = astra.create_proj_geom('fanflat', M, N_det, anglesRad, SOD/effPixelSize, ODD/effPixelSize)
    volGeom = astra.create_vol_geom(N,N)
    projId = astra.create_projector('strip_fanflat', projGeom, volGeom)
    matrixId = astra.projector.matrix(projId)
   
    return astra.matrix.get(matrixId)
# initialize everything
N = 560
N_det = 560
N_theta = 720 
theta = np.linspace(0,360,N_theta,endpoint=False)

# Load measurement data as sinogram
data = mat73.loadmat('/Users/mirjetapasha/Documents/Research_Projects/Helsinki/Pasha/2023-09-12_lego_singer/corrected/2023-09-12_lego_singer_ct_project_2d_binning_4.mat') # !!! Change the file path to suit yourself !!!
CtData = data["CtData"]
m = CtData["sinogram"]

# Load parameters
param = CtData["parameters"]
binningFactor = param["binningPost"]
SOD = param["distanceSourceOrigin"]
SDD = param["distanceSourceDetector"]
ODD = SDD - SOD # Origin-detector-distance
pixelSize = param["pixelSize"]*binningFactor

noise_level = 0.01
b_true = m
noise = np.random.randn(b_true.shape[0]).reshape(-1,1)
e = noise_level * np.linalg.norm(b_true) / np.linalg.norm(noise) * noise
b = b_true + e

# Create ASTRA forward operator
Aop = createASTRAoperator_fanbeam(N, SOD, ODD, pixelSize, N_det, theta)
BPop = Aop.T*m # Backproject with operator (sinogram can be flattened or matrix)
plt.matshow(BPop.reshape((N,N)))
plt.title('Backprojection using linear operator')
plt.show

from trips.solvers.GKS import *
# (x, x_history, lambdah, lambda_history, res_history, rre_history) = TP_gks(Aop, b_vec, D, regparam=regvals, projection_dim=3, n_iter=80, x_true=x_0, tol=10**(-16))
imagesize_x = 560
imagesize_y = 560
x_0 = Aop.T*m
I = pylops.Identity(imagesize_x*imagesize_y) # identity operator
D = first_derivative_operator(n=imagesize_x*imagesize_y) # first derivative operator
# first, run several iterations with crossvalidation.
solver = GKSClass(projection_dim=3, regparam='gcv', dp_stop=False, tol=10**(-16))
solver.run(Aop, b, D, iter=10, x_true = x_0)
# then use the estimated to initialize a sequence of values.
# regvals = [solver.lambda_history[-1]*(0.5**(x)) for x in range(0,80)]
# solver.change_regparam(regvals)
# solver.run(blur_operator, b, D, iter=80, x_true=x_true, warm_start=True)

plt.imshow(solver.x)