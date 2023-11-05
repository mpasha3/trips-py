import os, sys
sys.path.insert(0, '/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package')
from trips.solvers.tSVD import *
from trips.testProblems import Deblurring
import matplotlib.pyplot as plt
from trips.helpers import *
from trips.solvers.CGLS import *
# Deblurring example test problem
Deblur = Deblurring()
# In the class Deblurring we have can define the type of problem to be used.
generate_matrix = True #Defines a blurring operator where the forward operator matrix is formed explicitly
imagesize_x = 64 # Define the first dimension of the image
imagesize_y = 64 # Defines the second dimension of the image
spread = 1 # The PSF parameter
choose_image = 'pattern1' #The choice of the image
if generate_matrix == True:
        size = imagesize_x
        shape = (size, size)
        spreadnew = (spread, spread)
        A = Deblur.forward_Op_matrix(spreadnew, shape, imagesize_x, imagesize_y)
(x_true, nx, ny) = Deblur.generate_true(choose_image)
b_true = Deblur.generate_data(x_true, generate_matrix)
(b, delta) = Deblur.add_noise(b_true, 'Gaussian', noise_level = 0.01)
Deblur.plot_rec(x_true.reshape((shape), order = 'F'), save_imgs = False, save_path='./saveImagesDeblurring'+'rec'+choose_image)
Deblur.plot_data(b.reshape((shape), order = 'F'), save_imgs = False, save_path='./saveImagesDeblurring'+'data'+choose_image)
b_vec = b.reshape((-1,1))
x_0 = A.T@b_vec
(x, x_history, k) = TP_cgls(A, b_vec, x_0, max_iter = 30, tol = 0.001)
plt.imshow(x.reshape((imagesize_x, imagesize_y)))

