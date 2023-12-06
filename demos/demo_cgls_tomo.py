import os, sys
sys.path.insert(0, '/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package')
from trips.solvers.tSVD import *
from trips.utilities.testProblems import Deblurring
import matplotlib.pyplot as plt
from trips.utilities.helpers import *
from trips.solvers.CGLS import *
# Tomography test problem
imagesize_x = 64
imagesize_y = 64
Tomo = Tomography(sizex = imagesize_x, sizey = imagesize_y)
# To generate a small scale problem, set generate_matrix = True, otherwise generate_matrix = False
generate_matrix = True
if generate_matrix == True:
        A = Tomo.forward_Op_mat(imagesize_x, imagesize_y)
else:
        A = Tomo.forward_Op(imagesize_x, imagesize_y)
testproblem = 'ppower'
x_true = Tomo.generate_true(test_problem = testproblem)
shape = (np.int0(np.sqrt(x_true.shape[0])), np.int0(np.sqrt(x_true.shape[0])))
b_true = Tomo.generate_data(x_true, 1, 'no_given_operator', 60)
(b, delta) = Tomo.add_noise(b_true, 'Gaussian', noise_level= 0.01)
Tomo.plot_data(b)
Tomo.plot_rec(x_true.reshape((shape), order = 'F'), save_imgs = False, save_path='./saveImagesTomography'+'rec')
b_vec = b.reshape((-1,1))
x_0 = A.T@b_vec#x_true.reshape((-1,1))
x, k = CGLS(A, b_vec, x_0, maxit = 10, tol = 0.001)
plt.imshow(x.reshape((imagesize_x, imagesize_y)))

