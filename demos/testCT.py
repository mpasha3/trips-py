__authors__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha and Connor Sanderford"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com and csanderf@asu.edu; connorsanderford@gmail.com"

import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.io as spio
import scipy.sparse.linalg as spsla
from trips.testProblems import *
import sys, os
sys.path.insert(0, '/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package/demos')
sizex = 128
sizey = 128
Tomo = Tomography(sizex = sizex, sizey = sizey, dataset = 60)

xtrue = Tomo.generate_true(test_problem = 'CT60')
xtrue.shape
np.sqrt(16384)
plt.imshow(xtrue.reshape((128,128), order = 'F'))
plt.axis('off')

b_true = Tomo.generate_data(x = xtrue, proj_type = 1, flag = 'given_operator', dataset = 60)
# plt.imshow(b.reshape((60, 100)))

# mu_obs = np.zeros((6000))      # mean of noise
# noise = np.random.randn(b_true.shape[0]).reshape(-1,1)
# e = 0.01 * np.linalg.norm(b_true) / np.linalg.norm(noise) * noise
# e = e.reshape(-1,1)
# b_true = b_true.reshape(-1,1)
# b = b_true + e # add noise
# b_meas = b_true + e
# b_meas_i = b_meas.reshape((60, 100), order='F')


(b_noise, delta) = Tomo.add_noise(b_true, 'Gaussian', 0.01)

plt.imshow(b_noise.reshape((60, 100)))
