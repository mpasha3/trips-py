import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from numpy import array, diag, dot, maximum, empty, repeat, ones, sum
from numpy.linalg import inv
from trips.utilities.operators import *
##Specify the font
##Latex needs to be installed! If not installed, please comment the following 5 lines
# parameters = {'xtick.labelsize': 12, 'ytick.labelsize': 12,
#           'axes.titlesize': 18, 'axes.labelsize': 18, 'figure.titlesize': 14, 'legend.fontsize': 13}
# plt.rcParams.update(parameters)
import time
import numpy as np
import scipy as sp
import scipy.stats as sps
import scipy.io as spio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import astra
# import phantoms as phantom
from venv import create
import pylops
from scipy.ndimage import convolve
from scipy import sparse
from scipy.ndimage import convolve
import scipy.special as spe
from trips.utilities.testProblems import *
from trips.solvers.gks_all import *
import requests
from scipy import sparse
import numpy as np
import h5py
# functions to generate emoji data are stored in io_l.py
from trips.utilities.io import *
from trips.utilities.operators import *
from trips.solvers.AnisoTV import *
from trips.utilities.helpers import *
(A, b, AA, B, nx, ny, nt, delta) = generate_emoji(noise_level = 0.01, dataset = 30)
b_vec = b.reshape((-1,1))
# L = spatial_derivative_operator(nx, ny, nt)
L = time_derivative_operator(nx, ny, nt)
(x_dynamic_gks, info) = GKS(A, b, L, projection_dim = 2, n_iter = 3, regparam = 0.01, x_true = None)
plot_recstructions_series(x_dynamic_gks, (nx, ny, nt), dynamic = False, testproblem = 'Emoji', geome_x = 1,geome_x_small = 0,  save_imgs= False, save_path='./reconstruction/Emoji')

