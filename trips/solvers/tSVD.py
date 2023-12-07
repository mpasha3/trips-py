#!/usr/bin/env python
"""
Functions which implement decompositions based on Krylov subspaces.
--------------------------------------------------------------------------
Created in 2022 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, Connor Sanderford, and Ugochukwu Obinna Ugwu"
__affiliations__ = 'Tufts University, University of Bath, Arizona State University, and Tufts University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu; connorsanderford@gmail.com; Ugochukwu.Ugwu@tufts.edu"

import os, sys
import numpy as np
sys.path.insert(0, '/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package')

def TruncatedSVD_sol(A, k, b_vec):
  U, S, VT = np.linalg.svd(A)
  S_hat = S[0:k] #extract the first r singular values
  S_hat_mat = np.diag(S_hat) #form a diagonal matrix
  U_temp = U[:, 0:k]
  x_trunc = np.transpose(VT[0:k, :])@np.linalg.inv(S_hat_mat)@np.transpose(U_temp)@b_vec
  return x_trunc