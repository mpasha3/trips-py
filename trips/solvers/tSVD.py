#!/usr/bin/env python
"""
Definition of test problems
--------------------------------------------------------------------------
Created December 10, 2022 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha and Connor Sanderford"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com and csanderf@asu.edu; connorsanderford@gmail.com"
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