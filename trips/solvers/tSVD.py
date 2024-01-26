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
from trips.utilities.parameter_selection.gcv import *
from trips.utilities.parameter_selection.discrepancy_principle import *

def TruncatedSVD_sol(A, b, regparam = 'gcv', **kwargs):

  b = b.reshape((-1,1))

  delta = kwargs['delta'] if ('delta' in kwargs) else None
  if (regparam == 'dp') and delta == None:
        raise Exception("""A value for the noise level delta was not provided and the discrepancy principle cannot be applied. 
                    Please supply a value of delta based on the estimated noise level of the problem, or choose the regularization parameter according to gcv.""")
  
  U, S, VT = np.linalg.svd(A)
  if regparam == 'gcv':
    k = generalized_crossvalidation(U, S, VT, b, gcvtype = 'tsvd')
  elif regparam == 'dp':
    k = discrepancy_principle(U, S, VT, b, dptype = 'tsvd', **kwargs)
  else:
    k = regparam # make sure we have checks on the values of k (eventually have them here, if in general we check for a positive scalar value only)
  S_hat = S[0:k] #extract the first r singular values # CHECK IF EXTREMA ARE INCLUSIVE OR EXCLUSIVE
  U_temp = U[:, 0:k]
  x_trunc = np.transpose(VT[0:k, :])@(((np.transpose(U_temp)@b).reshape((-1,1)))/S_hat.reshape((-1,1)))
  return x_trunc, k
