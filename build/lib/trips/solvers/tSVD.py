#!/usr/bin/env python
"""
Functions which implement truncated SVD
--------------------------------------------------------------------------
Created in 2022 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha and Silvia Gazzola"
__affiliations__ = 'MIT and Tufts University, University of Bath'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk;"

import os, sys
import numpy as np
from trips.utilities.reg_param.gcv import *
from trips.utilities.reg_param.discrepancy_principle import *

def tSVD_sol(A, b, regparam = 'gcv', **kwargs):

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
    k = regparam # make sure we have checks on the values of k (eventually have them here, if in general we check for a positive scalar
  S_hat = S[0:k] #extract the first r singular values 
  U_temp = U[:, 0:k]
  x_trunc = np.transpose(VT[0:k, :])@(((np.transpose(U_temp)@b).reshape((-1,1)))/S_hat.reshape((-1,1)))
  return x_trunc, k
