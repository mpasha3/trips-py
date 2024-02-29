#!/usr/bin/env python
"""
Functions which implement truncated GSVD
--------------------------------------------------------------------------
Created in 2022 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, and Ugochukwu Obinna Ugwu"
__affiliations__ = 'MIT and Tufts University, University of Bath, and Tufts University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; Ugochukwu.Ugwu@tufts.edu"

import os, sys
import numpy as np
from trips.utilities.reg_param.gcv import *
from trips.utilities.reg_param.discrepancy_principle import *
from trips.utilities.decompositions import gsvd

def tGSVD_sol(A, L, b, regparam = 'gcv', **kwargs):

  b_vec = b.reshape((-1,1))

  delta = kwargs['delta'] if ('delta' in kwargs) else None
  if (regparam == 'dp') and delta == None:
        raise Exception("""A value for the noise level delta was not provided and the discrepancy principle cannot be applied. 
                    Please supply a value of delta based on the estimated noise level of the problem, or choose the regularization parameter according to gcv.""")
  
  U, _, Z, C, S = gsvd(A,L) 
  if regparam == 'gcv':
    k = generalized_crossvalidation(U, S, Z, b_vec, gcvtype = 'tgsvd')
  elif regparam == 'dp':
    k = discrepancy_principle(U, C, Z, b_vec, dptype = 'tgsvd', **kwargs)
  else:
    k = regparam 
  Y = np.linalg.inv(Z.T)
  C[:k,:k] = 0
  xsol = Y@C@(U.T@b_vec)
  return (xsol, k)
