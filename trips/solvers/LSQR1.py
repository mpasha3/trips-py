# -*- coding: utf-8 -*-
#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from cil.optimisation.algorithms import Algorithm
import warnings
import logging
import numpy as np
from numpy import linalg as LA


class LSQR(Algorithm):

    '''Least-Squares QR algorithm, mathematically equivalent to CGLS 
    
    Problem:  
    .. math::
      \min || A x - b ||^2_2
    
    |
    Parameters :
        
      :parameter operator : Linear operator for the inverse problem
      :parameter initial : Initial guess ( Default initial = 0)
      :parameter data : Acquired data to reconstruct       
      :parameter tolerance: Tolerance/ Stopping Criterion to end LSQR algorithm
      
    '''
    # Reference:
    #     https://web.stanford.edu/group/SOL/software/cgls/
    
    def __init__(self, initial=None, operator=None, data=None, tolerance=1e-6, **kwargs):
        '''initialisation of the algorithm
        :param operator : Linear operator for the inverse problem
        :param initial : Initial guess ( Default initial = 0)
        :param data : Acquired data to reconstruct       
        :param tolerance: Tolerance/ Stopping Criterion to end CGLS algorithm
        '''
        super(LSQR, self).__init__(**kwargs)

        if initial is None and operator is not None:
            initial = operator.domain_geometry().allocate(0.0)
        if initial is not None and operator is not None and data is not None:
            self.set_up(initial=initial, operator=operator, data=data, tolerance=tolerance)

    def set_up(self, initial, operator, data, tolerance=1e-6):
        '''initialisation of the algorithm
        :param operator: Linear operator for the inverse problem
        :param initial: Initial guess ( Default initial = 0)
        :param data: Acquired data to reconstruct       
        :param tolerance: Tolerance/ Stopping Criterion to end CGLS algorithm
        '''
        logging.info("{} setting up".format(self.__class__.__name__, ))
        
        self.x0 = initial.copy()
        self.operator = operator
        self.tolerance = tolerance
        
        d = self.operator.adjoint(data).as_array() # self.operator.adjoint(data, out=self.d)
        n = d.size
        m = data.as_array().size

        self.r = data - self.operator.direct(self.x0)
        self.beta = self.r.norm()
        self.nrmb = data.norm() # norm(data) ###???
        # allocate memory
        # B, V, U rhs should be BlockDataContainers
        B = np.zeros((self.max_iteration+1,self.max_iteration)) ###???
        #print(B.shape)
        V = np.zeros((n,self.max_iteration)) ###???
        #print(V.shape)
        U = np.zeros((m,self.max_iteration+1)) ###???
        #print(U.shape)
        rhs = np.zeros(self.max_iteration+1)        
        rhs[0] = self.beta
        self.u = self.r.copy()
        U[:,0] = self.u.as_array().flatten()/self.beta
        #print(U.shape)
        #print(self.u.as_array().flatten()/self.beta)
        #print(U)
        
        self.B = B
        self.V = V
        self.U = U
        self.rhs = rhs
        self.betaB = None
        
        self.s = self.operator.adjoint(self.r)
        self.norms0 = self.s.norm()
        self.norms = self.s.norm()
        self.normx = self.x0.norm()
        
        self.y = 0
        
        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))
     
    def _convert_vector_to_AcquisitionData(self, vector):
        op_range = self.operator.range_geometry()
        output = op_range.allocate()
        vv = np.reshape(vector, output.shape)
        output.fill(vv)
        
        return output
        
        
    def _convert_vector_to_ImageData(self, vector):
        op_domain = self.operator.domain_geometry()
        output = op_domain.allocate()
        vv = np.reshape(vector, output.shape)
        output.fill(vv)
        
        return output
        
        
    def update(self):
        '''single iteration'''
        k = self.iteration # k is the number of iterations (starting from k = 0)
        #print(k)
        # reorth used to decide wether reorthogonalisation of the basis vectors should be used
        w = self.operator.adjoint(
            self._convert_vector_to_AcquisitionData(self.U[:,k])
        ).as_array().flatten()
        #print(w.shape)
        
        
        if k>0:    
            w -= self.betaB*self.V[:,k-1] ### SG
        # if reorth:
        #     for jj in range(k-1):
        #         w = w - (V[:,jj].transpose()*w)*V[:,jj] # np.dot?
        alpha = LA.norm(w)
        # if abs(alpha) <= np.finfo(float).eps:
        #         print('Golub-Kahan bidiagonalization breaks down') 
        #         break
        self.V[:,k] = w/alpha
        u = self.V[:,k] # relabel 
        u = self.operator.direct(
                                self._convert_vector_to_ImageData(u)
                                ).as_array().flatten()
        u -= alpha*self.U[:,k]
        # if reorth:
        #     for jj in range(k-1):
        #         u = u - (U[:,jj].transpose()*u)*U[:,jj] # np.dot?
        beta = LA.norm(u) # check vector norm / Fr
        # if abs(beta) <= np.finfo(float).eps:
        #         print('Golub-Kahan bidiagonalization breaks down')
        #         break
        self.U[:,k+1] = u/beta
        self.B[k,k] = alpha # store diagonals for efficiency
        self.B[k+1,k] = beta
        self.betaB = beta ### SG
        #print(self.B)
        #print(self.rhs)
        rhsk = self.rhs[0:k+2] # current projected rhs
        #print(rhsk.shape)
        if abs(alpha) <= 1e-12 or abs(beta) <= 1e-12: # substitute with tolerance
                print('Golub-Kahan bidiagonalization breaks down')                         
        Bk = self.B[0:k+2,0:k+1]
        #print(Bk.shape)
        #print("====")
        uk, sk, vkt = LA.svd(Bk, full_matrices=True) # scipy sparse SVDS
        #print(uk.shape)
        #print(sk.shape)
        #print(vkt.shape)
        rhskhat = np.matmul(uk.transpose(),rhsk)
        #print(rhskhat.shape)
        lsqr_res = abs(rhskhat[k+1])
        #print(lsqr_res)
        
        #print("##############")
        Dk = sk**2
        
        #print(sk.shape)
        #print(rhskhat.shape)
        rhskhat = sk * rhskhat[0:k+1] # check matrix multiply diag(s)
        #rhskhat = np.multiply(sk,rhskhat[0:k+1].transpose())
        yhat = rhskhat/Dk # check component-wise division by svs
        #print(yhat.shape)
        y = np.matmul(vkt.transpose(), yhat)  #yhat.transpose())
        #print(y.shape)
        dx = np.matmul(self.V[:,0:k+1], y)
        self.x = self.x0 + self._convert_vector_to_ImageData(dx) # change to direct access
        self.y = y
        #print(self.y)
                    
    def update_objective(self):
        k = self.iteration
        #print(k)
        # Temporary fix to handle case of computing objective before starting iterations.
        if k == -1:
            a = -10
        else:
            Bk = self.B[0:k+1,0:k]
            y = self.y
            rhsk = self.rhs[0:k+1]
            
            #print(Bk)
            #print(y)
            # handle particular cases here (e.g., if Bk = [])
            a = LA.norm(rhsk - np.matmul(Bk,y))**2
            if a is np.nan:
                raise StopIteration()
        #print(a)
        self.loss.append(a)
        
    def should_stop(self):
        '''stopping criterion'''
        return self.flag() or self.max_iteration_stop_criterion()
 
    def flag(self):
        '''returns whether the tolerance has been reached'''
        flag  = (self.norms <= self.norms0 * self.tolerance) or (self.normx * self.tolerance >= 1)

        if flag:
            self.update_objective()
            if self.iteration > self._iteration[-1]:
                print (self.verbose_output())
            print('Tolerance is reached: {}'.format(self.tolerance))

        return flag