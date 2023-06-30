import numpy as np
from ..decompositions import generalized_golub_kahan, arnoldi
from ..parameter_selection.gcv import generalized_crossvalidation
from ..parameter_selection.discrepancy_principle import discrepancy_principle
from ..utils import smoothed_holder_weights
from collections.abc import Iterable
def Tikhonov(A, b, L, x_true, regparam = 'gcv', **kwargs):

    if max(A.shape[0], A.shape[1]) >= 100:

        raise Warning("For large observation matrices A, this method will be slow. Consider using other solvers.")

    if regparam in ['gcv', 'GCV', 'Gcv']:
        lambdah = generalized_crossvalidation(A, b, L)['x'].item() # find ideal lambda by crossvalidation
        xTikh = np.linalg.solve(A.T@A + lambdah*L.T@L, A.T@b)
    elif regparam in ['DP', 'dp', 'Dp', 'Discrepancy Principle', 'Discrepancy principle', 'discrepancy principle']:
        lambdah = discrepancy_principle(A, b, L, **kwargs)['x'].item() # find ideal lambdas by crossvalidation
        xTikh = np.linalg.solve(A.T@A + lambdah*L.T@L, A.T@b)
    else:
        lambdah = regparam
        xTikh = np.linalg.solve(A.T@A + lambdah*L.T@L, A.T@b)
    return xTikh  