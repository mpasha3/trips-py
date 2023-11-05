from trips.testProblems import *
from trips.solvers.Hybrid_LSQR_new import *
from trips.decompositions import golub_kahan_SG
D1D = Deblurring1D()
N = 64
A = D1D.forward_Op_matrix_1D(3, N)
x_true = D1D.gen_xtrue(N, test = 'piecewise')
b = A@x_true
plt.plot(x_true)
D1D.plot_data(b)
b = np.reshape(b, (b.size,1))
n = A.shape[1]
n_iter = 4
regparam = 'gcv'
beta = np.linalg.norm(b)
U = b/beta
B = np.empty(1)
V = np.empty((n,1))
RegParam = np.zeros(n_iter,)
print(U.shape)
# hybrid_lsqr_new(A, b, n_iter, regparam = 'gcv', **kwargs): # what's the naming convention here?
# 
x, U, B, V, RegParam = hybrid_lsqr(A, b, n_iter = 5, regparam = 'gcv')
plt.imshow(x)