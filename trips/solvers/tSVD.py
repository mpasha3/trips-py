import numpy as np
def TruncatedSVD_sol(A, r, b_vec):
  U, S, VT = np.linalg.svd(A)
  S_hat = S[0:r] #extract the first r singular values
  S_hat_mat = np.diag(S_hat) #form a diagonal matrix
  U_temp = U[:, 0:r]
  x_trunc = np.transpose(VT[0:r, :])@np.linalg.inv(S_hat_mat)@np.transpose(U_temp)@b_vec
  return x_trunc