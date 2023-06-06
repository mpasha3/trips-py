import numpy as np
def Tikhonov(A, b_vec, L, x_true, reg_param = 1/5):
    minerror = 100
    while reg_param > 1e-04:
        xTik = np.linalg.solve(A.T@A + reg_param**2*L.T@L, A.T@b_vec)
        error = np.linalg.norm(xTik - x_true)/np.linalg.norm(x_true)
        if error < minerror:
            minerror = error
            minlamb = reg_param
            mingTik = xTik
        reg_param /= 2
    return mingTik  