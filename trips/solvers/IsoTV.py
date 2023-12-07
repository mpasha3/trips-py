import numpy as np
from scipy.linalg import qr
from scipy.optimize import fminbound
from trips.utilities.decompositions import gsvd
from trips.parameter_selection import *
def l2lq_isoTV_(A, b, L, Ls, q, e, iter, delta, nx, nt):
    Rerr = []
    ml, nl = L.shape
    tol = 0.001
    b = b.reshape(-1,1)
    x = A.T @ b
    v = A.T @ b
    V = v / np.linalg.norm(v)
    AV = A @ V
    Rerr = np.zeros(iter + 1)
    LV = L @ V
    v = A @ x
    u = L @ x
    spacen = int(Ls.shape[0] / 2)
    spacent = spacen * nt
    storemu = []
    saveX = []
    storegrad = []
    storerel_err = []
    for k in range(0, iter):
        x_old = x
        # Modification needed in MMGKS
        #=====================================
        X = x.reshape(nx**2, nt)
        LsX = Ls @ X
        LsX1 = LsX[:spacen, :]
        LsX2 = LsX[spacen:2*spacen, :]
        weightx = (LsX1**2 + LsX2**2 + e**2)**((q-2) / 4)
        weightx = np.concatenate((weightx.flatten(), weightx.flatten()))
        weightt = (u[2*spacent:]**2 + e**2)**((q-2) / 4)
        wr = np.concatenate((weightx.reshape(-1,1), weightt))
        #=====================================
        AA = AV
        LL = LV * wr
        QA, RA = qr(AA, mode='economic')
        _, RL = qr(LL, mode='economic')
        #print(QA.shape,RA.shape,RL.shape)
        mu = gcvd(RA,RL,QA.T @ b) # use gsvd based gcv
        storemu.append(mu)
        y,_,_,_ = np.linalg.lstsq(np.concatenate((RA, np.sqrt(mu) * RL)), np.concatenate((QA.T@ b, np.zeros((RL.shape[0],1)))),rcond=None)
        x = V@y
        saveX.append(x)
        if k >= RL.shape[0]:
            break
        v = AV@y
        v = v - b
        u = LV @ y
        ra = AV @ y - b
        ra = A.T @ ra
        rb = wr * (LV @ y)
        rb = L.T @ rb
        r = ra + mu * rb
        r = r - V @ (V.T @ r)
        r = r - V @ (V.T @ r)
        storegrad.append(np.linalg.norm(r))
        storerel_err.append(np.linalg.norm(x_old - x) / np.linalg.norm(x_old))
        vn = r / np.linalg.norm(r)
        V = np.column_stack((V, vn))
        Avn = A @ vn
        AV = np.column_stack((AV, Avn))
        Lvn = vn
        Lvn = L*vn
        LV = np.column_stack((LV, Lvn))
    return x, np.array(storerel_err),np.array(storemu)
def gcvd(A,L,b):
    U, _, _, S, La = gsvd(A,L)
    # print(b.shape)
    # print(U.shape)
    bhat = U.T @ b
    l = np.diag(La)
    s = np.diag(S)
    mu = fminbound(gcv_funct, 0, 100, args=(s, l, bhat))
    return mu
def gcv_funct(mu, s, l, bhat):
    num = (l**2 * bhat / (s**2 + mu * l**2))**2
    num = np.sum(num)
    den = (l**2 / (s**2 + mu * l**2))**2
    den = np.sum(den)
    G = num / den
    return G
# Script for running IsoTV, plot
(A, b, AA, B, nx, ny, nt, delta) = generate_emoji(noise_level = 0.0, dataset = 30)
print(A.shape)
L = spatial_derivative_operator(nx, ny, nt)
Ls = first_derivative_operator_2d(nx, ny)
q = 1
e = 1
iter = 30
x, rel_err,paramu = l2lq_isoTV_(A, b, L, Ls, q, e, iter, delta, nx, nt)
xx = x.reshape((nt,nx,nx))
plt.imshow(xx[20,:,:])
plt.show()
fig,ax = plt.subplots(figsize=(4.5, 5))
plt.plot(rel_err,'-*r')
plt.xlabel('Iterations')
plt.ylabel('Relative Error')
plt.legend(title='IsoTV',fontsize=12)
plt.show()
fig,ax = plt.subplots(figsize=(4.5, 5))
plt.plot(paramu,'-*r')
plt.xlabel('Iterations')
plt.ylabel('Regularization Parameter')
plt.legend(title='IsoTV',fontsize=12)
plt.show()