import numpy as np
import numpy.linalg as la


"""
pdfp
"""
def current_sparsity(l, l_len, kappa):

    n_big = (l >= kappa).sum()
    cs = n_big/l_len

    return cs



def id_minus_thresh(l, alpha):

    l_out = np.where(l > alpha, alpha, l)
    l_out = np.where(l < -1*alpha, -1*alpha, l_out)

    return l_out





def pdfp(A, b, L, **kwargs):

    lambdah = kwargs['lambdah']
    gamma = kwargs['gamma']
    maxiter = kwargs['maxiter']
    norm_tol = kwargs['norm_tol']
    sparsity_tol = kwargs['sparsity_tol']

    sparsity_prior = kwargs['sparsity_prior']
    psi = kwargs['psi']
    omega = kwargs['omega']
    kappa = kwargs['kappa']

    f_len = A.shape[1]
    f = np.zeros((f_len, 1))
    l = L @ f
    fSz = l.shape[0]

    l_len = l.shape[0] * l.shape[1]

    rel_change = np.full((maxiter+2), np.nan)
    data_fit = np.full((maxiter+1), np.nan)
    l1_norm = np.full((maxiter+1), np.nan)
    alphas = np.full((maxiter+1), np.nan)
    sparsity = np.full((maxiter+1), np.nan)
    e = 1

    alpha = psi*(10**(-5))
    beta = omega*alpha
    iter = 1
    alphas[0] = alpha

    while (iter <= maxiter) and ( (rel_change[iter] > norm_tol) or ( np.abs(e) > sparsity_tol ) ):

        f_old = f
        Af = A @ f
        dif = Af - b
        BP = A.T @ dif

        data_fit[iter] = la.norm(dif) / la.norm(b)

        d = np.maximum(0, f - gamma*BP - lambdah * L.T @ l)
        Ld = L @ d
        l = l + Ld
        l = id_minus_thresh(l, alpha*gamma / lambdah)
        f = np.maximum(0, f - gamma*BP - lambdah* (L.T @ l))

        Lf = L @ f
        spar = current_sparsity(Lf, l_len, kappa)

        sparsity[iter] = spar
        l1_norm[iter] = np.abs(Lf).sum()

        rel_change[iter+1] = la.norm(f - f_old) / la.norm(f_old)

        e_old = e;
        e = spar - sparsity_prior

        if np.sign(e) != np.sign(e_old):
            beta = beta * (1 - np.abs(e - e_old))

        alpha = max(0, alpha + beta*e)
        alphas[iter] = alpha

        iter = iter + 1

    return f


if __name__ == "__main__":

    A = np.random.rand(10, 10)
    b = np.random.rand(10, 1)

    I = np.random.rand(10,10)


    out = pdfp(A, b, I, lambdah=1, gamma=1, maxiter=11, norm_tol = 0.01, sparsity_tol = 0.01, sparsity_prior = 0.1, psi=1, omega=1, kappa=10**(-7))

    print(out)








