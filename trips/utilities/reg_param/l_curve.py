import numpy as np
from scipy import optimize as op

def compute_x_l(l, C, D, B, E, b, d=None):
    """
    Solves the linear system: (C + l D) x_l = B^T b + l E^T d
    which arises from minimizing: ||Ax - b||^2 + l ||Lx - d||^2,
    where C = A^T A, D = L^T L, B = A^T, E = L^T.

    Parameters:
    - l: Regularization parameter λ
    - C: A^T A
    - D: L^T L
    - B: A^T
    - E: L^T
    - b, d: data vectors (if d is None, assume zero vector)

    Returns:
    - x_l: solution to the regularized system
    """
    if d is None:
        d = np.zeros(E.shape[0])
    lhs = C + l * D
    rhs = B.T @ b + l * E.T @ d
    x_l = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return x_l

def compute_dx_l_dl(l, C, D, E, x_l, d=None):
    """
    Computes the derivative dx_l/dλ, which describes how x_l changes with λ.

    Formula:
    dx_l/dλ = - (C + λD)^{-1} D x_l + (C + λD)^{-1} E^T d

    Parameters:
    - x_l: previously computed solution at λ
    - Other inputs as in compute_x_l

    Returns:
    - dx_l_dl: the derivative of x_l with respect to λ
    """
    if d is None:
        d = np.zeros((E.shape[0],1))
    lhs = C + l * D
    term1 = D @ x_l
    term2 = E.T @ d
    dx_l_dl = np.linalg.lstsq(lhs, term1 - term2, rcond=None)[0]
    return -dx_l_dl

def compute_d2x_l_dl2(l, C, D, E, x_l, dx_l_dl, d=None):
    """
    Computes the second derivative d²x_l/dλ² used for curvature calculation.

    Parameters:
    - dx_l_dl: first derivative
    - Other inputs as in compute_x_l

    Returns:
    - d2x_l_dl2: second derivative of x_l with respect to λ
    """
    if d is None:
        d = np.zeros((E.shape[0],1))
    lhs = C + l * D
    D_x_l = D @ x_l
    D_dx_l_dl = D @ dx_l_dl
    inv4 = np.linalg.lstsq(lhs, D_x_l, rcond=None)[0]
    inv3 = np.linalg.lstsq(lhs, D_dx_l_dl - D @ inv4, rcond=None)[0]
    return 2 * inv3

def term_prime(l, A, B, C, D, E, b, c, d=None):
    """
    Computes the derivative of f(λ) = ||A x_λ - c||^2 with respect to λ.

    Parameters:
    - A: operator for residual being differentiated
    - B: original operator used in fitting (usually same as A)
    - C, D: A^T A, L^T L
    - E: L^T
    - b, c: data vectors (c = target for fidelity or regularization)
    - d: regularization target

    Returns:
    - f'(λ)
    """
    if d is None:
        d = np.zeros((E.shape[0],1))
    x_l = compute_x_l(l, C, D, B, E, b, d)
    dx_l_dl = compute_dx_l_dl(l, C, D, E, x_l, d)
    f_l = A @ x_l
    f_prime_l = A @ dx_l_dl
    return 2 * (f_l - c).T @ f_prime_l

def term_prime_prime(l, A, B, C, D, E, b, c, d=None):
    """
    Computes the second derivative of f(λ) = ||A x_λ - c||^2 with respect to λ.

    Returns:
    - f''(λ)
    """
    if d is None:
        d = np.zeros((E.shape[0],1))

    x_l = compute_x_l(l, C, D, B, E, b, d)
    dx_l_dl = compute_dx_l_dl(l, C, D, E, x_l, d)
    d2x_l_dl2 = compute_d2x_l_dl2(l, C, D, E, x_l, dx_l_dl, d)
    f_l = A @ x_l
    f_prime_l = A @ dx_l_dl
    f_prime_prime_l = A @ d2x_l_dl2
    t1 = f_prime_l.T @ f_prime_l
    t2 = (f_l - c).T @ f_prime_prime_l
    return 2 * (t1 + t2)

def fid_prime(l, A, L, b, d=None):
    """
    First derivative of fidelity term: f(λ) = ||Ax_λ - b||^2
    """
    A2 = A.T @ A
    L2 = L.T @ L
    #print(term_prime(l, A, A, A2, L2, L, b, b, d))
    return term_prime(l, A, A, A2, L2, L, b, b, d).item()

def reg_prime(l, A, L, b, d=None):
    """
    First derivative of regularization term: g(λ) = ||Lx_λ - d||^2
    """
    if d is None:
        d = np.zeros((L.shape[0],1))
    A2 = A.T @ A
    L2 = L.T @ L
    #print(L.shape, A.shape,b.shape,d.shape)
    #print(term_prime(l, L, A, A2, L2, L, b, d, d))
    return term_prime(l, L, A, A2, L2, L, b, d, d).item()

def fid_prime_prime(l, A, L, b, d=None):
    """
    Second derivative of regularization term: g(λ) = ||Lx_λ - d||^2
    """
    A2 = A.T @ A
    L2 = L.T @ L
    return term_prime_prime(l, A, A, A2, L2, L, b, b, d).item()

def reg_prime_prime(l, A, L, b, d=None):
    """
    Second derivative of regularization term: g''(λ)
    """
    if d is None:
        d = np.zeros((L.shape[0],1))
    A2 = A.T @ A
    L2 = L.T @ L
    return term_prime_prime(l, L, A, A2, L2, L, b, d, d).item()

def curvature(l, A, L, b, d=None):
    """
    Computes the curvature κ(λ) of the L-curve at a given λ.

    Formula:
    κ(λ) = [ -g'(λ) f''(λ) + f'(λ) g''(λ) ] / ( [g'(λ)^2 + f'(λ)^2]^(3/2) )

    Where:
    - f(λ) = ||Ax_λ - b||^2 (fidelity)
    - g(λ) = ||Lx_λ - d||^2 (regularization)

    Returns:
    - κ(λ)
    """
    num = (-reg_prime(l, A, L, b, d) * fid_prime_prime(l, A, L, b, d) +
           fid_prime(l, A, L, b, d) * reg_prime_prime(l, A, L, b, d))
    denom = (reg_prime(l, A, L, b, d) ** 2 + fid_prime(l, A, L, b, d) ** 2) ** 1.5
    return num / denom

def l_curve(A, L, b, d=None):
    """
    Parameters:
    - A: forward model matrix
    - L: regularization matrix
    - b: observed data
    - d: target regularization vector (e.g., 0 or prior estimate)

    Returns:
    - λ that maximizes curvature κ(λ)
    """
    l_func = lambda l: -1 * curvature(l, A, L, b, d)
    lambdah = op.fminbound(func=l_func, x1=1e-9, x2=2, xtol=1e-12, maxfun=1000, full_output=0, disp=0)
    return lambdah