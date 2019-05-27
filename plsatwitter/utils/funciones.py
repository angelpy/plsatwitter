# -*- coding: latin-1 -*-
"""
    #########################
    Funciones (``utils.funciones``)
    #########################
    
    Funciones y reglas matemáticas utilizadas por el programa del TFM.
"""
import math
from math import sqrt
from sklearn.utils.extmath import randomized_svd, squared_norm
import numpy as np
from numpy.linalg import norm as norm_np
from numpy.linalg import inv
import scipy.sparse as sp


def nmf2plsa(W, H):
    """
    Realiza una transformación de las matrices utilizadas por NMF 
    a las utilizadas por PLSA.
    Args:
        W: array-like
            Matriz de bases.
        H: array-like
            Matriz de coeficientes
    Returns:
        P: array-like
            P(t,d)
        Q: array-like
            P(t|z)
        C: array-like
            P(z)
        S: array-like
            P(d|z)
    """
    if sp.issparse(W):
        W = W.toarray()
    if sp.issparse(H):
        H = H.toarray()
#    R = np.diag(np.sum(W, axis=0))
#    T = np.diag(np.sum(H, axis=1))
#    Q = np.dot(W, np.linalg.inv(R))
#    Q = Q / np.sum(Q, axis=0)
#    C = np.dot(R, T)
#    C = C / np.sum(C)
#    S = np.dot(np.linalg.inv(T), H)
#    S = S / np.sum(S, axis=1)[:, np.newaxis]
#    P = Q.dot(C).dot(S)
    A = np.diag(W.sum(0))
    B = np.diag(H.sum(1))
    P1 = W @ inv(A)
    P2 = inv(B) @ H
    P3 = A @ B
    P = P1 @ P3 @ P2
    return P, P1, P2, P3

def norm(x):
    """Dot product-based Euclidean norm implementation
    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm
    """
    return math.sqrt(squared_norm(x))

def kl_div(X1, X2):
    """
    Obtiene la divergencia de Kullback-Leibler
    entre dos funciones de probabilidad.
    Args:
        X1: array-like
        X2: array-like
    Returns: double
        Divergencia KL
    """
    tol = 1e-7
    if sp.issparse(X1):
        X1 = X1.toarray()
    if sp.issparse(X2):
        X2 = X2.toarray()
    X1 = X1.ravel()
    X2 = X2.ravel()
    indices = X1 > tol
    X2 = X2[indices]
    X1 = X1[indices]
    X2[X2 <= tol] = tol
#    eps= 1e-7
#    if sp.issparse(X1):
#        X1 = X1.toarray()
#    if sp.issparse(X2):
#        X2 = X2.toarray()
#    X1 += eps
#    X2 += eps
#    X1, X2 = X1.ravel(), X2.ravel()
    return np.sum(X1*np.log(X1/X2) + X2 - X1)

def svd_init(X, n_components = 10, random_state = None, eps = 1e-6):
    """
    @param X: numpy.ndarray
        Matriz objetivo.
    @param n_components: int
        Rango de la reducción.
    @param random_state:
    @param eps: int
        Tolerancia admitida
    @return: (numpy.ndarray, numpy.ndarray)
        Valor inicial de las matrices utilizadas en NMF (W, H).
    """
    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    return W, H

def svd_tfm(X, n_components = 10, criteria = 'abs'):
    """
    Proporciona una factorizacion inicial para las matrices W y H.
    Args:
        X: array-like
            Matriz objetivo
        n_components: int
            NÂº de componentes
        criteria: 'abs' | 'count'

    Returns:
        W, H
    """
    if sp.issparse(X):
        X = X.toarray()
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    m, n = X.shape
    W = np.zeros((m, n_components))
    H = np.zeros((n_components, n))
    if criteria == 'abs':
        for k in range(n_components):
            sum_pos_u = np.abs(u[:, k][u[:, k] >= 0]).sum()
            sum_pos_v = np.abs(vh[:, k][vh[:, k] >= 0]).sum()
            sum_neg_u = np.abs(u[:, k][u[:, k] < 0]).sum()
            sum_neg_v = np.abs(vh[:, k][vh[:, k] < 0]).sum()
            sum_pos = sum_pos_u + sum_pos_v
            sum_neg = sum_neg_u + sum_neg_v
            if sum_pos < sum_neg:
                u[:, k] *= -1
                vh[:, k] *= -1
            u[:, k][u[:, k] < 0] = 1e-7
            vh[:, k][vh[:, k] < 0] = 1e-7
            W[:, k] = (sqrt(s[k]) * u[:, k]).ravel()
            H[k, :] = (sqrt(s[k]) * vh[:, k]).ravel()
    elif criteria == 'count':
        for k in range(n_components):
            count_pos_u = len(u[:, k][u[:, k] >= 0])
            count_pos_v = len(vh[:, k][vh[:, k] >= 0])
            count_neg_u = len(u[:, k][u[:, k] < 0])
            count_neg_v = len(vh[:, k][vh[:, k] < 0])
            count_pos = count_pos_u + count_pos_v
            count_neg = count_neg_u + count_neg_v
            if count_pos < count_neg:
                u[:, k] *= -1
                vh[:, k] *= -1
            u[:, k][u[:, k] < 0] = 1e-7
            vh[:, k][vh[:, k] < 0] = 1e-7
            W[:, k] = (sqrt(s[k]) * u[:, k]).ravel()
            H[k, :] = (sqrt(s[k]) * vh[:, k]).ravel()
    else:
        raise  ValueError("El criterio especificado no esta contemplado")
    W /= W.sum(0)
    H /= H.sum(1)[:, None]
    return W, H


def EM_algorithm(X, frecs, n_iter=20, n_components=10, init=None):
    M, N, K = X.shape[0],X.shape[1], n_components

    if init != None: # Inicialización del tipo (Q, C, S)
        P1 = init[0].copy()
        P1 = P1.T
        P2 = init[2].copy()
        P3 = np.diagonal(init[1]).copy()
        P3 = P3.astype('float32')
        P1 = np.asfortranarray(P1, dtype='float32')
        P2 = np.asfortranarray(P2, dtype='float32')
    else:
        P1 = np.random.rand(K,M)
        P1 /= P1.sum(1)[:, None]
        P2 = np.random.rand(K, N)
        P2 /= P2.sum(1)[:, None]
        P3 = np.random.rand(K)
        P3 /= P3.sum()
    P1[P1 < np.finfo('float32').eps] = np.finfo('float32').eps
    P2[P2 < np.finfo('float32').eps] = np.finfo('float32').eps
    for it in range(n_iter):
        try:
            # E-step
            prod = P3[:, None, None] * \
                    (P1[:, :, None] @ P2[:, None, :])
            Pz_td = prod / prod.sum(0)
#                Pz_td = P3[:, None, None] * \
#                        (P1[:, :, None] @ P2[:, None, :]) / \
#                        (P1.T @ np.diag(P3) @ P2)[None,:,:]
            # M-step
            Pz_td *= frecs.A
            den = Pz_td.sum((1,2))[:, None]
            P1 = Pz_td.sum(2) / den  # (K, M)
            P2 = Pz_td.sum(1) / den
            P3 = (den / den.sum()).ravel()
            # Threshold
            P1[P1 < np.finfo('float32').eps] = np.finfo('float32').eps
            P2[P2 < np.finfo('float32').eps] = np.finfo('float32').eps

        except MemoryError:
            raise MemoryError("El programa se ha detenido por una ocupación excesiva de la memoria.")
            break            
    return P1.T @ np.diag(P3) @ P2, P1, P2, P3

def kkt_W(X, W, H):
    m, k = W.shape
    r_term = np.multiply(1 / (W @ H), W @ H -X) @ H.T
    W_res = (norm_np(np.minimum(W, r_term), 1)) / (m*k)
    return W_res

def kkt_H(X, W, H):
    k, n = H.shape
    r_term = W.T @ np.multiply(1 / (W @ H), W @ H -X)
    H_res = (norm_np(np.minimum(H, r_term), 1)) / (k*n)
    return H_res