import numpy as np
import nimfa
from nonnegfac.nmf import *
from sklearn.decomposition import NMF as NMF_sklearn
import plsatwitter.utils.funciones as mat
from plsatwitter.utils.visuals import plot_residual

def calc_nmf(X, n_iter = 20, n_components = 10, method='sklearn', wi = None, hi = None, metrica='kullback-leibler'):
    """
    Realiza una descomposición NMF utilizando diferentes métodos
    Args:
        n_iter: int
            Nº de iteraciones.
        n_components: int
            Rango de la reducción, k.
        method: 'sklearn' | 'nimfa_MU' | 'ANLS_BLOCKPIVOT' | 'ANLS_AS_NUMPY' | 'ANLS_AS_GROUP' | 'HALS'
        wi: array-like
            Valor inicial de la matriz W.
        hi:
            Valor inicial de la matriz H.
    Returns:
        W: array-like
            Matriz de coeficientes
        H: array-like
            Matriz de bases

    """
    if wi is not None and hi is not None:
        init_kim = (wi, hi.T)
    else:
        init_kim = None
    if method == 'sklearn':
        if wi is not None and hi is not None:
            nmf = NMF_sklearn(n_components=n_components, max_iter=n_iter, random_state=1, beta_loss=metrica, solver="mu",
                  alpha=0, l1_ratio=0, init='custom')
            W = nmf.fit_transform(X, W=wi, H=hi)
            H = nmf.components_
        else:
            nmf = NMF_sklearn(n_components = n_components, max_iter=n_iter, random_state=1, beta_loss=metrica, solver="mu",
                      alpha=0, l1_ratio=0)
            W = nmf.fit_transform(X)
            H = nmf.components_
    if method == 'nimfa_MU':
        nmf = nimfa.Nmf(X, W=wi, H=hi, rank=n_components, max_iter=n_iter, update='divergence',
                        objective='div')
        nmf_fit = nmf()
        W = nmf_fit.basis()
        W = np.asarray(W)
        H = nmf_fit.coef()
        H = np.asarray(H)
    if method == 'Lin_PG':
        nmf = nimfa.Lsnmf(X, W=wi,H=hi,H1=hi, rank=n_components, max_iter=n_iter, sub_iter=10,
                inner_sub_iter=10, beta=0.1)
        nmf_fit = nmf()
        W = nmf_fit.basis()
        W = np.asarray(W)
        H = nmf_fit.coef()
        H = np.asarray(H)
    if method == 'simple_MU':
        W, H, _ = NMF_MU().run(X, n_components, init = init_kim, max_iter=n_iter, verbose=-1)
        H = H.T
    if method == 'ANLS_BLOCKPIVOT':
        W, H, _ = NMF_ANLS_BLOCKPIVOT().run(X, n_components, init = init_kim, max_iter = n_iter, verbose=-1)
        H = H.T
    if method == 'ANLS_AS_NUMPY':
        W, H, _ = NMF_ANLS_AS_NUMPY().run(X, n_components, max_iter=n_iter, verbose=-1, init=init_kim)
        H = H.T
    if method == 'ANLS_AS_GROUP':
        W, H, _ = NMF_ANLS_AS_GROUP().run(X, n_components, max_iter=n_iter, verbose=-1, init=init_kim)
        H = H.T
    if method == 'HALS':
        W, H, _ = NMF_HALS().run(X, n_components, max_iter=n_iter, verbose=-1, init=init_kim)
        H = H.T
    return W, H    
def EM_algorithm(X, n_iter=20, n_topics=10, init = None, X_alt = None, frecs_alt = None):
    
    X = X
    frecs = frecs
    if X_alt is not None:
        X = X_alt
    if frecs_alt is not None:
        frecs = frecs_alt
      
    M, N, K = X.shape[0], X.shape[1], n_topics
    
    kkt_residual = {
            'W': [],
            'H': []
        }

    if init != None:
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
            if frecs_alt is not None:
                Pz_td *= frecs
            else:
                Pz_td *= frecs.A
            den = Pz_td.sum((1,2))[:, None]
            P1 = Pz_td.sum(2) / den  # (K, M)
            P2 = Pz_td.sum(1) / den
            P3 = (den / den.sum()).ravel()
            # Threshold
            P1[P1 < np.finfo('float32').eps] = np.finfo('float32').eps
            P2[P2 < np.finfo('float32').eps] = np.finfo('float32').eps
            
            
            kkt_residual['W'].append(mat.kkt_W(X, P1.T @ np.diag(P3), P2))
            kkt_residual['H'].append(mat.kkt_H(X, P1.T @ np.diag(P3), P2))

        except MemoryError:
            raise MemoryError("El programa se ha detenido por una ocupación excesiva de la memoria.")
            break  
    plot_residual(range(1, n_iter+1), kkt_residual)
    return P1.T @ np.diag(P3) @ P2, P1, P2, P3
def EM_algorithm_query(X, n_qw, n_iter=20, n_topics=10, init = None, verbose=False, calc_div=False, preStop=False):
      # Inicialización
    if init != None:
        P1 = init[0].copy()
        P1 = P1.T
        P2 = init[2].copy()
        P3 = init[1].copy()
        P3 = P3.astype('float64')
        P1 = np.asfortranarray(P1, dtype='float64')
        P2 = np.asfortranarray(P2, dtype='float64')
    P1[P1 == 0] = 1e-15
    P2[P2 == 0] = 1e-15
    for iter in range(n_iter):
        Pz_td = P3[:, None, None] *(P1.T[:, :, None] @ P2[:, None, :]) / \
                (P1 @ np.diag(P3) @ P2)[None,:,:]
        # M-step
        Pz_td *= n_qw[None :, :]
        P2 = Pz_td.sum(1)
        P2 /= P2.sum()
        P2[P2 < 1e-15] = 1e-15
        print("aa")
    return P2 # P(q|z)