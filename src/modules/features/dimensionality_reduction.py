import numpy as np
from numpy import ndarray
from modules.utils.operations import *
from scipy.linalg import eigh as scipy_eigh

def pca(D: ndarray, C: ndarray = None, m: int = 1, change_sign: bool = False) -> tuple[ndarray, ndarray]:
    """
    Calculate the specified Principal Components and project the data along the found directions.

    Parameters:
    D (ndarray): the dataset (sample = column vector).
    C (ndarray): the covariance matrix. If not specified, will be calculated internally.
    m (int): the number of components you want to find. It must be a number included in the interval [1, n_features]). Default to 1.

    Returns:
    ndarray: the found Principal Components.
    ndarray: the found eigenvalues (i.e. variances of projected points).
    ndarray: the projected data along the Principal Components.
    """
    if m < 1 or m > D.shape[0]:
        raise Exception(f"You're trying to extract {m} principal component (1 <= m <= {D.shape[0]})")
    if C is None:
        C = cov(D)
    eigvals, U = np.linalg.eigh(C)

    P = U[:, ::-1][:, :m]
    V = eigvals[::-1]

    if change_sign:
        P *= -1
    
    return P, V, np.dot(P.T, D)

def pca_pipe(D, m = 1, P = None):
    if P is None:
        P, _, D = pca(D, m=m)
    return D, m, P

def sb(D: ndarray, L: ndarray) -> ndarray:
    """
    Calculate between class covariance

    Parameters:
    D (ndarray): the dataset (sample = column vector).
    D (ndarray): the labels.

    Returns:
    ndarray: the between class covariance matrix
    """
    classes = np.unique(L)
    sum = 0
    mu = mean(D)
    for c in classes:
        D_c = D[:, L==c]
        mu_c = mean(D_c)
        len_c = D_c.shape[1]
        d_c = mu_c - mu
        sum += len_c * (np.dot(d_c, d_c.T))
    
    return sum/D.shape[1]

def sw(D: ndarray, L: ndarray) -> ndarray:
    """
    Calculate within class covariance

    Parameters:
    D (ndarray): the dataset (sample = column vector).
    D (ndarray): the labels.

    Returns:
    ndarray: the within class covariance matrix
    """
    classes = np.unique(L)
    sum = 0
    for c in classes:
        D_c = D[:, L==c]
        mu_c = mean(D_c)
        D_c_centered = D_c - mu_c
        sum += np.dot(D_c_centered, D_c_centered.T)
        # sum += D_c.shape[1] * np.cov(D_c)
    
    return sum/D.shape[1]

def lda(D: ndarray, L: ndarray, m: int = 1, change_sign: bool = False) -> tuple[ndarray, ndarray]:
    n_classes = np.unique(L).size
    if m < 1 or m > (n_classes-1):
        raise Exception(f"You're trying to extract {m} directions (1 <= m <= {n_classes-1})")
    # can probably optimize, but is already fast
    Sb = sb(D, L)
    Sw = sw(D, L)
    _, U = scipy_eigh(Sb, Sw)
    W = U[:, ::-1][:, :m]

    if change_sign:
        W *= -1

    return W, np.dot(W.T, D)

def lda_pipe(D, L, m = 1, W = None):
    if W is None:
        W, D = lda(D, L, m=m)
    return D, L, m, W