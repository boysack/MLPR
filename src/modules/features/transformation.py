import numpy as np
from numpy import ndarray

from modules.utils.operations import mean, col, var, cov

def center_data(D, mu = None) -> ndarray:
    """
    Calculate the mean centered data.

    Parameters:
    D (ndarray): the dataset (samples = columns).
    mu (ndarray): the mean vector.

    Returns:
    ndarray: the centered data.
    """
    if mu is None:
        mu = mean(D)
    else:
        mu = col(mu)
    return D - mu, mu

def standardize_variance(D, v = None):
    if v is None: 
        v = var(D)
    else:
        v = col(v)
    return D / v, v

def withening(D, C = None):
    if np.all(mean(D) != 0):
        D = center_data(D)
        C = cov(D)
    else:
        if C is None:
            C = cov(D)

    S, U = np.linalg.eigh(C)

    epsilon = 1e-5
    T = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))

    return np.dot(T, D), T

def L2_normalization(D):
    n = np.linalg.norm(D, axis=0)
    return D / np.linalg.norm(D, axis=0), n

def quadratic_feature_mapping(D):
    n_features, n_samples = D.shape
    
    # Compute the outer product for each sample (which is now along the second axis)
    quadratic_terms = np.einsum('ik,jk->ijk', D, D)
    
    # Vectorize the quadratic terms (flatten the 4x4 matrices into 16-element vectors)
    quadratic_features = quadratic_terms.reshape(n_features * n_features, n_samples)
    
    # Concatenate the original features with the quadratic features
    return np.vstack([quadratic_features, D])