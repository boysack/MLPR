import numpy as np
from numpy import ndarray

from modules.utils.operations import mean, col, var, cov

# all this function returns the found parameter to perform the transformation (used in kfold, implement all in this way)
# always return a tuple (even if a single parameter is returned)

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

def withening(D, T = None):
    if T is None:
        D, _ = center_data(D)
        C = cov(D)

        S, U = np.linalg.eigh(C)

        epsilon = 1e-5
        T = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))

    return np.dot(T, D), T

def L2_normalization(D, n = None):
    if n is None:
        n = np.linalg.norm(D, axis=0)
    return D / np.linalg.norm(D, axis=0), n

def z_normalization(D, m = None, v = None):
    if (m is None) ^ (v is None):
        raise Exception("Must pass both m and v or none of them.")
    elif (m is None) and (v is None):
        m = mean(D)
        v = var(D)

    return (D-m)/v, m, v

def quadratic_feature_mapping(D):
    n_features, n_samples = D.shape
    
    # Compute the outer product for each sample (which is now along the second axis)
    quadratic_terms = np.einsum('ik,jk->ijk', D, D)
    
    # Vectorize the quadratic terms (flatten the 4x4 matrices into 16-element vectors)
    quadratic_features = quadratic_terms.reshape(n_features * n_features, n_samples)
    
    # Concatenate the original features with the quadratic features
    return (np.vstack([quadratic_features, D]), )

def no_op(D):
    return (D, )