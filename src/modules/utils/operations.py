import numpy as np
from numpy import ndarray

def col(x: ndarray) -> ndarray:
    """
    Return the column vector of the given one.

    Parameters:
    x (ndarray): the vector to transform.

    Returns:
    ndarray: the column vector.
    """
    if len(x.shape) == 2 and x.shape[1] == 1:
        return x
    else:
        return x.reshape(-1, 1)
    
def row(x: ndarray) -> ndarray:
    """
    Return the row vector of the given one.

    Parameters:
    x (ndarray): the vector to transform.

    Returns:
    ndarray: the row vector.
    """
    if len(x.shape) == 2 and x.shape[0] == 1:
        return x
    else:
        return x.reshape(1, -1)

# TODO: CHECK IF NECESSARY TO CHECK IF THE DATASET IS NONE
# actually, the parameter is mandatory, so is probably useless to check if it's none since it will be checked atuomatically
def mean(D: ndarray) -> ndarray:
    """
    Calculate mean vector of the dataset.

    Parameters:
    D (ndarray): the dataset (sample = column vector).

    Returns:
    ndarray: the mean column vector.
    """
    return col(D.sum(axis=1)/D.shape[1])

def var(D: ndarray) -> ndarray:
    """
    Calculate variance vector of the dataset.

    Parameters:
    D (ndarray): the dataset (sample = column vector).

    Returns:
    ndarray: the variance column vector.
    """
    return col(((D - (col(D.sum(axis=1)/D.shape[1])))**2).sum(axis=1)/D.shape[1])

def cov(D: ndarray, mu: ndarray = None ) -> ndarray:
    """
    Calculate covariance matrix of the dataset.

    Parameters:
    D (ndarray): the dataset (sample = column vector).
    mu (ndarray): the mean vector.

    Returns:
    ndarray: the covariance matrix.
    """

    if mu is None:
        mu = mean(D)
    else:
        mu = col(mu)
    Dc = D - mu

    return np.dot(Dc, Dc.T)/D.shape[1]

def center_data(D: ndarray, mu: ndarray = None) -> ndarray:
    """
    Calculate the mean centered data.

    Parameters:
    D (ndarray): the dataset (sample = column vector).
    mu (ndarray): the mean vector.

    Returns:
    ndarray: the centered data.
    """
    if mu is None:
        mu = mean(D)
    else:
        mu = col(mu)
    return D - mu


def trunc(values, decs=0):
    """
    Used to truncate values precision to perform comparison of numpy arrays calculated using different techniques, and so having slightly different results
    """
    return np.trunc(values*10**decs)/(10**decs)
