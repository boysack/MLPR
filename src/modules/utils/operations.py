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

def p_corr(C: ndarray):
    return C / (col(C.diagonal()**0.5) * row(C.diagonal()**0.5))

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

# TODO: understand if the method is specific for gaussian model
def get_thresholds_from_llr(llr, L):
    llr_sorter = np.argsort(llr)
    llr = llr[llr_sorter]
    L = L[llr_sorter]

    P_fp = []
    P_fn = []
    
    p = (L==1).sum()
    n = (L==0).sum()
    fp = n
    fn = 0
    
    P_fp.append(fp / n)
    P_fn.append(fn / p)

    for idx in range(len(llr)):
        # llr[idx] => new threshold (>true, <=false)
        # (first it. -inf<x<=llr[0])
        # think at it as adding 0 predictions, approaching to a threshold that is gt llr[idx]
        if L[idx] == 1:
            fn += 1
        if L[idx] == 0:
            fp -= 1
        P_fp.append(fp / n)
        P_fn.append(fn / p)
    llr = np.concatenate([-np.array([np.inf]), llr])

    P_fp_out = []
    P_fn_out = []
    thresholds_out = []
    for idx in range(len(llr)):
        if idx == len(llr) - 1 or llr[idx+1] != llr[idx]:
            P_fp_out.append(P_fp[idx])
            P_fn_out.append(P_fn[idx])
            thresholds_out.append(llr[idx])

    return np.array(P_fn_out), np.array(P_fp_out), np.array(thresholds_out)