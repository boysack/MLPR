import numpy as np
from numpy import ndarray
import os
import matplotlib.pyplot as plt
from math import ceil, sqrt
from screeninfo import get_monitors
import scipy

def load(filename: str, delimiter: str = ",") -> tuple[ndarray, ndarray, dict]:
    # TODO: check if it's convenient to return even an inversed version of label_dictionary
    """
    Load the dataset from a file.

    Parameters:
    filename (str): filename from which extract the data.
    delimiter (str): delimiter used in the file to separate the data values.

    Returns:
    ndarray: the dataset (sample = column vector).
    ndarray: the labels.
    dict: the labels dictionary.
    """
    if not os.path.isfile(filename):
        raise Exception("Must insert a valid filename")
    # used to create label dictionary
    label_index = 0
    label_dict = {}
    with open(filename, "r") as f:
        first = True
        while True:
            line = f.readline()
            # if EOF, break
            if not line:
                break
            line = line.split(delimiter)

            # if first iteration, then instantiate the numpy arrays
            if first:
                D = np.empty((len(line)-1, 0), dtype=float)
                L = np.empty((0), dtype=int)
                first = False

            # create the label entry if not exists
            label = line[-1].strip()
            if label not in label_dict.keys():
                label_dict[label] = label_index
                label_index += 1

            # append the sample in the arrays
            D = np.hstack((D, col(np.array([float(i) for i in line[:-1]]))))
            L = np.append(L, label_dict[label])

    # if binary problem with class 0 and 1, transform in True and False their labels name
    if all([True if value=="0" or value=="1" else False for value in label_dict.keys()]):
        label_dict = {}
        label_dict["False"] = 0
        label_dict["True"] = 1

    return D, L, label_dict

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
    
def row(x):
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

def get_screen_size():
    primary_monitor = get_monitors()[0]
    screen_width = primary_monitor.width
    screen_height = primary_monitor.height
    return screen_width, screen_height

def hist_per_feat(D, L, label_dict, bins=None, subplots=True):
    #TODO: fix bug: if subplots=False it show an empty plot in the end

    feats = D.shape[0]
    plots_per_row = ceil(sqrt(feats))

    screen_width, screen_height = get_screen_size()
    dpi = 100
    plt.figure(layout="tight", figsize=(screen_width/dpi,(screen_height/dpi)-0.7), dpi=dpi)

    for i in range(D.shape[0]):
        if subplots:
            plt.subplot(plots_per_row, plots_per_row, i+1)
        else:
            plt.figure(layout="tight", figsize=(screen_width/dpi,(screen_height/dpi)-0.7), dpi=dpi)
        for key, value in label_dict.items():
            filtered_D = D[i, L==value]
            if bins is not None:
                b = bins
            else:
                b = fd_optimal_bins((filtered_D).flatten())
            plt.hist(filtered_D, bins=b, alpha=0.5, label=key, density=True)
        plt.legend(loc='upper right')

        # TODO: add features name label if provided
        #plt.xlabel("")

def fd_optimal_bins(D: ndarray):
    if D.ndim != 1:
        raise Exception("Wrong dimension number!")
    n = D.size
    p25, p75 = np.percentile(D, [25, 75])
    width = 2. * (p75 - p25)/n**(1./3)
    nbins = ceil((D.max() - D.min()) / width)
    nbins = max(1, nbins)
    return nbins

def scatter_hist_per_feat(D, L, label_dict, feature_dict=None, bins=None, subplots=True):
    #TODO: fix bug: if subplots=False it show an empty plot in the end

    plots_per_row = D.shape[0]
    
    screen_width, screen_height = get_screen_size()
    dpi = 50
    plt.figure(layout="tight", figsize=(screen_width/dpi,(screen_height/dpi)-0.7), dpi=dpi)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if subplots:
                pos = plots_per_row*i+j+1
                plt.subplot(plots_per_row, plots_per_row, pos)
            else:
                plt.figure(layout="tight", figsize=(screen_width/dpi,(screen_height/dpi)-0.7), dpi=dpi)
            for key, value in label_dict.items():
                filtered_D_i = D[i, L==value]
                if i == j:
                    if bins is not None:
                        b = bins
                    else:
                        b = fd_optimal_bins((filtered_D_i).flatten())
                    plt.hist(filtered_D_i, bins=b, alpha=0.5, label=key, density=True)
                else:
                    filtered_D_j = D[j, L==value]
                    plt.scatter(filtered_D_i, filtered_D_j, label=key, s=5, alpha=.5)
            plt.legend(loc='upper right')

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
    if D is None:
        raise Exception("Must insert dataset as argument")
    return col(D.sum(axis=1)/D.shape[1])

def var(D: ndarray) -> ndarray:
    """
    Calculate variance vector of the dataset.

    Parameters:
    D (ndarray): the dataset (sample = column vector).

    Returns:
    ndarray: the variance column vector.
    """
    if D is None:
        raise Exception("Must insert dataset as argument")
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
    if D is None:
        raise Exception("Must insert dataset as argument")
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
    if D is None:
        raise Exception("Must insert dataset as argument")
    if mu is None:
        mu = mean(D)
    else:
        mu = col(mu)
    return D - mu

def pca(D: ndarray, C: ndarray = None, m: int = 1, change_sign: bool = False) -> tuple[ndarray, ndarray]:
    """
    Calculate the specified Principal Components and project the data along the found directions.

    Parameters:
    D (ndarray): the dataset (sample = column vector).
    C (ndarray): the covariance matrix. If not specified, will be calculated internally.
    m (int): the number of components you want to find. It must be a number included in the interval [1, n_features]). Default to 1.

    Returns:
    ndarray: the found Principal Components.
    ndarray: the projected data along the Principal Components.
    """
    if D is None:
        raise Exception("Must insert dataset as argument")
    if m < 1 or m > D.shape[0]:
        raise Exception(f"You're trying to extract {m} principal component (1 <= m <= {D.shape[0]})")
    if C is None:
        #C = np.cov(D)
        C = cov(D)
    eigvals, U = np.linalg.eigh(C)

    P = U[:, ::-1][:, :m]
    V = eigvals[::-1][:m]

    if change_sign:
        P *= -1
    
    return P, V, np.dot(P.T, D)

def sb(D: ndarray, L: ndarray) -> ndarray:
    """
    Calculate between class covariance

    Parameters:
    D (ndarray): the dataset (sample = column vector).
    D (ndarray): the labels.

    Returns:
    ndarray: the between class covariance matrix
    """
    if D is None:
        raise Exception("Must insert dataset as argument")
    if L is None:
        raise Exception("Must insert labels as argument")
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
    if D is None:
        raise Exception("Must insert dataset as argument")
    if L is None:
        raise Exception("Must insert labels as argument")
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
    _, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, :m]

    if change_sign:
        W *= -1

    return W, np.dot(W.T, D)

def cov(D, mu=None):
    if mu is None:
        mu = mean(D)
    else:
        mu = col(mu)
    Dc = D - mu

    return np.dot(Dc, Dc.T)/D.shape[1]

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)

def trunc(values, decs=0):
    """
    Used to truncate values precision to perform comparison of numpy arrays calculated using different techniques, and so having slightly different results
    """
    return np.trunc(values*10**decs)/(10**decs)

# TODO: change x in D (since it works for a whole dataset)
# TODO: find a way to apply the same calculation if there's a series of parameters (means and covariance matrices). In this way
    # externally can be applied the logpdf_GAU_ND without any for loop for each parameter member
def logpdf_GAU_ND(x, mu = None, C = None):
    if np.isscalar(C):
        logdet = np.log(C)
        inv = 1/C
    else:
        logdet = np.linalg.slogdet(C)[1]
        inv = np.linalg.inv(C)
    M = x.shape[0]
    x_c = x - mu
    # fix the fact that C could be even a scalar
    return -(M/2) * np.log(2*np.pi) - .5 * logdet - .5 * (np.dot(x_c.T, inv).T * x_c).sum(0)

def loglikelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()