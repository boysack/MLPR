from project.packages.utils import *
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import scipy
from time import time

def pca(D: ndarray, C: ndarray = None, m : int = 1) -> tuple[ndarray, ndarray]:
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
    _, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, :m]
    
    return P, np.dot(P.T, D)

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

def lda(D, L, m):
    n_classes = np.unique(L).size
    if m < 1 or m > (n_classes-1):
        raise Exception(f"You're trying to extract {m} directions (1 <= m <= {n_classes-1})")
    # can probably optimize, but is already fast
    Sb = sb(D, L)
    Sw = sw(D, L)
    _, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, :m]

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

if __name__=="__main__":
    D, L, label_dict = load("labs/data/iris.csv")
    #print(mean(D))
    # result slightly different from professor one.
    #print(np.cov(D))

    # PCA - OK
    P_m4, D_m4 = pca(D, m=4)
    SP_m4 = np.load("labs/lab03/Solution/IRIS_PCA_matrix_m4.npy")

    print("testing PCA")
    print((trunc(P_m4, decs=2)==trunc(SP_m4, decs=2)))

    # LDA - OK
    W_m2, D_m2 = lda(D, L, m=2)
    SL_m2 = np.load("labs/lab03/Solution/IRIS_LDA_matrix_m2.npy")

    print("testing LDA")
    print((trunc(W_m2, decs=2)==trunc(SL_m2, decs=2)))

    """ hist_per_feat(D_m4, L, label_dict)
    hist_per_feat(D_m2, L, label_dict)
    plt.show() """

    D = D[:,(L==label_dict["Iris-versicolor"]) | (L==label_dict["Iris-virginica"])]
    L = L[(L==label_dict["Iris-versicolor"]) | (L==label_dict["Iris-virginica"])]
    print(label_dict)
    label_dict.pop("Iris-setosa")

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # LDA - classification
    WTR_m1, DTR_m1 = lda(DTR, LTR, m=1)
    DVAL_m1 = np.dot(WTR_m1.T, DVAL)

    hist_per_feat(DTR_m1, LTR, label_dict, bins=5)
    hist_per_feat(DVAL_m1, LVAL, label_dict, bins=5)
    plt.show()

    threshold = (DTR_m1[0, LTR==1].mean() + DTR_m1[0, LTR==2].mean()) / 2.0
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_m1[0] >= threshold] = 2
    PVAL[DVAL_m1[0] < threshold] = 1

    print(f"accuracy for LDA calssification: {LVAL.shape[0]-(PVAL==LVAL).sum()}")

    # PCA - classification
    PTR_m1, DTR_m1 = pca(DTR, m=1)
    DVAL_m1 = np.dot(PTR_m1.T, DVAL)
    DVAL_m1 *= -1
    DTR_m1 *= -1

    hist_per_feat(DTR_m1, LTR, label_dict, bins=5)
    hist_per_feat(DVAL_m1, LVAL, label_dict, bins=5)
    plt.show()

    threshold = (DTR_m1[0, LTR==1].mean() + DTR_m1[0, LTR==2].mean()) / 2.0
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_m1[0] >= threshold] = 2
    PVAL[DVAL_m1[0] < threshold] = 1

    print(f"accuracy for PCA calssification: {LVAL.shape[0]-(PVAL==LVAL).sum()}")

    # LDA + PCA - classification

    PTR_m2, DTR_pca_m2 = pca(DTR, m=2)
    DVAL_pca_m2 = np.dot(PTR_m2.T, DVAL)

    WTR_m1, DTR_pca_m2_lda_m1 = lda(DTR_pca_m2, L=LTR, m=1)
    DVAL_pca_m2_lda_m1 = np.dot(WTR_m1.T, DVAL_pca_m2)

    hist_per_feat(DTR_pca_m2_lda_m1, LTR, label_dict, bins=5)
    hist_per_feat(DVAL_pca_m2_lda_m1, LVAL, label_dict, bins=5)
    plt.show()

    threshold = (DTR_pca_m2_lda_m1[0, LTR==1].mean() + DTR_pca_m2_lda_m1[0, LTR==2].mean()) / 2.0
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_pca_m2_lda_m1[0] >= threshold] = 2
    PVAL[DVAL_pca_m2_lda_m1[0] < threshold] = 1

    print(f"accuracy for PCA(2) + LDA(1) calssification: {LVAL.shape[0]-(PVAL==LVAL).sum()}")

    PTR_m3, DTR_pca_m3 = pca(DTR, m=2)
    DVAL_pca_m3 = np.dot(PTR_m3.T, DVAL)

    WTR_m1, DTR_pca_m3_lda_m1 = lda(DTR_pca_m3, L=LTR, m=1)
    DVAL_pca_m3_lda_m1 = np.dot(WTR_m1.T, DVAL_pca_m3)

    hist_per_feat(DTR_pca_m3_lda_m1, LTR, label_dict, bins=5)
    hist_per_feat(DVAL_pca_m3_lda_m1, LVAL, label_dict, bins=5)
    plt.show()

    threshold = (DTR_pca_m3_lda_m1[0, LTR==1].mean() + DTR_pca_m3_lda_m1[0, LTR==2].mean()) / 2.0
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_pca_m3_lda_m1[0] >= threshold] = 2
    PVAL[DVAL_pca_m3_lda_m1[0] < threshold] = 1

    print(f"accuracy for PCA(3) + LDA(1) calssification: {LVAL.shape[0]-(PVAL==LVAL).sum()}")