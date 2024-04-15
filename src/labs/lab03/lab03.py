from project.packages.utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy

def pca(D, C=None, m=1):
    if D is None:
        raise Exception("Must insert dataset as argument")
    if m < 1 or m > D.shape[0]:
        raise Exception(f"You're trying to extract {m} principal component (1 <= m <= {D.shape[0]})")
    if C is None:
        C = np.cov(D)
    _, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, :m]
    
    return P, np.dot(P.T, D)

def sb(D, L):
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

def sw(D, L):
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

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

if __name__=="__main__":
    D, L, label_dict = load("labs/data/iris.csv")
    #print(mean(D))
    # result slightly different from professor one.
    #print(np.cov(D))

    # PCA - OK
    P_m4, D_m4 = pca(D, m=4)
    SP_m4 = np.load("labs/lab03/Solution/IRIS_PCA_matrix_m4.npy")
    print((trunc(P_m4, decs=2)==trunc(SP_m4, decs=2)))

    # LDA -
    W_m2, L_m2 = lda(D, L, m=2)
    SL_m2 = np.load("labs/lab03/Solution/IRIS_LDA_matrix_m2.npy")
    print((trunc(W_m2, decs=2)==trunc(SL_m2, decs=2)))

    #hist_per_feat(D_m2, L, label_dict, subplots=False)
    #plt.show()
