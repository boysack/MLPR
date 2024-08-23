import numpy as np

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