import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.models.gaussians import MVGMModel, NaiveGMModel, TiedGMModel, logpdf_GMM
from modules.utils.operations import col, trunc

import numpy as np

import json

def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]

def plot_1D_density():
    #D = np.load("./labs/lab10/Data/GMM_data_1D.npy")
    D = np.load("./labs/lab10/Data/GMM_data_4D.npy")
    #gmm = load_gmm("./labs/lab10/Data/GMM_1D_3G_init.json")

    gm = MVGMModel(D, [], [], n=4)
    avg_l_likelihood = gm.fit()
    DPLOT = np.linspace(np.min(D), np.max(D), 1000).reshape(1,-1)
    import matplotlib.pyplot as plt
    gmm = [(w, mu[0], C[0,0].reshape(1,1)) for w, mu, C in gm.gmm]
    p = np.exp(logpdf_GMM(DPLOT, gmm)[0])
    plt.plot(DPLOT.ravel(), p)

    gm = NaiveGMModel(D, [], [], n=4)
    avg_l_likelihood = gm.fit()
    DPLOT = np.linspace(np.min(D), np.max(D), 1000).reshape(1,-1)
    import matplotlib.pyplot as plt
    gmm = [(w, mu[0], C[0,0].reshape(1,1)) for w, mu, C in gm.gmm]
    p = np.exp(logpdf_GMM(DPLOT, gmm)[0])
    plt.plot(DPLOT.ravel(), p)

    gm = TiedGMModel(D, [], [], n=4)
    avg_l_likelihood = gm.fit()
    DPLOT = np.linspace(np.min(D), np.max(D), 1000).reshape(1,-1)
    import matplotlib.pyplot as plt
    gmm = [(w, mu[0], C[0,0].reshape(1,1)) for w, mu, C in gm.gmm]
    p = np.exp(logpdf_GMM(DPLOT, gmm)[0])
    plt.plot(DPLOT.ravel(), p)

    plt.show()

if __name__=="__main__":
    """ D = np.load("./labs/lab10/Data/GMM_data_4D.npy")
    gmm = load_gmm("./labs/lab10/Data/GMM_4D_3G_init.json")

    logpdfs, l_joint_g,  l_posterior_g = l_marginal_g, l_joint_g, l_posterior_g = logpdf_GMM(D, gmm)
    logpdfs_sol = np.load("./labs/lab10/Data/GMM_4D_3G_init_ll.npy")
    #print(np.all(logpdfs == logpdfs_sol))

    #print(l_posterior_g.shape)
    gm = GaussianMixtureModel(D, [], [], gmm=gmm)
    avg_l_likelihood = gm.fit()
    #print(avg_l_likelihood)

    # simili
    gmm_sol = load_gmm("./labs/lab10/Data/GMM_4D_3G_EM.json")
    for g in range(3):
        sol = gmm_sol[g]
        my = gm.gmm[g]
        for p in range(3):
            #print(f"{my[p]} == {sol[p]}?")
            #print(np.all(my[p] == sol[p]))
            pass """

    #plot_1D_density()

    # CLASSIFICATION
    mvgmm = MVGMModel()