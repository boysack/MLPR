import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.models.gaussians import MVGMModel, NaiveGMModel, TiedGMModel, logpdf_GMM
from modules.utils.operations import col, trunc
from modules.data.dataset import load, split_db_2to1
from modules.utils.metrics import error_rate, min_DCF_binary, empirical_bayes_risk_binary

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

    gm = MVGMModel(D, n=4)
    avg_l_likelihood = gm.fit()
    DPLOT = np.linspace(np.min(D), np.max(D), 1000).reshape(1,-1)
    import matplotlib.pyplot as plt
    gmm = [(w, mu[0], C[0,0].reshape(1,1)) for w, mu, C in gm.gmm[0]]
    p = np.exp(logpdf_GMM(DPLOT, gmm)[0])
    plt.plot(DPLOT.ravel(), p)

    gm = NaiveGMModel(D, n=4)
    avg_l_likelihood = gm.fit()
    DPLOT = np.linspace(np.min(D), np.max(D), 1000).reshape(1,-1)
    import matplotlib.pyplot as plt
    gmm = [(w, mu[0], C[0,0].reshape(1,1)) for w, mu, C in gm.gmm[0]]
    p = np.exp(logpdf_GMM(DPLOT, gmm)[0])
    plt.plot(DPLOT.ravel(), p)

    gm = TiedGMModel(D, n=4)
    avg_l_likelihood = gm.fit()
    DPLOT = np.linspace(np.min(D), np.max(D), 1000).reshape(1,-1)
    import matplotlib.pyplot as plt
    gmm = [(w, mu[0], C[0,0].reshape(1,1)) for w, mu, C in gm.gmm[0]]
    p = np.exp(logpdf_GMM(DPLOT, gmm)[0])
    plt.plot(DPLOT.ravel(), p)

    plt.show()

if __name__=="__main__":
    # INTIAL TESTS
    """ 
    D = np.load("./labs/lab10/Data/GMM_data_4D.npy")
    gmm = load_gmm("./labs/lab10/Data/GMM_4D_3G_init.json")

    logpdfs, l_joint_g,  l_posterior_g = l_marginal_g, l_joint_g, l_posterior_g = logpdf_GMM(D, gmm)
    logpdfs_sol = np.load("./labs/lab10/Data/GMM_4D_3G_init_ll.npy")
    #print(np.all(logpdfs == logpdfs_sol))

    #print(l_posterior_g.shape)
    gm = MVGMModel(D, gmm={0: gmm})
    avg_l_likelihood = gm.fit()
    #print(avg_l_likelihood)

    # simili
    gmm_sol = load_gmm("./labs/lab10/Data/GMM_4D_3G_EM.json")

    for g in range(3):
        sol = gmm_sol[g]
        my = gm.gmm[0][g]
        for p in range(3):
            #print(f"{my[p]} == {sol[p]}?")
            #print(np.all(my[p] == sol[p]))
            pass
     """
    # PLOT 1D DENSITY
    #plot_1D_density()
    
    # CLASSIFICATION
    D, L, label_dict = load("labs/data/iris.csv")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    results = {
        "Multivariate": [],
        "Naive": [],
        "Tied covariance": []
    }
    # uniform priors
    l_priors = np.log([1/len(label_dict) for _ in range(len(label_dict))])
    #l_priors = np.log([.1,.1,.8])

    # num components OK
    # the difference from the likelihood is calculated OK (I mean, it stops, but there could be some errors)
    for i in range(5):
        n = 2**i
        mvgmm = MVGMModel(DTR, LTR, label_dict=label_dict, l_priors=l_priors, n=n, d=10**-6)
        mvgmm.fit()
        predictions, l_scores = mvgmm.predict(DVAL)
        results["Multivariate"].append(error_rate(LVAL, predictions))
        
        """ gmm = mvgmm.gmm
        print(2**i)
        for k in gmm.keys():
            print(f"Class {k}")
            for tup in gmm[k]:
                w = tup[0]
                mu = tup[1]
                C = tup[2]
                print(f"w\n{w}")
                print(f"mu\n{mu}")
                print(f"C\n{C}")
                print() """
        
        mvgmm = TiedGMModel(DTR, LTR, label_dict=label_dict, l_priors=l_priors, n=n, d=10**-6)
        mvgmm.fit()
        predictions, l_scores = mvgmm.predict(DVAL)
        results["Tied covariance"].append(error_rate(LVAL, predictions))

        mvgmm = NaiveGMModel(DTR, LTR, label_dict=label_dict, l_priors=l_priors, n=n, d=10**-6)
        mvgmm.fit()
        predictions, l_scores = mvgmm.predict(DVAL)
        results["Naive"].append(error_rate(LVAL, predictions))

    for k, v in results.items():
        print(f"{k:>20} {[f"{er*100}%" for er in v]}")

    # BINARY CLASSIFICATION

    D = np.load("./labs/lab10/Data/ext_data_binary.npy")
    L = np.load("./labs/lab10/Data/ext_data_binary_labels.npy")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    label_dict = {}
    for l in np.unique(L):
        label_dict[f"{l}"] = l

    results = {
        "Multivariate": [],
        "Naive": [],
        "Tied covariance": []
    }

    for i in range(5):
        n = 2**i
        mvgmm = MVGMModel(DTR, LTR, label_dict=label_dict, n=n, d=10**-6)
        mvgmm.fit()

        l_scores = mvgmm.get_scores(DVAL)
        llr = l_scores[1]-l_scores[0]
        min_DCF = min_DCF_binary(prior=0.5, L=LVAL, llr=llr)
        act_DCF = empirical_bayes_risk_binary(prior=0.5, L=LVAL, llr=llr)
        results["Multivariate"].append((min_DCF, act_DCF))
        
        mvgmm = TiedGMModel(DTR, LTR, label_dict=label_dict, n=n, d=10**-6)
        mvgmm.fit()
        
        l_scores = mvgmm.get_scores(DVAL)
        llr = l_scores[1]-l_scores[0]
        min_DCF = min_DCF_binary(prior=0.5, L=LVAL, llr=llr)
        act_DCF = empirical_bayes_risk_binary(prior=0.5, L=LVAL, llr=llr)
        results["Tied covariance"].append((min_DCF, act_DCF))

        mvgmm = NaiveGMModel(DTR, LTR, label_dict=label_dict, n=n, d=10**-6)
        mvgmm.fit()

        l_scores = mvgmm.get_scores(DVAL)
        llr = l_scores[1]-l_scores[0]
        min_DCF = min_DCF_binary(prior=0.5, L=LVAL, llr=llr)
        act_DCF = empirical_bayes_risk_binary(prior=0.5, L=LVAL, llr=llr)
        results["Naive"].append((min_DCF, act_DCF))

    for k, v in results.items():
        print(f"{k:>20} {[f'{min_DCF:.4f} / {act_DCF:.4f}' for min_DCF, act_DCF in v]}")