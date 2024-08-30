import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from modules.data.dataset import load, split_db_2to1
from modules.models.gaussians import MVGModel, TiedGModel
from modules.models.model import Model
from modules.utils.operations import row, col

def conf_matrix(L, predictions, label_dict):
    cf = np.zeros((len(label_dict),len(label_dict)), dtype=int)
    for j in label_dict.values():
        p = predictions[L==j]
        for i in label_dict.values():
            cf[i,j] = (p==i).sum()
    return cf

def print_conf_matrix(label_dict, cfm = None, L = None, predictions = None, integer = True):
    if cfm is None:
        if L is None or predictions is None:
            raise Exception("Insert or a confusion matrix (cfm) parameter, or both ground truth (L) and predictions")
        else:
            cfm = conf_matrix(L, predictions, label_dict)
    if integer is True:
        row_labels = label_dict.values()
        column_labels = label_dict.values()
    else:
        row_labels = label_dict.keys()
        column_labels = label_dict.keys()

    print(pd.DataFrame(cfm, index=row_labels, columns=column_labels))

def expected_bayes_cost(C, posteriors, is_log = False):
    if is_log is True:
        posteriors = np.exp(posteriors)
    
    return np.dot(C, posteriors)

# TODO: implement just the multiclass?
def empirical_bayes_risk_binary(L, predictions, label_dict, prior, C, true_idx = 1, false_idx = 0, normalize = True):
    # in gaussian, check for true_idx
    cfm = conf_matrix(L, predictions, label_dict)

    P_fn = cfm[false_idx, true_idx] / (cfm[false_idx, true_idx]+cfm[true_idx, true_idx])
    P_fp = cfm[true_idx, false_idx] / (cfm[true_idx, false_idx]+cfm[false_idx, false_idx])

    C_fn = C[false_idx, true_idx]
    C_fp = C[true_idx, false_idx]

    if normalize:
        d = min(prior * C_fn, (1-prior) * C_fp)
    else:
        d = 1
    
    return (prior * C_fn * P_fn + (1-prior) * C_fp * P_fp) / d

def empirical_bayes_risk(L, predictions, label_dict, priors, C, normalize = True):
    # priors and costs must be aligned (user responsibility)
    priors = col(np.array(priors))
    cfm = conf_matrix(L, predictions, label_dict)
    norm = cfm.sum(0)
    R = cfm / norm

    if normalize:
        d = np.min(np.dot(C, priors))
    else:
        d = 1

    return (row(priors) * (R * C).sum(0)).sum() / d

def min_DCF_binary(prior, C, llr = None, L = None, P_fn = None, P_fp = None, thresholds = None, true_idx = 1, false_idx = 0, normalize = True, return_threshold = False):
    if P_fn is None or P_fp is None:
        # must compute them
        if llr is None or llr is None:
            raise Exception("One of the couple of parameters P_fn, P_fp or llr, L must be passed to calculate minDCF")
        P_fn, P_fp, thresholds = get_thresholds_binary(llr, L)
    C_fn = C[false_idx, true_idx]
    C_fp = C[true_idx, false_idx]

    DCFs = prior * C_fn * P_fn + (1-prior) * C_fp * P_fp
    min_DCF_arg = np.argmin(DCFs)

    if normalize:
        d = min(prior * C_fn, (1-prior) * C_fp)
    else:
        d = 1

    if return_threshold and thresholds is not None:
        return DCFs[min_DCF_arg] / d, thresholds[min_DCF_arg]
    
    return DCFs[min_DCF_arg] / d

# TODO: understand if the method is specific for gaussian model
def get_thresholds_binary(llr, L):
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

def roc(P_fn, P_fp, plot_title = None):
    plt.plot(P_fp, 1-P_fn)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=':')
    if plot_title:
        plt.title(plot_title)

# object implementation
""" def bayes_error_plot_binary(D, L, label_dict, start, stop, num, model: Model):
    eff_prior_log_odds = np.linspace(start, stop, num)
    priors = 1/(1+np.exp(-eff_prior_log_odds))
    DCFs = []
    minDCFs = []
    scores = mvg.get_scores(D)
    for prior in priors:
        model.set_threshold_from_priors_binary(prior)
        predictions = model.get_prediction(scores)
        cfm = conf_matrix(L, predictions, label_dict)
        DCFs.append(empirical_bayes_risk_binary(L, predictions, label_dict, model.prior[1], model.C))
        P_fn, P_fp, t = get_thresholds_binary(scores, L)
        minDCFs.append(min_DCF_binary(model.prior[1], model.C, P_fn=P_fn, P_fp=P_fp))
    
    plt.plot(eff_prior_log_odds, DCFs, label='DCF', color='r')
    plt.plot(eff_prior_log_odds, minDCFs, label='min DCF', color='b') """

def bayes_error_plot_binary(llr, L, label_dict, start, stop, num, model, plot_title = None):
    eff_prior_log_odds = np.linspace(start, stop, num)
    eff_priors = 1/(1+np.exp(-eff_prior_log_odds))
    DCFs = []
    minDCFs = []
    scores = llr
    for eff_prior in eff_priors:
        model.set_threshold_from_priors_binary(eff_prior)
        predictions = model.get_predictions(scores, bin=True)
        DCFs.append(empirical_bayes_risk_binary(L, predictions, label_dict, eff_prior, model.cost_matrix))
        P_fn, P_fp, t = get_thresholds_binary(scores, L)
        minDCFs.append(min_DCF_binary(eff_prior, model.cost_matrix, P_fn=P_fn, P_fp=P_fp))
    plt.plot(eff_prior_log_odds, DCFs, label='DCF')
    plt.plot(eff_prior_log_odds, minDCFs, label='min DCF')
    plt.xlabel(r'$\log \dfrac{\tilde{\pi}}{1-\tilde{\pi}}$')
    plt.ylabel("DCF")

    plt.xlim(start, stop)

    ylow = min(DCFs + minDCFs)
    ymax = max(DCFs + minDCFs)
    ymin = ymax - (ymax - ylow)*4/3
    plt.ylim(ymin, ymax)

    plt.grid(True, linestyle=':')

    if plot_title:
        plt.title(plot_title)


if __name__ == "__main__":
    D, L, label_dict = load("labs/data/iris.csv")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    # CONFUSION MATRIX FUNCTIONS TEST
    """ 
    mvg = MVGModel(DTR, LTR, label_dict)
    mvg.fit()
    mvg_p, l_scores = mvg.predict(DVAL)

    tg = TiedGModel(DTR, LTR, label_dict)
    tg.fit()
    tg_p, _ = tg.predict(DVAL)

    print(conf_matrix(LVAL, tg_p, label_dict))

    print_conf_matrix(label_dict, L=LVAL, predictions=tg_p) 
    """
    
    # TEST OF THE NEWLY IMPLEMENTED BINARY STUFF
    """ 
    # TEST MULTICLASS CASE 
    mvg = MVGModel(DTR, LTR, label_dict)
    mvg.fit()
    mvg_p, l_scores = mvg.predict(DVAL)

    #print(conf_matrix(LVAL, mvg_p, label_dict))

    # TEST BINARY CASE
    D = D[:, (L==0) | (L==1)]
    L = L[(L==0) | (L==1)]
    label_dict = {k:v for k,v in label_dict.items() if v!=2}
    label_dict = {k:v for k,v in list(label_dict.items())[::-1]}
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    mvg = MVGModel(DTR, LTR, label_dict, l_priors=np.log([0,1]))
    mvg.fit()
    mvg_p, l_scores = mvg.predict(DVAL)

    print("bin", mvg.binary)

    print(conf_matrix(LVAL, mvg_p, label_dict))

    # TEST BINARY CASE WITH MULTICLASS
    mvg_p, l_scores = mvg.predict(DVAL, bin=False)

    print(conf_matrix(LVAL, mvg_p, label_dict))
    exit() """

    # NEUTRAL COST MATRIX TEST
    """ 
    mvg = MVGModel(DTR, LTR, label_dict)
    mvg.fit()
    mvg_p, res = mvg.debug_predict(DVAL)
    C = np.arange(9).reshape(3,-1)
    np.fill_diagonal(C, 0)
    print(C)
    b_costs = bayes_cost(C, res["l_posteriors"], is_log=True)
    print(b_costs.shape) """

    label_dict = {
        "Paradiso": 0,
        "Inferno": 1
    }
    llr = np.load("labs/lab07/Solution/commedia_llr_infpar.npy")
    L = np.load("labs/lab07/Solution/commedia_labels_infpar.npy")
    
    config = [
        (0.5, 1, 1),
        (0.8, 1, 1),
        (0.5, 10, 1),
        (0.8, 1, 10)
    ]
    from time import time

    tot = len(config)
    count = 0
    for c in config:
        print()
        print(c)
        C = np.array([0,c[1],
                    c[2],0]).reshape(2,-1)
        p = np.array([1-c[0], c[0]])
        # check if it works even with numpy arrays

        mvg = MVGModel([0,1], [0,1], label_dict=label_dict, l_priors=np.log(p), cost_matrix=C)
        predictions = mvg.get_predictions(llr, bin=True)

        #print(f"threshold = {-1 * np.log(p[1]*mvg.cost_matrix[0, 1]/(p[0]*mvg.cost_matrix[1, 0]))}")

        print_conf_matrix(label_dict=label_dict, L=L, predictions=predictions)

        print("Binary func")
        print(f"uDCF = {empirical_bayes_risk_binary(L, predictions, label_dict, c[0], C):0.3f}")
        print(f" DCF = {empirical_bayes_risk_binary(L, predictions, label_dict, c[0], C, normalize=True):0.3f}")
        print("Multiclass func")
        print(f"uDCF = {empirical_bayes_risk(L, predictions, label_dict, p, C):0.3f}")
        print(f" DCF = {empirical_bayes_risk(L, predictions, label_dict, p, C, normalize=True):0.3f}")

    P_fn, P_fp, t = get_thresholds_binary(llr, L)
    for c in config:
        print()
        print(c)
        C = np.array([0,c[1],
                    c[2],0]).reshape(2,-1)
        p = [1-c[0], c[0]]

        print(f"minDCF = {min_DCF_binary(c[0], C, P_fn=P_fn, P_fp=P_fp):0.3f}")

    roc(P_fn, P_fp)
    #plt.savefig("./")
    plt.show()

    mvg = MVGModel(D, L, label_dict)
    #print(mvg.get_predictions(llr, bin=True))
    bayes_error_plot_binary(llr, L, label_dict, -3, 3, 21, mvg)
    llr_e1 = np.load("labs/lab07/Solution/commedia_llr_infpar_eps1.npy")
    bayes_error_plot_binary(llr_e1, L, label_dict, -3, 3, 21, mvg)
    plt.show()