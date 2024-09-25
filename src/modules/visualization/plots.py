import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from math import ceil, sqrt
from screeninfo import get_monitors

from modules.utils.metrics import empirical_bayes_risk_binary, min_DCF_binary
from modules.utils.operations import get_thresholds_from_llr

# TODO: make the function receive a boolean parameter show, and a filepath where to save eventually the plot

def get_screen_size():
    primary_monitor = get_monitors()[0]
    screen_width = primary_monitor.width
    screen_height = primary_monitor.height
    return screen_width, screen_height

def hist_per_feat(D, L, label_dict, bins=None, subplots=True):
    # TODO: fix bug: if subplots=False it show an empty plot in the end

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
    # TODO: fix bug: if subplots=False it show an empty plot in the end

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

def bayes_error_plot_binary(L, llr, true_idx = 1, start = -3, stop = 3, num = 21, plot_title = None, plot_min_DCF = True, act_DCF_prefix = "", min_DCF_prefix = ""):
    eff_prior_log_odds = np.linspace(start, stop, num)
    eff_priors = 1/(1+np.exp(-eff_prior_log_odds))

    DCFs = []
    minDCFs = []

    P_fn, P_fp, _ = get_thresholds_from_llr(llr, L)
    color = None
    if plot_min_DCF:
        for eff_prior in eff_priors:
            DCFs.append(empirical_bayes_risk_binary(prior=eff_prior, L=L, llr=llr, true_idx=true_idx))
            minDCFs.append(min_DCF_binary(prior=eff_prior, P_fn=P_fn, P_fp=P_fp, true_idx=true_idx))

        line = plt.plot(eff_prior_log_odds, minDCFs, label=f'{min_DCF_prefix} min DCF'.strip(), linestyle="dashed")
        color = line[0]._color
    else:
        for eff_prior in eff_priors:
            DCFs.append(empirical_bayes_risk_binary(prior=eff_prior, L=L, llr=llr, true_idx=true_idx))

    if color is not None:
        plt.plot(eff_prior_log_odds, DCFs, label=f'{act_DCF_prefix} DCF'.strip(), color=color)
    else:
        plt.plot(eff_prior_log_odds, DCFs, label=f'{act_DCF_prefix} DCF'.strip())

    plt.legend()

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

# TODO: check if it's convenient to do as in min_DCF_binary, i.e. to give the possibility to insert directly the scores and the labels and calculate the
# P_fn and P_fp in place
def roc(P_fn, P_fp, plot_title = None):
    plt.plot(P_fp, 1-P_fn)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=':')
    if plot_title:
        plt.title(plot_title)