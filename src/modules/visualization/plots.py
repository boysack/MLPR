import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from math import ceil, sqrt
from screeninfo import get_monitors

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
