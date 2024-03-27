import numpy as np
import os
import matplotlib.pyplot as plt
from math import ceil, sqrt

def load(filename: str, delimiter: str = ","):
    if not os.path.isfile(filename):
        raise Exception("File doesn't exists!")
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

def col(x):
    if len(x.shape) == 2 and x.shape[1] == 1:
        return x
    else:
        return x.reshape(-1, 1)
    
def row(x):
    if len(x.shape) == 2 and x.shape[0] == 1:
        return x
    else:
        return x.reshape(1, -1)

def hist_per_feat(D, L, label_dict, bins=None, subplots=True):
    feats = D.shape[0]
    plots_per_row = ceil(sqrt(feats))

    for i in range(D.shape[0]):
        if subplots:
            plt.subplot(plots_per_row, plots_per_row, i+1)
        else:
            plt.figure()
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

def fd_optimal_bins(D: np.ndarray):
    if D.ndim != 1:
        raise Exception("Wrong dimension number!")
    n = D.size
    p25, p75 = np.percentile(D, [25, 75])
    width = 2. * (p75 - p25)/n**(1./3)
    nbins = ceil((D.max() - D.min()) / width)
    nbins = max(1, nbins)
    return nbins

def scatter_hist_per_feat(D, L, label_dict, feature_dict=None, bins=None, subplots=True):
    plots_per_row = D.shape[0]
    
    # TODO: dynamic specification of size of the figure (it is plotted really bad)
    plt.figure(layout="tight", figsize=(20,15), dpi=200)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if subplots:
                pos = plots_per_row*i+j+1
                plt.subplot(plots_per_row, plots_per_row, pos)
            else:
                plt.figure()
            for key, value in label_dict.items():
                filtered_D_i = D[i, L==value]
                if i == j:
                    if bins is not None:
                        b = bins
                    else:
                        b = fd_optimal_bins((filtered_D_i).flatten())
                    plt.hist(filtered_D_i, bins=b, alpha=0.5, label=key, density=True)
                    if feature_dict:
                        plt.xlabel(feature_dict[i])
                else:
                    filtered_D_j = D[j, L==value]
                    plt.scatter(filtered_D_i, filtered_D_j, s=5)
                    # TODO: if features_dict is present, use it for this
                    if feature_dict:
                        plt.xlabel(feature_dict[i])
                        plt.ylabel(feature_dict[j])
            plt.legend(loc='upper right')

def mean(D):
    return col(D.mean(axis=1))

def center_data(D, mu=None):
    if mu is None:
        mu = mean(D)
    else:
        mu = col(mu)
    return D - mu

def var(D):
    return col(D.var(axis=1))