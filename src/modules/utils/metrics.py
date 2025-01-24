import numpy as np
import pandas as pd

from modules.utils.operations import col, row, get_thresholds_from_llr

# TODO: check true_idx < 2

def error_rate(L, predictions):
    wrong_p = (L!=predictions).sum()
    error_rate = wrong_p/L.size
    return error_rate

def accuracy(L, predictions):
    return 1 - error_rate(L, predictions)

def conf_matrix(L, predictions, label_dict):
    # TODO: fix the non-zero based problem
    cfm = np.zeros((len(label_dict),len(label_dict)), dtype=int)
    for j in label_dict.values():
        # save prediction where actual label is j
        p = predictions[L==j]
        for i in label_dict.values():
            # i = predicted class, j = actual class
            cfm[i,j] = (p==i).sum()
    return cfm

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

def empirical_bayes_risk_binary(prior, L, llr = None, predictions = None, cost_matrix = None, true_idx = 1, normalize = True):
    false_idx = (true_idx + 1) % 2
    # to manage if the true_idx is not 0 or 1
    true_idx = (false_idx + 1) % 2

    if true_idx == 1:
        label_dict = {"False": 0, "True": 1}
    else:
        label_dict = {"True": 1, "False": 0}

    if cost_matrix is None:
        cost_matrix = np.ones((2, 2))
        np.fill_diagonal(cost_matrix, 0)

    if predictions is None:
        if llr is None:
            raise Exception("One between the parameters llr and predictions must be passed to calculate actDCF")
        # TODO: doesn't make sense to do so -> is passed 1 for true class every time
        l_scores = llr + np.log (prior*cost_matrix[false_idx, true_idx]/((1-prior)*cost_matrix[true_idx, false_idx]))
        predictions = np.empty(l_scores.shape)
        predictions[l_scores > 0] = 1
        predictions[l_scores <= 0] = 0
        
    if cost_matrix is None:
        cost_matrix = np.ones((2, 2))
        np.fill_diagonal(cost_matrix, 0)

    cfm = conf_matrix(L, predictions, label_dict)

    P_fn = cfm[false_idx, true_idx] / (cfm[false_idx, true_idx]+cfm[true_idx, true_idx])
    P_fp = cfm[true_idx, false_idx] / (cfm[true_idx, false_idx]+cfm[false_idx, false_idx])

    C_fn = cost_matrix[false_idx, true_idx]
    C_fp = cost_matrix[true_idx, false_idx]

    if normalize:
        d = min(prior * C_fn, (1-prior) * C_fp)
    else:
        d = 1

    return (prior * C_fn * P_fn + (1-prior) * C_fp * P_fp) / d

def empirical_bayes_risk(L, predictions, label_dict, priors, cost_matrix = None, normalize = True):
    # use neutral cost_matrix (useful with effective prior)
    if cost_matrix is None:
        n = len(label_dict)
        cost_matrix = np.ones((n, n))
        np.fill_diagonal(cost_matrix, 0)

    # priors and costs must be aligned as index (user responsibility)
    priors = col(np.array(priors))
    cfm = conf_matrix(L, predictions, label_dict)
    norm = cfm.sum(0)
    R = cfm / norm

    if normalize:
        d = np.min(np.dot(cost_matrix, priors))
    else:
        d = 1

    return (row(priors) * (R * cost_matrix).sum(0)).sum() / d

def min_DCF_binary(prior, L = None, llr = None, P_fn = None, P_fp = None, cost_matrix = None, thresholds = None, true_idx = 1, normalize = True, return_threshold = False):
    false_idx = (true_idx + 1) % 2
    if P_fn is None or P_fp is None:
        # must compute them
        if llr is None or L is None:
            raise Exception("One between the couple of parameters P_fn, P_fp or llr, L must be passed to calculate minDCF")
        P_fn, P_fp, thresholds = get_thresholds_from_llr(llr, L)

    # use neutral cost_matrix (useful with effective prior)
    if cost_matrix is None:
        cost_matrix = np.ones((2, 2))
        np.fill_diagonal(cost_matrix, 0)
    C_fn = cost_matrix[false_idx, true_idx]
    C_fp = cost_matrix[true_idx, false_idx]

    DCFs = prior * C_fn * P_fn + (1-prior) * C_fp * P_fp
    min_DCF_arg = np.argmin(DCFs)

    if normalize:
        d = min(prior * C_fn, (1-prior) * C_fp)
    else:
        d = 1

    if return_threshold and thresholds is not None:
        return DCFs[min_DCF_arg] / d, thresholds[min_DCF_arg]
    
    return DCFs[min_DCF_arg] / d


# Overlap Area Between Probability Density Function
def calculate_overlap(D0, D1):
    from modules.visualization.plots import fd_optimal_bins
    n_features, _ = D0.shape

    if n_features != D1.shape[0]:
        raise ValueError("Number of features must be the same for both classes.")

    overlaps = np.zeros(n_features)

    for feature_idx in range(n_features):
        feature_D0 = D0[feature_idx, :]
        feature_D1 = D1[feature_idx, :]

        combined_data = np.concatenate([feature_D0, feature_D1])
        bin_count = fd_optimal_bins(combined_data)
        bin_edges = np.histogram_bin_edges(combined_data, bins=bin_count)

        hist0, _ = np.histogram(feature_D0, bins=bin_edges, density=True)
        hist1, _ = np.histogram(feature_D1, bins=bin_edges, density=True)

        overlap = np.sum(np.minimum(hist0, hist1)) * np.diff(bin_edges).mean()
        overlaps[feature_idx] = overlap

    return overlaps