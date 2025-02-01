import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil, sqrt
from screeninfo import get_monitors

from modules.utils.metrics import empirical_bayes_risk_binary, min_DCF_binary
from modules.utils.operations import get_thresholds_from_llr, row, mean, var, col
from modules.models.gaussians import logpdf_GAU_ND

# TODO: make the function receive a boolean parameter show, and a filepath where to save eventually the plot

def get_screen_size():
    primary_monitor = get_monitors()[0]
    screen_width = primary_monitor.width
    screen_height = primary_monitor.height
    return screen_width, screen_height

def hist_per_feat(D, L, label_dict, bins=None, subplots=True, plot_title = None, show = True, save_path = None):
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
    
    if plot_title:
        plt.title(plot_title)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

def fd_optimal_bins(D: ndarray):
    if D.ndim != 1:
        raise Exception("Wrong dimension number!")
    n = D.size
    p25, p75 = np.percentile(D, [25, 75])
    width = 2. * (p75 - p25)/n**(1./3)
    nbins = ceil((D.max() - D.min()) / width)
    nbins = max(1, nbins)
    return nbins

def scatter_hist_per_feat(D, L, label_dict, feature_dict=None, bins=None, subplots=True, plot_title=None, show=True, save_path=None):
    plots_per_row = D.shape[0]
    
    screen_width, screen_height = get_screen_size()
    base_dpi = 60 * 6**0.7
    dpi = int(base_dpi / (plots_per_row**0.7))
    fig, axes = plt.subplots(plots_per_row, plots_per_row, figsize=(screen_width/dpi, (screen_height/dpi)-0.7), dpi=dpi)

    # Check if axes is a single Axes object (when subplots is 1x1)
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])

    # Make each subplot square by setting the aspect ratio to equal
    for i in range(plots_per_row):
        for j in range(plots_per_row):
            ax = axes[i, j]  # Access the specific subplot (ax)
            
            for key, value in label_dict.items():
                filtered_D_i = D[i, L == value]
                if i == j:
                    # Histogram on diagonal
                    b = bins if bins is not None else fd_optimal_bins(filtered_D_i.flatten())
                    ax.hist(filtered_D_i, bins=b, alpha=0.5, label=key, density=True)
                    # Only show vertical grid on the histograms (diagonal plots)
                    ax.grid(True, linestyle=":", alpha=0.6, axis='x')  # Only vertical grid
                else:
                    filtered_D_j = D[j, L == value]
                    ax.scatter(filtered_D_j, filtered_D_i, label=key, s=5, alpha=0.5)
                    ax.grid(True, linestyle=":", alpha=0.6)

            # Make ticks invisible (but keep them for grid)
            ax.tick_params(axis='both', which='both', length=0)

            # Only remove Y ticks for non-first columns
            if i == 0 and j == plots_per_row - 1:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            elif j != 0 or (j == 0 and i == 0):
                ax.set_yticklabels([])
            # Only remove X ticks for non-last rows
            if i != plots_per_row - 1:
                ax.set_xticklabels([])

            # Only put legend on the top-right plot
            if i == 0 and j == plots_per_row - 1:
                ax.legend(loc='upper right')

    # Adjust layout to remove white space around subplots
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0, hspace=0)

    if plot_title:
        fig.suptitle(plot_title)

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()

# TODO: heatmap
def correlation_heatmap(D, plot_title = None, show = True, save_path = None):
    """
    Plot a heatmap of the correlation matrix for the given data array.

    Parameters:
    data (ndarray): A 2D array where rows are features and columns are samples.
    labels (list): Optional list of labels for the features.

    """
    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(D)
    
    # Create the heatmap plot
    sns.heatmap(correlation_matrix, annot=True, cmap="RdRd", cbar=True)

    # Add title and show the plot
    if plot_title:
        plt.title(plot_title)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

# TODO: gaussian distribution plot
def gaussian_hist_plot(D, L, label_dict, bins=None, plot_title = None, show = True, save_path = None, tied_cov = False):
    plots = D.shape[0]
    
    screen_width, screen_height = get_screen_size()
    base_dpi = 50 * 6**0.7
    dpi = int(base_dpi/(plots**(0.7/2)))
    plt.figure(layout="tight", figsize=(screen_width/dpi,(screen_height/dpi)-0.7), dpi=dpi)
    cols = ceil(plots**0.5)
    rows = round(plots**0.5)

    for f in range(D.shape[0]):
        ax = plt.subplot(rows, cols, f+1)
        ax.margins(x=0)
        fD = D[f, :].reshape(1,-1)
        min = np.min(fD) - 0.5
        max = np.max(fD) + 0.5
        if tied_cov:
            v = 0
            for label_str, label_int in label_dict.items():
                fcD = fD[:, L==label_int].reshape(1,-1)
                v += col(((fcD - (col(fcD.sum(axis=1)/fcD.shape[1])))**2).sum(axis=1))
            v /= fD.shape[1]
        for label_str, label_int in label_dict.items():
            fcD = fD[:, L==label_int].reshape(1,-1)
            mu = mean(fcD)
            if not tied_cov:
                v = var(fcD)
            if bins is not None:
                b = bins
            else:
                b = fd_optimal_bins((fcD).flatten())

            dist = np.linspace(min, max, b*4)
            p, = plt.plot(dist.ravel(), np.exp(logpdf_GAU_ND(row(dist), mu, v)), alpha=.8)
            c = p.get_color()

            plt.hist(fcD.ravel(), bins=b, density=True, alpha=.4, color=c, label=label_str)
        plt.legend(loc='upper right')

    if plot_title:
        plt.title(plot_title)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def bayes_error_plot_binary(L, llr, true_idx = 1, start = -3, stop = 3, num = 21, plot_title = None, plot_min_DCF = True, act_DCF_prefix = "", min_DCF_prefix = "", show = True, save_path = None, alpha = 1.0):
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

        line = plt.plot(eff_prior_log_odds, minDCFs, linestyle="dashed", alpha=alpha)
        color = line[0]._color
    else:
        for eff_prior in eff_priors:
            DCFs.append(empirical_bayes_risk_binary(prior=eff_prior, L=L, llr=llr, true_idx=true_idx))

    if color is not None:
        plt.plot(eff_prior_log_odds, DCFs, label=f'{act_DCF_prefix} DCF'.strip(), color=color, alpha=alpha)
    else:
        plt.plot(eff_prior_log_odds, DCFs, label=f'{act_DCF_prefix} DCF'.strip(), alpha=alpha)

    plt.legend()

    #plt.xlabel(r'$\log \dfrac{\tilde{\pi}}{1-\tilde{\pi}}$')
    plt.xlabel(r'$\log\left(\tilde{\pi}/(1-\tilde{\pi})\right)$')
    plt.ylabel("DCF")

    plt.xlim(start, stop)

    ylow = min(DCFs + minDCFs)
    ymax = max(DCFs + minDCFs)
    ymin = ymax - (ymax - ylow)*4/3
    ymin = 0
    ymax = 1
    plt.ylim(ymin, ymax)

    plt.grid(True, linestyle=':')

    if plot_title:
        plt.title(plot_title)
    if show:
        # TODO: not showing plots, make the figure to keep all the axis in it: try to write a better cleaning procedure than this (for now, clean it when saving)
        plt.show()
    if save_path:
        plt.savefig(save_path, pad_inches=0.2)
        plt.clf()

# TODO: check if it's convenient to do as in min_DCF_binary, i.e. to give the possibility to insert directly the scores and the labels and calculate the
# P_fn and P_fp in place
def roc(P_fn, P_fp, plot_title = None, show = True, save_path = None):
    plt.plot(P_fp, 1-P_fn)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=':')
    if plot_title:
        plt.title(plot_title)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

def plot_svm_boundary_2d_binary(model, D, L, dim_to_keep=[1,2], grid_resolution=100, margin=0.2, plot_title=None):
    if len(dim_to_keep) != 2:
        raise Exception("Can plot just 2D decision boundary: choose which features to keep.")
    # select only the last two features
    D_reduced = D[dim_to_keep, :]

    x_min, x_max = D_reduced[0, :].min() - margin, D_reduced[0, :].max() + margin
    y_min, y_max = D_reduced[1, :].min() - margin, D_reduced[1, :].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )

    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    fixed_values = mean(np.delete(D, dim_to_keep, axis=0))
    samples = fixed_values @ np.ones((1, grid_points.shape[1]))
    for i in range(len(dim_to_keep)):
        samples = np.insert(samples, dim_to_keep[i], grid_points[i,:], axis=0)

    Z = model.get_scores(samples)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.1, colors=["blue", "black", "red"])
    plt.contour(xx, yy, Z, levels=[0], colors="black", linewidths=2)  # Decision boundary
    plt.contour(xx, yy, Z, levels=[-1, 1], colors="red", linestyles="dashed")  # Margins

    for label in np.unique(L):
        if label == 0:
            l = "False"
        else:
            l = "True"
        plt.scatter(
            D_reduced[0, L == label], D_reduced[1, L == label],
            label=f"{l}", alpha=0.8, s=1
        )

    plt.xlabel(f"Feature {dim_to_keep[0]}")
    plt.ylabel(f"Feature {dim_to_keep[1]}")
    plt.legend()
    if plot_title is not None:
        plt.title(plot_title)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.show()

def plot_boundary_2d_binary(model, D, L, dim_to_keep=[1,2], grid_resolution=100, margin=.5, plot_title=None):
    if len(dim_to_keep) != 2:
        raise Exception("Can plot just 2D decision boundary: choose which features to keep.")
    # select only the last two features
    D_reduced = D[dim_to_keep, :]

    x_min, x_max = D_reduced[0, :].min() - margin, D_reduced[0, :].max() + margin
    y_min, y_max = D_reduced[1, :].min() - margin, D_reduced[1, :].max() + margin

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution), 
                         np.linspace(y_min, y_max, grid_resolution))

    
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    fixed_values = mean(np.delete(D, dim_to_keep, axis=0))
    samples = fixed_values @ np.ones((1, grid_points.shape[1]))
    for i in range(len(dim_to_keep)):
        samples = np.insert(samples, dim_to_keep[i], grid_points[i,:], axis=0)

    ll = model.get_scores(samples)
    # addapt to work even directly with llrs
    llr = ll[1] - ll[0] # binary
    llr = llr.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, llr, levels=0, alpha=0.1, colors=["blue", "red"])

    for label in np.unique(L):
        if label == 0:
            l = "False"
        else:
            l = "True"
        plt.scatter(
            D_reduced[0, L == label], D_reduced[1, L == label],
            label=f"{l}", alpha=0.8, s=1
        )
    
    # Labels and formatting
    plt.xlabel(f"Feature {dim_to_keep[0]}")
    plt.ylabel(f"Feature {dim_to_keep[1]}")
    if plot_title is not None:
        plt.title(plot_title)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.show()