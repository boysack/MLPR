from packages.utils import *
from labs.labs import lab02, lab03
from pprint import pprint

if __name__=="__main__":
    D, L, label_dict = load("project/data/trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # LAB 2 - features visualization

    lab02(D, L, label_dict)
    
    # LAB 3 - pca, lda
    
    #lab03(D, L, label_dict, DTR, LTR, DVAL, LVAL)

    # LAB 4 - gaussian model

    # calculate mean and covariance
    mu = []
    C = []
    for label_str, label_int in label_dict.items():
        cD = D[:, L==label_int]
        mu.append(mean(cD))
        C.append(cov(cD))

    screen_width, screen_height = get_screen_size()
    dpi = 50
    plt.figure(layout="tight", figsize=(screen_width/dpi,(screen_height/dpi)-0.7), dpi=dpi)
    M = D.shape[0]
    ppr = ceil(sqrt(M))
    for i in range(M):
        plt.subplot(ppr, ppr, i+1)
        for label_str, label_int in label_dict.items():
            fcD = D[i, L==label_int]
            plt.hist(fcD.ravel(), bins=50, density=True, alpha=.5)
            XPlot = np.linspace(-8, 12, 1000)
            plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(row(XPlot), mu[label_int][i], C[label_int][i, i])), alpha=.5)
            plt.title(f"feature {i+1:02}")

    plt.show()
    plt.savefig("/Users/claudio/Documents/turin/polito/anno I/semestre II/mlpr/2324/git/MLPR/src/project/plots/UGM_histograms_per_features.png")
