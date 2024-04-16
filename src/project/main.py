from packages.utils import *
from pprint import pprint

if __name__=="__main__":
    # LAB 2 - features visualization
    D, L, label_dict = load("project/data/trainData.txt")
    scatter_hist_per_feat(D, L, label_dict)
    plt.title("Spoofing data")
    #plt.show()

    print(var(D))
    print(D.var(axis=1))
    print(mean(D))

    # testing with LDA, what is the most representative direction

    W_lda_m1, D_lda_m1 = lda(D, L, m=1)
    print(W_lda_m1)

    P_pca_m1, D_pca_m1 = pca(D, m=1)
    print(P_pca_m1)
    
    # LAB 3 - pca, lda
    """ P_pca_m6, D_pca_m6 = pca(D, m=6)
    scatter_hist_per_feat(D_pca_m6, L, label_dict, bins=100)
    plt.title("pca")


    W_m5, D_pca_m6 = lda(D_pca_m6, L, m=1)
    scatter_hist_per_feat(D_pca_m6, L, label_dict, bins=100)
    plt.title("lda")
    plt.show() """

    # LDA direction evaluation
