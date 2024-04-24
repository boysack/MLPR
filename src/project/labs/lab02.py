from packages.utils import *

def lab02(D, L, label_dict):
    scatter_hist_per_feat(D, L, label_dict)
    plt.title("Spoofing data")
    plt.show()

    print("Data variance vector")
    print(var(D))
    print("Data mean vector")
    print(mean(D))

    # testing with LDA, what is the most representative direction
    print("Further work with the knowledge of lab03")
    W_lda_m1, D_lda_m1 = lda(D, L, m=1)
    
    print("LDA directions")
    print(W_lda_m1)

    P_pca_m1, D_pca_m1 = pca(D, m=1)
    print("PCA directions")
    print(P_pca_m1)