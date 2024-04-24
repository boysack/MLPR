from packages.utils import *
from labs.lab02 import lab02
from pprint import pprint

if __name__=="__main__":
    D, L, label_dict = load("project/data/trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # LAB 2 - features visualization

    #lab02(D, L, label_dict)
    
    # LAB 3 - pca, lda
    # to quantitatively evaluate the overlap, I could calculate the distance between the means for each feature

    # TODO: ask professor if is better to perform PCA and LDA just on training data to evaluate graphically
    
    ###### VISUALIZATION ######

    # plot scatter matrix of original data
    scatter_hist_per_feat(D, L, label_dict)
    plt.show()

    # find 6 (max) principal components ad project data
    P_pca_m6, D_pca_m6 = pca(D, m=6)
    # plot scatter matrix of projected data
    scatter_hist_per_feat(D_pca_m6, L, label_dict)
    plt.show()

    # find 1 (max) discriminant direction using LDA and project data
    W_lda_m1, D_lda_m1 = lda(D, L, m=1)
    # plot scatter matrix of projected data
    scatter_hist_per_feat(D_lda_m1, L, label_dict)
    plt.show()


    ###### LDA CLASSIFICATION ######

    # find 1 (max) discriminant direction using LDA just on training data and project them
    W_lda_m1, DTR_lda_m1 = lda(DTR, LTR, m=1, change_sign=True)
    # find the threshold using the mean of the means of the classes
    threshold = (mean(DTR_lda_m1[:,LTR==0]) + mean(DTR_lda_m1[:,LTR==0]))/2
    # use a shift for the found threshold
    shift = 0

    # project validation dataset
    DVAL_lda_m1 = np.dot(W_lda_m1.T, DVAL)

    # classify validation dataset
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[(DVAL_lda_m1[0] >= threshold + shift)[0]] = 1
    PVAL[(DVAL_lda_m1[0] < threshold + shift)[0]] = 0

    # count wrong labels
    error_p = (LVAL!=PVAL).sum()
    # calculate error rate
    error_rate = error_p/LVAL.shape[0]

    # print results
    print("LDA 1")
    print(f"threshold = {threshold[0,0]}")
    print(f"shift = {shift} (sum to threshold)")
    print(f"error rate = {error_rate}")

    # use a shift for the found threshold
    shift = 1.511

    # classify validation dataset
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[(DVAL_lda_m1[0] >= threshold + shift)[0]] = 1
    PVAL[(DVAL_lda_m1[0] < threshold + shift)[0]] = 0
    
    # count wrong labels
    error_p = (LVAL!=PVAL).sum()
    # calculate error rate
    error_rate = error_p/LVAL.shape[0]

    # print results
    print("LDA 1")
    print(f"threshold = {threshold[0,0]}")
    print(f"shift = {shift} (sum to threshold)")
    print(f"error rate = {error_rate}")
    

    ###### PCA + LDA CLASSIFICATION ######
    
    # find 6 (max) principal components
    P_pca_m6, _ = pca(DTR, m=6)
    for m in range(6, 0, -1):
        # project training data based on first m principal components
        DTR_pca = np.dot(P_pca_m6[:,:m].T, DTR)
        # apply LDA on previously projected training data
        W_lda_m1, DTR_lda_m1 = lda(DTR_pca, LTR, m=1, change_sign=True)
        # find the threshold using the mean of the means of the classes of training data
        threshold = (mean(DTR_lda_m1[:,LTR==0]) + mean(DTR_lda_m1[:,LTR==0]))/2
        # use a shift for the found threshold
        shift = 0

        # project validation data based on previously performed PCA and LDA on training data
        DVAL_lda_m1 = np.dot(W_lda_m1.T, (np.dot(P_pca_m6[:,:m].T, DVAL)))

        # classify validation dataset
        PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
        PVAL[(DVAL_lda_m1[0] >= threshold + shift)[0]] = 1
        PVAL[(DVAL_lda_m1[0] < threshold + shift)[0]] = 0
        
        # count wrong labels
        error_p = (LVAL!=PVAL).sum()
        # calculate error rate
        error_rate = error_p/LVAL.shape[0]

        # print results
        print(f"PCA {m} + LDA 1")
        print(f"threshold = {threshold[0,0]}")
        print(f"shift = {shift} (sum to threshold)")
        print(f"error rate = {error_rate}")
