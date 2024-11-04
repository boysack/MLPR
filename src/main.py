
from modules.data.dataset import load, split_db_2to1
from modules.visualization.plots import scatter_hist_per_feat
from modules.utils.operations import var, mean
from modules.utils.metrics import error_rate
from modules.features.dimensionality_reduction import lda, pca

import matplotlib.pyplot as plt
import numpy as np

def lab03(D, L, label_dict, DTR, LTR, DVAL, LVAL):

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

    # print results
    print("LDA 1")
    print(f"threshold = {threshold[0,0]}")
    print(f"shift = {shift} (sum to threshold)")
    print(f"error rate = {error_rate(LVAL, PVAL)}")

    # use a shift for the found threshold
    shift = 1.511

    # classify validation dataset
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[(DVAL_lda_m1[0] >= threshold + shift)[0]] = 1
    PVAL[(DVAL_lda_m1[0] < threshold + shift)[0]] = 0
    
    # print results
    print("LDA 1")
    print(f"threshold = {threshold[0,0]}")
    print(f"shift = {shift} (sum to threshold)")
    print(f"error rate = {error_rate(LVAL, PVAL)}")
    

    ###### PCA + LDA CLASSIFICATION ######
    
    # TODO: ask professor if it's normal the behaviour in function of m
    # m = 6 -> error_rate = 0.2410 (same as LDA with m=1 alone)
    # m = 5 -> error_rate = 0.2405 (optimal)
    # m = 4 -> error_rate = 0.2410
    # m = 3 -> error_rate = 0.2415
    # m = 2 -> error_rate = 0.7605
    # m = 1 -> error_rate = 0.2415 (?? why here is lower than m=2 ??)

    print()
    # find 6 (max) principal components
    P_pca_m6, V_pca_m6, _ = pca(DTR, m=6)
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

        # print results
        print(f"PCA {m} + LDA 1")
        print(f"PCA explained variance = {V_pca_m6[:m].sum()/V_pca_m6.sum()}")
        print(f"threshold = {threshold[0,0]}")
        print(f"shift = {shift} (sum to threshold)")
        print(f"error rate = {error_rate(LVAL, PVAL)}")

def main():
    D, L, label_dict = load("project/data/trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    ######### LAB 2 #########
    # FEATURE VISUALIZATION AND ANALYSIS (using whole dataset)

    #scatter_hist_per_feat(D, L, label_dict, plot_title="Spoofing data", save_path="./project/plots/original_data_scatter_matrix.png")
    
    ######### LAB 3 #########
    # PCA AND LDA EVALUATION (using whole dataset)
    
    # plot scatter matrix of original data
    scatter_hist_per_feat(D, L, label_dict)

    P_pca_m6, V_pca_m6, D_pca_m6 = pca(D, m=6)
    # plot scatter matrix of projected data onto principal components
    scatter_hist_per_feat(D_pca_m6, L, label_dict, save_path="./project/plots/PCA_6_scatter_matrix.png")

    W_lda_m1, D_lda_m1 = lda(D, L, m=1)
    # plot scatter matrix of projected data onto LDA directions
    scatter_hist_per_feat(D_lda_m1, L, label_dict, save_path="./project/plots/LDA_1_scatter_matrix.png")



    # LAB 4 - multivariate gaussian density

    #lab04(DTR, LTR, label_dict)

    # LAB 5 - multivariate gaussian classifiers

    #lab05(DTR, LTR, DVAL, LVAL, label_dict)

if __name__=="__main__":
    main()