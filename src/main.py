
from modules.data.dataset import load, split_db_2to1
from modules.visualization.plots import scatter_hist_per_feat, correlation_heatmap, gaussian_hist_plot
from modules.utils.operations import var, mean, cov
from modules.utils.metrics import error_rate
from modules.features.dimensionality_reduction import lda, pca
from modules.models.mean_classifier import LdaBinaryClassifier

import matplotlib.pyplot as plt
import numpy as np

def main():
    D, L, label_dict = load("project/data/trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)


    ######### LAB 2 #########
    ######### FEATURE VISUALIZATION AND ANALYSIS (using whole dataset)
    """ 
    scatter_hist_per_feat(D, L, label_dict, plot_title="Spoofing data", save_path="./project/plots/original_data_scatter_matrix.png")
    correlation_heatmap(D)
    #scatter_hist_per_feat(D, L, label_dict, plot_title="Spoofing data")
    #W, _ = lda(D, L, m=1)
    #print(W)
     """

    ######### LAB 3 #########
    ######### PCA AND LDA EVALUATION (using whole dataset)
    """ 
    # plot scatter matrix of original data
    #scatter_hist_per_feat(D, L, label_dict)

    # plot scatter matrix of projected data onto principal components
    P_pca_m6, V_pca_m6, D_pca_m6 = pca(D, m=6)
    # compute differences in variance and mean wrt classes
    print(np.abs(var(D_pca_m6[:,L==0])-var(D_pca_m6[:,L==1])))
    print(np.abs(mean(D_pca_m6[:,L==0])-mean(D_pca_m6[:,L==1])))
    
    #scatter_hist_per_feat(D_pca_m6, L, label_dict, save_path="./project/plots/PCA_6_scatter_matrix.png")
    #scatter_hist_per_feat(D_pca_m6, L, label_dict)

    # plot scatter matrix of projected data onto LDA directions
    #W_lda_m1, D_lda_m1 = lda(D, L, m=1)
    #scatter_hist_per_feat(D_lda_m1, L, label_dict, save_path="./project/plots/LDA_1_scatter_matrix.png")
    #scatter_hist_per_feat(D_lda_m1, L, label_dict)

    # binary classification using LDA
    lmc = LdaBinaryClassifier(DTR, LTR, label_dict)
    lmc.fit()
    print(f"LDA classifier with shift {lmc.shift} error rate = {error_rate(LVAL, lmc.predict(DVAL))*100:.2f}%")

    # code used to find best threshold shift
    err = 1.0
    shift = None
    start = np.min(lmc.D)
    end = np.max(lmc.D)
    num = 10000
    for s in np.linspace(start, end, num):
        lmc.shift = s
        curr_err = error_rate(LVAL, lmc.predict(DVAL))
        if curr_err < err:
            err = curr_err
            shift = s
    print(f"Best threshold found for a range of {num} numbers in the interval [{start}, {end}]")
    print(f"LDA classifier with shift {shift} error rate = {err*100:.2f}%")

    # test PCA + LDA classification
    for m in range(6, 0, -1):
        DTR_pca = np.dot(P_pca_m6[:, :m].T, DTR)
        DVAL_pca = np.dot(P_pca_m6[:, :m].T, DVAL)

        lmc = LdaBinaryClassifier(DTR_pca, LTR, label_dict)
        lmc.fit()
        
        print(f"PCA m = {m} | LDA classifier error rate = {error_rate(LVAL, lmc.predict(DVAL_pca))*100:.2f}% | explained variance = {sum(V_pca_m6[:m])/sum(V_pca_m6)*100:.2f}%")
     """
    
    ######### LAB 4 #########
    ######### GAUSSIAN DENSITY EVALUATION
    """ 
    #gaussian_hist_plot(D, L, label_dict, save_path="./project/plots/gaussian_dist_histograms.png")
    gaussian_hist_plot(D, L, label_dict)
     """
    
    ######### LAB 5 #########
    ######### GAUSSIAN CLASSIFIERS

    #lab05(DTR, LTR, DVAL, LVAL, label_dict)

if __name__=="__main__":
    main()