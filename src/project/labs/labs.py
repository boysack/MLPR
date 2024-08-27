from modules.utils.operations import *
from modules.visualization.plots import *
from modules.models.gaussians import logpdf_GAU_ND, MVGModel, TiedGModel, NaiveGModel, error_rate
from modules.models.mean_classifier import LdaBinaryClassifier
from modules.features.dimensionality_reduction import lda, pca
import numpy as np

def lab02(D, L, label_dict):
    scatter_hist_per_feat(D, L, label_dict)
    plt.title("Spoofing data")
    plt.show()
    #plt.savefig("/Users/claudio/Documents/turin/polito/anno I/semestre II/mlpr/2324/git/MLPR/src/project/plots/original_data_scatter_matrix.png")

    print("Data variance vector")
    print(var(D))
    print("Data mean vector")
    print(mean(D))

    # testing with LDA, what is the most representative direction
    print("Further work with the knowledge of lab03")
    W_lda_m1, D_lda_m1 = lda(D, L, m=1)
    
    print("LDA directions")
    print(W_lda_m1)

    P_pca_m1, _, D_pca_m1 = pca(D, m=1)
    print("PCA directions")
    print(P_pca_m1)

def lab03(D, L, label_dict, DTR, LTR, DVAL, LVAL):
    # to quantitatively evaluate the overlap, I could calculate the distance between the means for each feature

    # TODO: ask professor if is better to perform PCA and LDA just on training data to evaluate graphically

    ###### VISUALIZATION ######

    # plot scatter matrix of original data
    scatter_hist_per_feat(D, L, label_dict)
    plt.show()

    # find 6 (max) principal components ad project data
    P_pca_m6, V_pca_m6, D_pca_m6 = pca(D, m=6)
    # plot scatter matrix of projected data
    scatter_hist_per_feat(D_pca_m6, L, label_dict)
    #plt.show()
    plt.savefig("/Users/claudio/Documents/turin/polito/anno I/semestre II/mlpr/2324/git/MLPR/src/project/plots/PCA_6_scatter_matrix.png")

    # find 1 (max) discriminant direction using LDA and project data
    W_lda_m1, D_lda_m1 = lda(D, L, m=1)
    # plot scatter matrix of projected data
    scatter_hist_per_feat(D_lda_m1, L, label_dict)
    #plt.show()
    plt.savefig("/Users/claudio/Documents/turin/polito/anno I/semestre II/mlpr/2324/git/MLPR/src/project/plots/LDA_1_scatter_matrix.png")


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
        
        # count wrong labels
        error_p = (LVAL!=PVAL).sum()
        # calculate error rate
        error_rate = error_p/LVAL.shape[0]

        # print results
        print(f"PCA {m} + LDA 1")
        print(f"PCA explained variance = {V_pca_m6[:m].sum()/V_pca_m6.sum()}")
        print(f"threshold = {threshold[0,0]}")
        print(f"shift = {shift} (sum to threshold)")
        print(f"error rate = {error_rate}")

def lab04(D, L, label_dict):
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
        plt.subplot(3, 2, i+1)
        for label_str, label_int in label_dict.items():
            fcD = D[i, L==label_int]
            plt.hist(fcD.ravel(), bins=50, density=True, alpha=.5)
            XPlot = np.linspace(-8, 12, 1000)
            plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(row(XPlot), mu[label_int][i], C[label_int][i, i])), alpha=.5)
            plt.title(f"feature {i+1:02}")

    plt.show()
    #plt.savefig("/Users/claudio/Documents/turin/polito/anno I/semestre II/mlpr/2324/git/MLPR/src/project/plots/UGM_histograms_per_features.png")

def lab05(DTR, LTR, DTE, LTE, label_dict):
    # set manually priors
    priors = np.array([0.5, 0.5])
    l_priors = np.log(priors)

    mvg = MVGModel(DTR, LTR, label_dict, l_priors)
    mvg.fit()
    mvg_p, _ = mvg.predict(DTE)

    ng = NaiveGModel(DTR, LTR, label_dict, l_priors)
    ng.fit()
    ng_p, _ = ng.predict(DTE)

    tg = TiedGModel(DTR, LTR, label_dict, l_priors)
    tg.fit()
    tg_p, _ = tg.predict(DTE)

    ldac = LdaBinaryClassifier(DTR, LTR, label_dict)
    ldac.fit()
    ldac_p = ldac.predict(DTE)

    print("#####################################")
    print("############ ALL FEATURES ###########")
    print("#####################################\n")

    print(f"MULTIVARIATE GAUSSIAN MODEL {error_rate(LTE, mvg_p)*100:0.2f}")
    print(f"NAIVE GAUSSIAN MODEL {error_rate(LTE, ng_p)*100:0.2f}")
    print(f"TIED GAUSSIAN MODEL {error_rate(LTE, tg_p)*100:0.2f}")
    print()
    #print(f"LDA BINARY CLASSIFIER {error_rate(LTE, ldac_p)*100:0.2f}")

    """ 
    p_max_val = {}
    with np.printoptions(suppress=True, formatter={'float': '{:0.5f}'.format}, linewidth=100):    
        for label_int in label_dict.values():
            print(f"CLASS {label_int} PARAMETERS")
            print("Mean")
            print(mvg.parameters[label_int][0])
            print("Covariance matrix")
            print(mvg.parameters[label_int][1])
            print("Pearson correlation coefficient")
            p_corr_m = p_corr(mvg.parameters[label_int][1])
            print(p_corr_m, end="\n\n")

            np.fill_diagonal(p_corr_m, 0)
            max_cols = np.argmax(p_corr_m, axis=1)
            row = np.argmax(np.array([p_corr_m[row, col] for row, col in enumerate(max_cols)]))
            col = max_cols[row]

            p_max_idx = (row, col)

            p_max_val[label_int] = (p_corr_m[row, col], p_max_idx)

    print(p_max_val, end="\n\n")

    print("##### REMOVE LAST TWO FEATURES ######")
    print("(Gaussian assumption poorly satisfied)")

    DTR_red = DTR[:-2,:]
    DTE_red = DTE[:-2,:]
    mvg = MVGModel(DTR_red, LTR, label_dict, l_priors)
    mvg.fit()
    mvg_p, _ = mvg.predict(DTE_red)

    ng = NaiveGModel(DTR_red, LTR, label_dict, l_priors)
    ng.fit()
    ng_p, _ = ng.predict(DTE_red)

    tg = TiedGModel(DTR_red, LTR, label_dict, l_priors)
    tg.fit()
    tg_p, _ = tg.predict(DTE_red)

    ldac = LdaBinaryClassifier(DTR_red, LTR, label_dict)
    ldac.fit()
    ldac_p = ldac.predict(DTE_red)

    print("#### MULTIVARIATE GAUSSIAN MODEL ####")
    print(error_rate(LTE, mvg_p)*100, end="\n\n")
    # +0.95
    print("######## NAIVE GAUSSIAN MODEL #######")
    print(error_rate(LTE, ng_p)*100, end="\n\n")
    # +0.45
    print("######## TIED GAUSSIAN MODEL ########")
    print(error_rate(LTE, tg_p)*100, end="\n\n")
    # +0.20
    print("####### LDA BINARY CLASSIFIER #######")
    print(error_rate(LTE, ldac_p)*100, end="\n\n")

    # Naive Bayes outperforms Multivariate, probably due to robustness to small datasets (can lead to overfitting if parameters are too much)
    # Tied degradates less than the other, but still perform worse (9.5 vs ~7.5)

    print("##### KEEP FIRST TWO FEATURES ######")
    print("(similar means, different variance)")

    DTR_red = DTR[:2,:]
    DTE_red = DTE[:2,:]

    mvg = MVGModel(DTR_red, LTR, label_dict, l_priors)
    mvg.fit()
    mvg_p, _ = mvg.predict(DTE_red)

    tg = TiedGModel(DTR_red, LTR, label_dict, l_priors)
    tg.fit()
    tg_p, _ = tg.predict(DTE_red)

    print("#### MULTIVARIATE GAUSSIAN MODEL ####")
    print(error_rate(LTE, mvg_p)*100, end="\n\n")
    print("######## TIED GAUSSIAN MODEL ########")
    print(error_rate(LTE, tg_p)*100, end="\n\n")

    # BAD BAD, multivariate bad, tied really worse (class gaussians are really overlapped [see saved plot])

    print("##### KEEP 3 AND 4 FEATURES ######")
    print("(different means, similar variance)")

    DTR_red = DTR[2:4,:]
    DTE_red = DTE[2:4,:]
    
    mvg = MVGModel(DTR_red, LTR, label_dict, l_priors)
    mvg.fit()
    mvg_p, _ = mvg.predict(DTE_red)

    tg = TiedGModel(DTR_red, LTR, label_dict, l_priors)
    tg.fit()
    tg_p, _ = tg.predict(DTE_red)

    print("#### MULTIVARIATE GAUSSIAN MODEL ####")
    print(error_rate(LTE, mvg_p)*100, end="\n\n")
    print("######## TIED GAUSSIAN MODEL ########")
    print(error_rate(LTE, tg_p)*100, end="\n\n")

    # JUST USING 3 AND 4 AS FEATURES PERFORMANCES DEGRADATES JUST OF 2%, BESIDES TIED PERFORM EVEN
    # BETTER THAN MULTIVARIATE (9.45 mvg, 9.4 tied) """

    C = cov(DTR)
    for m in range(1, DTR.shape[0]+1)[::-1]:
        print("#####################################")
        print(f"############# PCA m = {m} #############")
        print("#####################################\n")
        P, V, DTR_m = pca(DTR, C, m=m)
        print(f"Explained variance {V[:m].sum()/V.sum()}")
        DTE_m = np.dot(P.T, DTE)

        mvg = MVGModel(DTR_m, LTR, label_dict, l_priors)
        mvg.fit()
        mvg_p, _ = mvg.predict(DTE_m)

        ng = NaiveGModel(DTR_m, LTR, label_dict, l_priors)
        ng.fit()
        ng_p, _ = ng.predict(DTE_m)

        tg = TiedGModel(DTR_m, LTR, label_dict, l_priors)
        tg.fit()
        tg_p, _ = tg.predict(DTE_m)

        print(f"MULTIVARIATE GAUSSIAN MODEL {error_rate(LTE, mvg_p)*100:0.2f}")
        print(f"NAIVE GAUSSIAN MODEL {error_rate(LTE, ng_p)*100:0.2f}")
        print(f"TIED GAUSSIAN MODEL {error_rate(LTE, tg_p)*100:0.2f}\n")
    
    """ _, _, DTR_m = pca(DTR, C, m=6)
    with np.printoptions(suppress=True, formatter={'float': '{:0.10f}'.format}, linewidth=100000):    
        print(p_corr(cov(DTR_m))) """
    
    # found Principal Components are not correlated

    print("#####################################")
    print(f"############# LDA m = 1 #############")
    print("#####################################\n")

    W, DTR_m = lda(DTR, LTR)
    DTE_m = np.dot(W.T, DTE)

    mvg = MVGModel(DTR_m, LTR, label_dict, l_priors)
    mvg.fit()
    mvg_p, _ = mvg.predict(DTE_m)

    ng = NaiveGModel(DTR_m, LTR, label_dict, l_priors)
    ng.fit()
    ng_p, _ = ng.predict(DTE_m)

    tg = TiedGModel(DTR_m, LTR, label_dict, l_priors)
    tg.fit()
    tg_p, _ = tg.predict(DTE_m)

    print(f"MULTIVARIATE GAUSSIAN MODEL {error_rate(LTE, mvg_p)*100:0.2f}")
    print(f"NAIVE GAUSSIAN MODEL {error_rate(LTE, ng_p)*100:0.2f}")
    print(f"TIED GAUSSIAN MODEL {error_rate(LTE, tg_p)*100:0.2f}\n")