
from modules.data.dataset import load, split_db_2to1
from modules.models.support_vector_machine import SupportVectorMachine
from modules.models.training import kfold_scores_pooling, llr_calibration, model_fusion
from modules.visualization.plots import scatter_hist_per_feat, correlation_heatmap, gaussian_hist_plot, bayes_error_plot_binary, plot_svm_boundary_2d_binary, plot_boundary_2d_binary
from modules.utils.operations import var, mean, cov, p_corr, trunc, effective_prior_binary, col
from modules.utils.metrics import error_rate, calculate_overlap, empirical_bayes_risk_binary, min_DCF_binary
from modules.features.dimensionality_reduction import lda, pca, pca_pipe, lda_pipe
from modules.features.transformation import L2_normalization, center_data, no_op, quadratic_feature_mapping, withening, z_normalization
from modules.models.mean_classifier import LdaBinaryClassifier
from modules.models.gaussians import MVGModel, TiedGModel, NaiveGModel, TiedNaiveGModel, MVGMModel, TiedGMModel, DiagGMModel
from modules.models.logistic_regression import LogisticRegression

import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import namedtuple

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

def main():
    # REMEMBER: the load function ensures that, for dataset labeled with 0 for false and 1 for true, the data is correctly labeled in the label_dictionary
    D, L, label_dict = load("project/data/trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    DEVAL, LEVAL, _ = load("project/data/evalData.txt")
    # label are correctly assigned

    ######### LAB 2 #########
    ######### FEATURE VISUALIZATION AND ANALYSIS (using whole dataset)
    """ 
    scatter_hist_per_feat(D, L, label_dict)
    #scatter_hist_per_feat(D, L, label_dict, save_path="./project/plots/original_data_scatter_matrix.png")
    #correlation_heatmap(D)
    #W, _ = lda(D, L, m=1)
    #print(W)
     """
    # TODO: overlap quantitative evaluation
    """ print(calculate_overlap(D[:,L==0], D[:,L==1]).mean())
    print(calculate_overlap(pca(D, m=D.shape[0])[2][:,L==0], pca(D, m=D.shape[0])[2][:,L==1]).mean()) """

    ######### LAB 3 #########
    ######### PCA AND LDA EVALUATION (using whole dataset)
    
    """ 
    # plot scatter matrix of original data
    #scatter_hist_per_feat(D, L, label_dict)

    # plot scatter matrix of projected data onto principal components
    P_pca_m6, V_pca_m6, D_pca_m6 = pca(D, m=6)
    scatter_hist_per_feat(D_pca_m6, L, label_dict, save_path="./project/plots/PCA_6_scatter_matrix.png")
    #scatter_hist_per_feat(D_pca_m6, L, label_dict)
    
    # compute differences in variance and mean wrt classes
    #print(np.abs(var(D_pca_m6[:,L==0])-var(D_pca_m6[:,L==1])))
    #print(np.abs(mean(D_pca_m6[:,L==0])-mean(D_pca_m6[:,L==1])))
    
    # plot scatter matrix of projected data onto LDA directions
    W_lda_m1, D_lda_m1 = lda(D, L, m=1)
    scatter_hist_per_feat(D_lda_m1, L, label_dict, save_path="./project/plots/LDA_1_scatter_matrix.png")
    #scatter_hist_per_feat(D_lda_m1, L, label_dict)
    # binary classification using LDA
    lmc = LdaBinaryClassifier(DTR, LTR, label_dict)
    lmc.fit()
    print(f"LDA classifier with shift {lmc.shift} error rate = {error_rate(LVAL, lmc.predict(DVAL)[0])*100:.2f}%")

    # code used to find best threshold shift
    err = 1.0
    shift = None
    start = np.min(lmc.D)
    end = np.max(lmc.D)
    num = 10000
    for s in np.linspace(start, end, num):
        lmc.shift = s
        curr_err = error_rate(LVAL, lmc.predict(DVAL)[0])
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
        
        print(f"PCA m = {m} | LDA classifier error rate = {error_rate(LVAL, lmc.predict(DVAL_pca)[0])*100:.2f}% | explained variance = {sum(V_pca_m6[:m])/sum(V_pca_m6)*100:.2f}%")
     """
    
    ######### LAB 4 #########
    ######### GAUSSIAN DENSITY EVALUATION
    """ 
    # plot per class features histograms and compare with gaussian distribution found with mean and variance of data itself
    #gaussian_hist_plot(D, L, label_dict, save_path="./project/plots/gaussian_dist_histograms.png")
    gaussian_hist_plot(D, L, label_dict)
    
    # apply PCA first
    #gaussian_hist_plot(pca(D, m=D.shape[0])[2], L, label_dict, save_path="./project/plots/PCA_gaussian_dist_histograms.png")
    gaussian_hist_plot(pca(D, m=D.shape[0])[2], L, label_dict)
    
    # do the same for tied covariance model (use the same variance for both classes)
    #gaussian_hist_plot(D, L, label_dict, save_path="./project/plots/tied_cov_gaussian_dist_histograms.png", tied_cov=True)
    gaussian_hist_plot(D, L, label_dict, tied_cov=True)
    # apply PCA first
    #gaussian_hist_plot(pca(D, m=D.shape[0])[2], L, label_dict, save_path="./project/plots/PCA_tied_cov_gaussian_dist_histograms.png", tied_cov=True)
    gaussian_hist_plot(pca(D, m=D.shape[0])[2], L, label_dict, tied_cov=True)
     """
    
    ######### LAB 5 #########
    ######### GAUSSIAN CLASSIFIERS

    # Gaussian models using uniform class priors
    
    models = [
        MVGModel,
        NaiveGModel,
        TiedGModel,
        TiedNaiveGModel,
        #LdaBinaryClassifier
    ]

    P_pca_m6, _, _ = pca(DTR, m=6)

    descs = ["RAW"]
    tr_datas = [DTR]
    vl_datas = [DVAL]
    for m in range(6,0,-1):
        descs.append(f"PC{m}")
        tr_datas.append(np.dot(P_pca_m6[:, :m].T, DTR))
        vl_datas.append(np.dot(P_pca_m6[:, :m].T, DVAL))
    
    """ descs.append("features 1-2")
    tr_datas.append(DTR[[0,1], :])
    vl_datas.append(DVAL[[0,1], :])

    descs.append("features 3-4")
    tr_datas.append(DTR[[2,3], :])
    vl_datas.append(DVAL[[2,3], :])
        
    descs.append("features 1-4")
    tr_datas.append(DTR[:4, :])
    vl_datas.append(DVAL[:4, :]) """

    results = {}
    pi = .1
    """ 
    for with_app_lines in [True, False]:
        for model_cstr in models:
            print()
            print(f"{model_cstr.get_model_name():5.5}")
            for desc, tr_data, vl_data in zip(descs, tr_datas, vl_datas):
                
                model = model_cstr(tr_data, LTR, label_dict, l_priors=np.log([(1-pi),pi]))
                model.fit()
                predictions, scores = model.predict(vl_data)

                results[(m, model.get_model_name())] = {
                        "model": model,
                        "prior": pi,
                        "error_rate": error_rate(LVAL, predictions),
                        "min_dcf": min_DCF_binary(.1, LVAL, scores),
                        "act_dcf": empirical_bayes_risk_binary(.1, LVAL, scores),
                        "llr": scores
                }
                bayes_error_plot_binary(LVAL, scores, plot_title=f"{model.get_model_name()} / {desc}", start = -4, stop = 4, show=False)
                save_name = f"bayes_error_plot_{model.get_model_name()}_{desc}"
                if with_app_lines:
                    save_name += "_with_app_lines"
                    plt.axvline(np.log(.1/(1-.1)), c="red", alpha=.5)
                    plt.axvline(np.log(.5/(1-.5)), c="red", alpha=.5)
                    plt.axvline(np.log(.9/(1-.9)), c="red", alpha=.5)
                plt.savefig(f"./project/plots/gaussian_model_bayes_error_plots/{save_name}.png")
                plt.clf()

                #print(f"{model.get_model_name():5.5} |{"".join([" "+str(k)+": "+str(v)+" |" for k, v in model.get_model_params().items()])} error_rate = {results[(m, model.get_model_name())]*100:5.2f}% | minDCF = {min_DCF_binary(.1, LVAL, scores):.4f} | actDCF = {empirical_bayes_risk_binary(.1, LVAL, scores):.4f}")
                # use this to make results for latex table
                #print(f"& {results[(m, model.get_model_name())]*100:.2f}\% & {min_DCF_binary(pi, LVAL, scores):.4f} & {empirical_bayes_risk_binary(pi, LVAL, scores):.4f} & {empirical_bayes_risk_binary(pi, LVAL, scores)-min_DCF_binary(pi, LVAL, scores):.4f}")
    """
    # print per class covariance and correlation
    """ 
    np.set_printoptions(suppress=True)
    print("covariance matrices")
    print(trunc(cov(DTR[:,LTR==0]), decs=3))
    print(trunc(cov(DTR[:,LTR==1]), decs=3))
     
    print("correlation matrices")
    print(trunc(p_corr(cov(DTR[:,LTR==0])), decs=3))
    print(trunc(p_corr(cov(DTR[:,LTR==1])), decs=3))
    np.set_printoptions(suppress=False)
    #print(trunc(p_corr(cov(DTR)), decs=3))
     """

    # EXTRA: try stuff with Naive Bayes
    """ p_mat0 = p_corr(cov(DTR[:,LTR==0]))
    p_mat1 = p_corr(cov(DTR[:,LTR==1]))

    #print(trunc(p_mat0, decs=3))
    #print(trunc(p_mat1, decs=3))
    
    np.fill_diagonal(p_mat0, 0)
    np.fill_diagonal(p_mat1, 0)

    args0 = [(int(i/p_mat0.shape[0]), int(i%p_mat0.shape[0])) for i in np.argsort(abs(p_mat0).flatten())[::-1] if int(i/p_mat0.shape[0])!=i%p_mat0.shape[0]][::2]
    features = range(p_mat0.shape[0])
    removed = []
    print("CLASS 0")
    print("USED FEATURES | ERROR RATE")
    for i in range(len(args0)):
        if args0[i][0] not in removed and args0[i][1] not in removed:
            removed.append(args0[i][1])
        features_loc = [f for f in features if f not in removed]
        if len(features_loc) == 0:
            break
        m = NaiveGModel(DTR[features_loc,:], LTR, label_dict)
        m.fit()
        pred, _ = m.predict(DVAL[features_loc,:])
        print(f"{features_loc} -> {error_rate(LVAL, pred)*100:.2f}%")

    args1 = [(int(i/p_mat1.shape[1]), int(i%p_mat1.shape[0])) for i in np.argsort(abs(p_mat1).flatten())[::-1] if int(i/p_mat1.shape[0])!=i%p_mat1.shape[0]][::2]
    print("CLASS 1")
    print("USED FEATURES | ERROR RATE")
    for i in range(len(args1)):
        if args1[i][0] not in removed and args1[i][1] not in removed:
            removed.append(args1[i][1])
        features_loc = [f for f in features if f not in removed]
        if len(features_loc) == 0:
            break
        m = NaiveGModel(DTR[features_loc,:], LTR, label_dict)
        m.fit()
        pred, _ = m.predict(DVAL[features_loc,:])
        print(f"{features_loc} -> {error_rate(LVAL, pred)*100:.2f}%")
     """

    ######### LAB 7 #########
    ######### DIFFERENT APPLICATION | MINIMUM DETECTION COSTS | ROC CURVES | BAYES ERROR PLOTS
    """ 
    # chosen configuration
    final_config = [
        (MVGModel, DTR, DVAL, "RAW"),
        (NaiveGModel, DTR, DVAL, "RAW"),
        (TiedGModel, np.dot(P_pca_m6[:,:4].T, DTR), np.dot(P_pca_m6[:,:4].T, DVAL), "PC4"),
        (TiedNaiveGModel, np.dot(P_pca_m6[:,:2].T, DTR), np.dot(P_pca_m6[:,:2].T, DVAL), "PC2")
    ]

    for with_app_lines in [True, False]:
        save_name = "bayes_error_plot_Gaussian_selection"
        if with_app_lines:
            save_name += "_with_app_lines"
        if with_app_lines:
            plt.axvline(np.log(.1/(1-.1)), c="red", alpha=.5)
            plt.axvline(np.log(.5/(1-.5)), c="red", alpha=.5)
            plt.axvline(np.log(.9/(1-.9)), c="red", alpha=.5)
        for cfg in final_config:
            model = cfg[0](cfg[1], LTR, label_dict, l_priors=np.log([1-pi, pi]))
            model.fit()
            p, s = model.predict(cfg[2])

            bayes_error_plot_binary(LVAL, s, plot_title=f"Selected Gaussian Models", start = -4, stop = 4, show=False, act_DCF_prefix=f"{model.get_model_name()}[{cfg[3]}]", alpha=.5)
        plt.show()
        #plt.savefig(f"./project/plots/gaussian_model_bayes_error_plots/{save_name}.png")
        #plt.clf()
     """
    ######### LAB 8 #########
    ######### LOGISTIC REGRESSION

    ls = np.logspace(-4, 2, 100)
    pi = 0.1

    """ 
    # non-prior-weighted logistic regression plot and minDCF minimum

    # use 50 samples to check if regularization have less impact using 50 samples only
    #DTR = DTR[:, ::50]
    #LTR = LTR[::50]
    #D = D[:, ::50]
    #L = L[::50]

    dcfs = []
    min_dcfs = []
    llrs = []
    for l in ls:
        ### decomment/comment this part to use/not use held-out validation set 
        #lr = LogisticRegression(DTR, LTR, label_dict, l)
        #lr.fit()
        #predictions, llr = lr.predict(DVAL) 
        #dcfs.append(empirical_bayes_risk_binary(pi, LVAL, llr))
        #min_dcfs.append(min_DCF_binary(pi, LVAL, llr))

        ### decomment/comment this part to use/not use kfold validation
        llr, predictions, loc_LVAL = kfold_scores_pooling(D, L, LogisticRegression, {"l":l, "label_dict":label_dict})
        dcfs.append(empirical_bayes_risk_binary(pi, loc_LVAL, llr))
        min_dcfs.append(min_DCF_binary(pi, loc_LVAL, llr)) 

        llrs.append(llr)
    plt.xscale('log', base=10)
    plt.grid(True, linestyle=':')
    plt.title("Non-prior-weighted LR (50 samples)")
    line = plt.plot(ls, dcfs, label="LR", alpha=0.5)
    c = line[0]._color
    plt.plot(ls, min_dcfs, linestyle="dashed", color=c, alpha=0.5)
    
    plt.ylim((0.2, 1.01))
    plt.xlim(min(ls), max(ls))
    plt.xlabel("lambda")
    plt.ylabel("DCF/minDCF")

    #plt.savefig("./project/plots/logistic_regression_minDCF_DCF_lambda_plots/50_samples_LR_DCF_minDCF_lambda_plot.png")
    #plt.savefig("./project/plots/logistic_regression_minDCF_DCF_lambda_plots/50_samples_LR_DCF_minDCF_lambda_plot_xval.png")

    #plt.savefig("./project/plots/logistic_regression_minDCF_DCF_lambda_plots/LR_DCF_minDCF_lambda_plot_xval.png")
    #plt.savefig("./project/plots/logistic_regression_minDCF_DCF_lambda_plots/LR_DCF_minDCF_lambda_plot.png")
    #plt.show()
    #exit()

    idx = np.argmin(min_dcfs)
    print(np.min(dcfs))
    print(f"{str("Non-prior-weighted LR"):<40} optimal values with l = {ls[idx]:.10f} -> minDCF = {min_dcfs[idx]:.4f} | DCF = {dcfs[idx]:.4f}")

    # prior-weighted logistic regression plot and minDCF minimum
    dcfs = []
    min_dcfs = []
    llrs = []
    for l in ls:
        ### decomment/comment this part to use/not use held-out validation set 
        #lr = LogisticRegression(DTR, LTR, label_dict, l, l_priors=np.log([pi, 1-pi]))
        #lr.fit()
        #predictions, llr = lr.predict(DVAL)
        #dcfs.append(empirical_bayes_risk_binary(pi, LVAL, llr))
        #min_dcfs.append(min_DCF_binary(pi, LVAL, llr))

        ### decomment/comment this part to use/not use kfold validation
        llr, predictions, loc_LVAL = kfold_scores_pooling(D, L, LogisticRegression, {"l":l, "label_dict":label_dict, "l_priors": np.log([pi, 1-pi])})
        dcfs.append(empirical_bayes_risk_binary(pi, loc_LVAL, llr))
        min_dcfs.append(min_DCF_binary(pi, loc_LVAL, llr))

        llrs.append(llr)
    plt.xscale('log', base=10)
    plt.grid(True, linestyle=':')
    plt.title("Prior-weighted LR")
    line = plt.plot(ls, dcfs, label="PWLR", alpha=0.5)
    c = line[0]._color
    plt.plot(ls, min_dcfs, linestyle="dashed", color=c, alpha=0.5)
    #plt.show()
    idx = np.argmin(min_dcfs)
    print(f"{str("Prior-weighted LR"):<40} optimal values with l = {ls[idx]:.10f} -> minDCF = {min_dcfs[idx]:.4f} | DCF = {dcfs[idx]:.4f}")

    # show plots of non-weighted and weighted LR DCF and minDCF as a function of lambda
    #plt.axis([np.min(ls), np.min(ls), .2, 1.2])
    plt.title("Prior-weighted LR / Not prior-weighted LR")
    plt.legend(loc='upper left')
    plt.ylim((0.2, 1.01))
    plt.xlim(min(ls), max(ls))
    plt.xlabel("lambda")
    plt.ylabel("DCF/minDCF")
    #plt.savefig(f"./project/plots/logistic_regression_minDCF_DCF_lambda_plots/PWLR_LR_DCF_minDCF_plot.png")
    #plt.savefig(f"./project/plots/logistic_regression_minDCF_DCF_lambda_plots/PWLR_LR_DCF_minDCF_plot_xval.png")
    plt.show()

    
    # non-prior-weighted logistic regression with preprocessing technique plot and minDCF minimum
    preproc_func = [
        (quadratic_feature_mapping, [], "Quadratic mapped LR"),
        (center_data, [], "Centered data LR"),
        #(L2_normalization, [], "L2-normalized LR"),
        (z_normalization, [], "Z-normalized LR"),
        (withening, [], "Withened LR")
    ]

    for prep_func, prep_args, desc in preproc_func:
        dcfs = []
        min_dcfs = []
        llrs = []
        for l in ls:
            ### decomment/comment this part to use/not use held-out validation set
            #ret = prep_func(DTR, *prep_args)
            #loc_DTR = ret[0]
            #loc_DVAL = prep_func(DVAL, *ret[1:])[0]
            #lr = LogisticRegression(loc_DTR, LTR, label_dict, l)
            #lr.fit()
            #predictions, llr = lr.predict(loc_DVAL)
            #dcfs.append(empirical_bayes_risk_binary(pi, LVAL, llr))
            #min_dcfs.append(min_DCF_binary(pi, LVAL, llr)) 

            ### decomment/comment this part to use/not use kfold validation
            llr, predictions, loc_LVAL = kfold_scores_pooling(D, L, LogisticRegression, {"l": l, "label_dict": label_dict}, preprocess_func=prep_func, preprocess_args=prep_args)
            dcfs.append(empirical_bayes_risk_binary(pi, loc_LVAL, llr))
            min_dcfs.append(min_DCF_binary(pi, loc_LVAL, llr))

            llrs.append(llr)
        plt.xscale('log', base=10)
        plt.grid(True, linestyle=':')
        plt.title(desc)
        
        line = plt.plot(ls, dcfs, label=f"{desc}", alpha=0.5)
        c = line[0]._color
        plt.plot(ls, min_dcfs, linestyle="dashed", color=c, alpha=0.5)
        #plt.show()
        idx = np.argmin(min_dcfs)
        print(idx)
        print(f"{desc:<40} optimal values with l = {ls[idx]:.10f} -> minDCF = {min_dcfs[idx]:.4f} | DCF = {dcfs[idx]:.4f}")
   
    # show plots of preprocessed non-weighted LR DCF and minDCF as a function of lambda
    plt.title("Preprocessed LR comparison")
    plt.legend(loc="upper left")
    plt.ylim((0.2, 1.01))
    plt.xlim(min(ls), max(ls))
    #plt.savefig(f"./project/plots/logistic_regression_minDCF_DCF_lambda_plots/preproc_LR_DCF_minDCF_plot.png")
    #plt.savefig(f"./project/plots/logistic_regression_minDCF_DCF_lambda_plots/preproc_LR_DCF_minDCF_plot_xval.png")
    plt.show()

    # non-prior-weighted logistic regression plot and minDCF minimum using PCA
    preproc_func = []
    for m in range(6,0,-1):
        preproc_func.append((
            pca_pipe,
            [L, m],
            f"PCA (m={m}) LR"
        ))
    
    for prep_func, prep_args, desc in preproc_func:
        dcfs = []
        min_dcfs = []
        llrs = []
        for l in ls:
            ### decomment/comment this part to use/not use held-out validation set
            #ret = prep_func(DTR, *prep_args)
            #loc_DTR = ret[0]
            #loc_DVAL = prep_func(DVAL, *ret[1:])[0]
            #lr = LogisticRegression(loc_DTR, LTR, label_dict, l)
            #lr.fit()
            #predictions, llr = lr.predict(loc_DVAL)
            #dcfs.append(empirical_bayes_risk_binary(pi, LVAL, llr))
            #min_dcfs.append(min_DCF_binary(pi, LVAL, llr)) 

            ### decomment/comment this part to use/not use kfold validation
            llr, predictions, loc_LVAL = kfold_scores_pooling(D, L, LogisticRegression, {"l": l, "label_dict": label_dict}, preprocess_func=prep_func, preprocess_args=prep_args)
            dcfs.append(empirical_bayes_risk_binary(pi, loc_LVAL, llr))
            min_dcfs.append(min_DCF_binary(pi, loc_LVAL, llr))

            llrs.append(llr)
        plt.xscale('log', base=10)
        plt.grid(True, linestyle=':')
        plt.title(desc)
        
        line = plt.plot(ls, dcfs, label=f"{desc}", alpha=0.5)
        c = line[0]._color
        plt.plot(ls, min_dcfs, linestyle="dashed", color=c, alpha=0.5)
        #plt.show()
        idx = np.argmin(min_dcfs)
        print(f"{desc:<40} optimal values with l = {ls[idx]:.10f} -> minDCF = {min_dcfs[idx]:.4f} | DCF = {dcfs[idx]:.4f}")

    # show plots of non-weighted LR DCF and minDCF as a function of lambda using PCA
    plt.title("PCA LR comparison")
    plt.legend(loc="upper left")
    plt.ylim((0.2, 1.01))
    plt.xlim(min(ls), max(ls))
    #plt.savefig(f"./project/plots/logistic_regression_minDCF_DCF_lambda_plots/PCA_LR_DCF_minDCF_plot.png")
    #plt.savefig(f"./project/plots/logistic_regression_minDCF_DCF_lambda_plots/PCA_LR_DCF_minDCF_plot_xval.png")
    plt.show()
     """
    """ 
    #ls[35] = np.float(0.013219411484660288)
    scores, predictions, KFLVAL = kfold_scores_pooling(D, L, LogisticRegression, {"label_dict": label_dict, "l": ls[35]}, preprocess_func=quadratic_feature_mapping)
    print(f"LogReg | QuadFeatMap | l = {ls[35]:.10f} | Error Rate = {error_rate(KFLVAL, predictions)*100:.2f}% | minDCF = {min_DCF_binary(pi, KFLVAL, scores):.4f} | actDCF = {empirical_bayes_risk_binary(pi, KFLVAL, scores):.4f}")
     """
    
    ######### LAB 9 #########
    ######### SUPPORT VECTOR MACHINE

    # optional part
    """
    trans_DTR = (DTR[-2:,:] * DTR[-2:,:]).sum(axis=0)
    print(trans_DTR.shape)
    plt.scatter(DTR[4,::10][LTR[::10]==0], DTR[5,::10][LTR[::10]==0], s=1, alpha=.5)
    plt.scatter(DTR[4,::10][LTR[::10]==1], DTR[5,::10][LTR[::10]==1], s=1, alpha=.5)

    plt.figure(figsize=[7,1])
    plt.ylim([.5,-.5])
    plt.grid(True, linestyle=':')
    plt.xticks(color='w') 
    plt.yticks(color='w')

    plt.scatter(trans_DTR[:][LTR[:]==0], [.1]*trans_DTR[:][LTR[:]==0].shape[0], s=1)
    plt.scatter(trans_DTR[:][LTR[:]==1], [-.1]*trans_DTR[:][LTR[:]==1].shape[0], s=1)
    plt.show()

    exit()  
    """

    Configuration = namedtuple("Configuration", ["tr_data", "tr_label", "val_data", "val_label", "preproc_f", "preproc_a", "model_a", "desc", "plot_title", "line_title"])

    confs = [
        Configuration(DTR[:,:], LTR[:], DVAL[:,:], LVAL[:], no_op,       [], {"xi":1},
                      "SVM_xi_1",               r"SVM - $\sqrt{K}$ = 1", r"SVM"),

        Configuration(DTR[:,:], LTR[:], DVAL[:,:], LVAL[:], center_data, [], {"xi":1}, 
                      "SVM_centered_data_xi_1", r"SVM - $\sqrt{K}$ = 1 - centered data", r"SVM centered"),

        Configuration(DTR[:,:], LTR[:], DVAL[:,:], LVAL[:], no_op,       [], {"kernel":"polynomial", "xi":0, "kernel_args":[2, 1]}, 
                      "SVM_poly_xi_0_d_2_c_1",  r"SVM - poly (d=2, c=1) - $\sqrt{K}$ = 0", r"SVM poly"),

        Configuration(DTR[:,:], LTR[:], DVAL[:,:], LVAL[:], no_op,       [], {"kernel":"radial_basis_function", "xi":1, "kernel_args":[np.exp(-4)]}, 
                      "SVM_RBF_xi_1_g_e^-4",    r"SVM - RBF ($\gamma=e^{{-4}}$) - $\sqrt{K}$ = 1", r"SVM RBF $\gamma = \exp{-4}$"),

        Configuration(DTR[:,:], LTR[:], DVAL[:,:], LVAL[:], no_op,       [], {"kernel":"radial_basis_function", "xi":1, "kernel_args":[np.exp(-3)]}, 
                      "SVM_RBF_xi_1_g_e^-3)",   r"SVM - RBF ($\gamma=e^{{-3}}$) - $\sqrt{K}$ = 1", r"SVM RBF $\gamma = \exp{-3}$"),

        Configuration(DTR[:,:], LTR[:], DVAL[:,:], LVAL[:], no_op,       [], {"kernel":"radial_basis_function", "xi":1, "kernel_args":[np.exp(-2)]}, 
                      "SVM_RBF_xi_1_g_e^-2)",   r"SVM - RBF ($\gamma=e^{{-2}}$) - $\sqrt{K}$ = 1", r"SVM RBF $\gamma = \exp{-2}$"),

        Configuration(DTR[:,:], LTR[:], DVAL[:,:], LVAL[:], no_op,       [], {"kernel":"radial_basis_function", "xi":1, "kernel_args":[np.exp(-1)]}, 
                      "SVM_RBF_xi_1_g_e^-1)",   r"SVM - RBF ($\gamma=e^{{-1}}$) - $\sqrt{K}$ = 1", r"SVM RBF $\gamma = \exp{-1}$"),
    ]

    # TODO: try using even parameters = [np.exp(-0.5), np.exp(-0.25)] (the configuration achieving best performance have higher value of gamma)

    plot_points = {
        "SVM_centered_data_xi_1": r"SVM / SVM centered_data ($\xi = 1.0$)",
        "SVM_poly_xi_0_d_2_c_1": r"SVM poly ($\xi = 1.0$)",
        "SVM_RBF_xi_1_g_e^-1)": r"SVM RBF ($\xi = 1.0$)"
    }

    """
    pi = 0.1
    cs = np.logspace(-5, 0, 11)
    results = {}
    plot_idx = 0

    #for conf in tqdm(confs):
    for conf in confs:
        act_dcfs = []
        min_dcfs = []
        break
    
        if "kernel" in conf.model_a.keys() and conf.model_a["kernel"] == "radial_basis_function":
            cs = np.logspace(-3, 2, 11)
        for c in cs:
            ret = conf.preproc_f(conf.tr_data, *conf.preproc_a)
            tr_data = ret[0]
            val_data = conf.preproc_f(conf.val_data, *ret[1:])[0]

            #svm = SupportVectorMachine(tr_data, conf.tr_label, label_dict, C=c, **conf.model_a)
            #svm.fit()
            # save the model
            #with open(f"./project/saved_models/{conf.desc}_c_{c:.5f}.pkl", "wb") as file:
            #    pickle.dump(svm, file)

            # load the model
            model_desc = f"{conf.desc}_c_{c:.5f}"
            with open(f"./project/saved_models/{model_desc}.pkl", "rb") as file:
                svm = pickle.load(file)

            predictions, scores = svm.predict(val_data)

            act_dcf = empirical_bayes_risk_binary(pi, conf.val_label, scores)
            act_dcfs.append(act_dcf)

            min_dcf = min_DCF_binary(pi, conf.val_label, scores)
            min_dcfs.append(min_dcf)
            
            results[model_desc] = {
                "model": svm,
                "model_desc": model_desc,
                "min_dcf": min_dcf,
                "act_dcf": act_dcf,
                "error_rate": error_rate(conf.val_label, predictions)
            }
        plt.grid(True, linestyle=':')
        plt.xscale('log', base=10)
        plt.xlim(min(cs), max(cs))
        plt.xlabel("C")
        plt.ylabel("DCF/minDCF")
        plt.ylim((-0.01, 1.01))

        line = plt.plot(cs, act_dcfs, label=conf.line_title, alpha=0.5)
        c = line[0]._color

        plt.plot(cs, min_dcfs, linestyle="dashed", color=c, alpha=0.5)

        #plt.title(conf.plot_title)
        #plt.show()

        if conf.desc in plot_points.keys():
            plt.title(plot_points[conf.desc])
            plt.legend(loc="upper right")
            #plt.savefig(f"./project/plots/svm_minDCF_DCF_C_plots/SVM_{plot_idx}.png")
            #plt.show()
            plot_idx+=1

    print("Result sorted based on minDCF")
    sorted_results = sorted(results, key=lambda k: results[k]["min_dcf"])
    for res_k in sorted_results:
        res_v = results[res_k]
        model = res_v["model"]
        print(f"Model: {res_v["model_desc"]:<33} | Error Rate = {res_v["error_rate"]*100:.2f}% | minDCF = {res_v["min_dcf"]:.4f} | actDCF = {res_v["act_dcf"]:.4f} | primal loss = {model.svm_primal_obj_binary(model.w):#.6e} | dual loss = {model.svm_dual_obj_binary(model.alpha)[0].item():#.6e} | duality gap ={model.get_duality_gap():#.6e}")

    print("Result sorted based on duality gap")
    sorted_results = sorted(results, key=lambda k: results[k]["model"].get_duality_gap())
    for res_k in sorted_results:
        res_v = results[res_k]
        model = res_v["model"]
        print(f"Model: {res_v["model_desc"]:<33} | Error Rate = {res_v["error_rate"]*100:.2f}% | minDCF = {res_v["min_dcf"]:.4f} | actDCF = {res_v["act_dcf"]:.4f} | primal loss = {model.svm_primal_obj_binary(model.w):#.6e} | dual loss = {model.svm_dual_obj_binary(model.alpha)[0].item():#.6e} | duality gap ={model.get_duality_gap():#.6e}")

    print("Best model")
    best_model_k = min(results, key=lambda k: results[k]["min_dcf"])
    best_model = results[best_model_k]["model"]
    print(f"Model: {results[best_model_k]["model_desc"]} | Error Rate = {results[best_model_k]["error_rate"]*100:.2f}% | minDCF = {results[best_model_k]["min_dcf"]:.4f} | actDCF = {results[best_model_k]["act_dcf"]:.4f} | primal loss = {best_model.svm_primal_obj_binary(best_model.w):#.6e} | dual loss = {best_model.svm_dual_obj_binary(best_model.alpha)[0].item():#.6e} | duality gap ={best_model.get_duality_gap():#.6e}")
    """
    # optional part:
    """ 
    cs = np.logspace(-5, 0, 11)
    for c in tqdm(cs[10:]):
        svm = SupportVectorMachine(DTR, LTR, label_dict, C=c, xi=0, kernel="polynomial", kernel_args=[4,1])
        svm.fit()
        with open(f"./project/saved_models/opt/SVM_poly_xi_0_d_4_c_1_c_{c:.5f}.pkl", "wb") as file:
            pickle.dump(svm, file)
    """
    """ 
    # plot decision boundaries of the chosen model
    with open(f"./project/saved_models/SVM_RBF_xi_1_g_e^-2)_c_31.62278.pkl", "rb") as file:
        svm = pickle.load(file)
    plot_svm_boundary_2d_binary(svm, DTR, LTR, dim_to_keep=[4,5], plot_title="SVM Decision Boundary (Last Two Features)")
    """
    ######### LAB 10 #########
    ######### GAUSSIAN MIXTURE MODEL
    """ 
    models = [
        MVGMModel,
        DiagGMModel,
        TiedGMModel
    ]

    ns = [1,2,4,8,16,32]
    #ns = [1]
    results = {}
    #pbar = tqdm(total=len(models)*len(ns)*len(ns))
    for n_T in ns:
        break
        for n_F in ns:
            #print(f"Components number = n_T = {n_T}, n_F = {n_F}")
            for model_cstr in models:
                # n = [false_num, true_num]
                n = [n_F, n_T]
                model = model_cstr(DTR, LTR, label_dict, l_priors=np.log([.9,.1]), n=n, d=1e-5)
                model.fit()
                predictions, scores = model.predict(DVAL)
                llr = scores[1,:] - scores[0,:]
                results[f"{model.get_model_name():<10}(n_T={n[1]:>2}, n_F={n[0]:>2})"] = {
                    "model": model,
                    "prior": .1,
                    "error_rate": error_rate(LVAL, predictions),
                    "min_dcf": min_DCF_binary(.1, LVAL, llr),
                    "act_dcf": empirical_bayes_risk_binary(.1, LVAL, llr),
                    "llr": llr
                }
                #print(f"Model components = gmm[1] = {len(model.gmm[1])}, gmm[0] = {len(model.gmm[0])}")
                #print(f"{model.get_model_name():<10} -> Error Rate = {error_rate(LVAL, predictions)*100:.2f}% | minDCF = {min_DCF_binary(.1, LVAL, llr):.4f} | actDCF = {empirical_bayes_risk_binary(.1, LVAL, llr):.4f}")
                #pbar.update(1)
    #pbar.close()
    #with open("./project/saved_models/GMM_results.pkl", "wb") as file:
    #    pickle.dump(results, file)
    with open("./project/saved_models/GMM_results.pkl", "rb") as file:
        results = pickle.load(file)
    sorted_k = sorted(results, key=lambda k: results[k]["min_dcf"])
    for k in sorted_k:
        res = results[k]
        print(f"{k:<26} -> Error Rate = {res["error_rate"]*100:>5.2f}% | minDCF = {res["min_dcf"]:.4f} | actDCF = {res["act_dcf"]:.4f}")
     """
    #plot_boundary_2d_binary(results[sorted_k[0]]["model"], DTR, LTR, dim_to_keep=[4,5], plot_title="GMM Decision Boundary (Last Two Features)")
    
    #for selected model, plot bayes error plots
    
    ######### LAB 11 #########
    ######### CALIBRATION AND FUSION

    """ 
    """ 
    # NBG
    nbgm = NaiveGModel(DTR, LTR, label_dict)
    nbgm.fit()
    # QFM
    ret = quadratic_feature_mapping(DTR)
    DTR_q = ret[0]
    DVAL_q = quadratic_feature_mapping(DVAL, *ret[1:])[0]
    lr_qfm = LogisticRegression(DTR_q, LTR, label_dict, l=0.013219411484660288)
    lr_qfm.fit()

    with open(f"./project/saved_models/SVM_RBF_xi_1_g_e^-2)_c_31.62278.pkl", "rb") as file:
        svm = pickle.load(file)
    with open("./project/saved_models/GMM_results.pkl", "rb") as file:
        gmm = pickle.load(file)["DGM model (n_T=16, n_F= 8)"]["model"]

    models = [
        (nbgm, (DVAL), "NBG[RAW]"),
        (lr_qfm, (DVAL_q), "LR[QFM]"),
        (svm, (DVAL), "SVM[RBF]"),
        (gmm, (DVAL), "DGM[RAW]"),
    ]
    models_res = [m[0].predict(m[1])[1] for m in models]
    calibs_res = []
    for model_res in models_res:
        calibs_res.append(
            llr_calibration(model_res, LVAL, label_dict)
        )
    #(l_scores, predictions, CLVAL), cal_lr
    c_models = [lr for _, lr in calibs_res]

    # plot DCF of no calib/calib models 
    for model, calib_res, model_res in zip(models, calibs_res, models_res):
        # no calib
        #bayes_error_plot_binary(LVAL, model_res, start=-4, stop=4, act_DCF_prefix=model[2], show=False, alpha=0.5, plot_title="Selected models Bayes Error plots")
        # calib
        bayes_error_plot_binary(calib_res[0][2], calib_res[0][0], start=-4, stop=4, act_DCF_prefix=model[2]+"(calib.)", show=False, alpha=0.5, plot_title="Selected models Bayes Error plots (calib.)")
    plt.show() 
    """

    # fuse no calib models 
    all_scores = np.vstack([scores for scores in models_res])
    (fus_scores, predictions, FLVAL), fus_lr = model_fusion(all_scores, LVAL, label_dict)
    bayes_error_plot_binary(FLVAL, fus_scores, start=-4, stop=4, act_DCF_prefix="fused models", plot_title="Fused models Bayes Error plots (no calib.)")
    (l_scores, predictions, CLVAL), lr = llr_calibration(fus_scores, FLVAL, label_dict)
    bayes_error_plot_binary(CLVAL, l_scores, start=-4, stop=4, act_DCF_prefix="fused models", plot_title="Fused models Bayes Error plots (calib. after fusion)")

    # get scores with DVAL original order, without shuffle
    all_cal_scores = np.vstack([
        c_m.get_scores(m[0].predict(m[1])[1]) for m, c_m in zip(models, c_models)
    ])
    # fuse calib models
    (fus_scores, predictions, FLVAL), fus_lr = model_fusion(all_cal_scores, LVAL, label_dict)
    bayes_error_plot_binary(FLVAL, fus_scores, start=-4, stop=4, act_DCF_prefix="fused models", plot_title="Fused models Bayes Error plots (calib. before fusion)")
     """
    # SELECTED MODEL:
    #   - GM : Naive Bayes / no PCA (without calibration) (prior=0.1)
    #   - LR : Non-prior-weighted LogReg Quadratic Feature Mapping (without calibration) using lambda=0.013219411484660288
    #   - SVM: RBF using xi=1, gamma=e^-2, C=31.62278 (name = SVM_RBF_xi_1_g_e^-2)_c_31.62278.pkl)
    #       - poly using xi=0 d=2 c=1 C=0.00003 ( this achieve a better duality gap, is it worth to consider? )
    #   - GMM: Diagonal covariance using components: 16(T) and 8(F) and default parameters

    # FINAL MODEL
    with open("./project/saved_models/GMM_results.pkl", "rb") as file:
        gmm = pickle.load(file)["DGM model (n_T=16, n_F= 8)"]["model"]
    predictions, scores = gmm.predict(DEVAL)
    print(f"DGM model (n_T=16, n_F= 8) -> Error Rate={error_rate(LEVAL, predictions)*100:.2f}% | minDCF={min_DCF_binary(.1, LEVAL, scores)} | actDCF={empirical_bayes_risk_binary(.1, LEVAL, scores)}")
    bayes_error_plot_binary(LEVAL, scores, start=-4, stop=4, plot_title="DGM Mixture Model Bayes Error (nT=16, nF=8) - EVAL SET")

    """ with open(f"./project/saved_models/SVM_RBF_xi_1_g_e^-2)_c_31.62278.pkl", "rb") as file:
        svm = pickle.load(file)
    predictions, scores = svm.predict(DEVAL)
    (scores, predictions, LEVAL), cal_lr = llr_calibration(scores, LEVAL, label_dict)
    print(f"SVM[RBF] (xi=1,g=e^-2,c=31.62278) -> Error Rate={error_rate(LEVAL, predictions)*100:.2f}% | minDCF={min_DCF_binary(.1, LEVAL, scores)} | actDCF={empirical_bayes_risk_binary(.1, LEVAL, scores)}")
    bayes_error_plot_binary(LEVAL, scores, start=-4, stop=4) """
    
    
    # model training:
    # - train using training data split
    # - calibrate using kfold on validation data
    # - evaluate (in the end for the choosen system) the goodness of the classificator
    
    # loop example using multiple models:
    # - instantiate model
    # - fit model
    # - retrieve prediction
    # - print error_rate (min_DCF/act_DCF?)
    # - save bayes error plots (ROC curve? other plots?)

    # TODO: where to save? as a csv? or save the entire model? save a ranking?
    # metrics and plot to save:
    # - parameter/model configuration
    # - accuracy/error rate
    # - actDCF/minDCF
    # - Bayes error plot

if __name__=="__main__":
    main()