
from modules.data.dataset import load, split_db_2to1
from modules.models.training import kfold_scores_pooling
from modules.visualization.plots import scatter_hist_per_feat, correlation_heatmap, gaussian_hist_plot, bayes_error_plot_binary
from modules.utils.operations import var, mean, cov, p_corr, trunc, effective_prior_binary
from modules.utils.metrics import error_rate, calculate_overlap, empirical_bayes_risk_binary, min_DCF_binary
from modules.features.dimensionality_reduction import lda, pca, pca_pipe, lda_pipe
from modules.features.transformation import L2_normalization, center_data, quadratic_feature_mapping, withening, z_normalization
from modules.models.mean_classifier import LdaBinaryClassifier
from modules.models.gaussians import MVGModel, TiedGModel, NaiveGModel, TiedNaiveGModel
from modules.models.logistic_regression import LogisticRegression

import matplotlib.pyplot as plt
import numpy as np

def main():
    # REMEMBER: the load function ensures that, for dataset labeled with 0 for false and 1 for true, the data is correctly labeled in the label_dictionary
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
    #scatter_hist_per_feat(D_pca_m6, L, label_dict, save_path="./project/plots/PCA_6_scatter_matrix.png")
    #scatter_hist_per_feat(D_pca_m6, L, label_dict)
    
    # compute differences in variance and mean wrt classes
    #print(np.abs(var(D_pca_m6[:,L==0])-var(D_pca_m6[:,L==1])))
    #print(np.abs(mean(D_pca_m6[:,L==0])-mean(D_pca_m6[:,L==1])))

    # plot scatter matrix of projected data onto LDA directions
    #W_lda_m1, D_lda_m1 = lda(D, L, m=1)
    #scatter_hist_per_feat(D_lda_m1, L, label_dict, save_path="./project/plots/LDA_1_scatter_matrix.png")
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
    """ 
    models = [
        MVGModel,
        TiedGModel,
        NaiveGModel,
        TiedNaiveGModel,
        LdaBinaryClassifier
    ]

    P_pca_m6, _, _ = pca(DTR, m=6)

    descs = ["no_pca"]
    tr_datas = [DTR]
    vl_datas = [DVAL]
    for m in range(6,0,-1):
        descs.append(f"pca m={m}")
        tr_datas.append(np.dot(P_pca_m6[:, :m].T, DTR))
        vl_datas.append(np.dot(P_pca_m6[:, :m].T, DVAL))
    
    descs.append("features 1-2")
    tr_datas.append(DTR[[0,1], :])
    vl_datas.append(DVAL[[0,1], :])

    descs.append("features 3-4")
    tr_datas.append(DTR[[2,3], :])
    vl_datas.append(DVAL[[2,3], :])
        
    descs.append("features 1-4")
    tr_datas.append(DTR[:4, :])
    vl_datas.append(DVAL[:4, :])

    results = {}

    for model_cstr in models:
        print()
        for desc, tr_data, vl_data in zip(descs, tr_datas, vl_datas):
            print(desc)

            model = model_cstr(tr_data, LTR, label_dict)
            model.fit()
            predictions, scores = model.predict(vl_data)

            results[(m, model.get_model_name())] = error_rate(LVAL, predictions)
            print(f"{model.get_model_name()} |{"".join([" "+str(k)+": "+str(v)+" |" for k, v in model.get_model_params().items()])} error_rate = {results[(m, model.get_model_name())]*100:.2f}%")
            #bayes_error_plot_binary(LVAL, scores, plot_title=f"{model.get_model_name()} / {desc}")
    
    #print(f"Best result = {list(results.keys())[np.argmin(list(results.values()))]}")
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
    # try this models
    models = [
        MVGModel,
        NaiveGModel,
        TiedGModel,
        TiedNaiveGModel
    ]

    P_pca_m6, _, _ = pca(DTR, m=6)

    # try using this preprocessing techniques
    tr_datas = [DTR]
    vl_datas = [DVAL]
    descs = ["no PCA"]
    for m in range(6,0,-1):
        descs.append(f"PCA {m}")
        tr_datas.append(np.dot(P_pca_m6[:,:m].T, DTR))
        vl_datas.append(np.dot(P_pca_m6[:,:m].T, DVAL))


    # try this working points
    parameters = []
    parameters.append((0.5,1.0,1.0))
    parameters.append((0.9,1.0,1.0))
    parameters.append((0.1,1.0,1.0))
    parameters.append((0.5,1.0,9.0))
    parameters.append((0.5,9.0,1.0))

    # calculate effective prior of working points
    print("\nEffective priors")
    for p in parameters:
        print(f"prior={p[0]} | C_fn={p[1]} | C_fp={p[2]} => eff prior={effective_prior_binary(p[0], p[1], p[2])}")
    
    # remove last two, since same as effective prior to other already considered
    parameters.remove((0.5,1.0,9.0))
    parameters.remove((0.5,9.0,1.0))
    
    # get results from classifications
    min_dcfs = {}
    dcfs = {}
    miscal_loss = {}
    llrs = []

    app_results = {}

    # for each model
    for model_cstr in models:
        print()
        to_print = ""

        # for each preprocessing configuration
        for tr_data, vl_data, desc in zip(tr_datas, vl_datas, descs):
            first_it = True

            # for each working point
            for p in parameters:
                print("\n\n")
                print(p)

                #print(f"prior={p[0]} | C_fn={p[1]} | C_fp={p[2]}")

                model = model_cstr(tr_data, LTR, label_dict, l_priors=np.log([1-p[0], p[0]]), cost_matrix=np.array([
                    [0, p[1]],
                    [p[2], 0]
                ]))
                model.fit()
                predictions, scores = model.predict(vl_data)

                llrs.append(scores)
                dcf = empirical_bayes_risk_binary(prior=p[0], L=LVAL, llr=scores, cost_matrix=np.array([
                    [0, p[1]],
                    [p[2], 0]
                ]))

                min_dcf = min_DCF_binary(prior=p[0], L=LVAL, llr=scores, cost_matrix=np.array([
                    [0, p[1]],
                    [p[2], 0]
                ]))

                min_dcfs[f"{model.get_model_name():<40} | {desc:<6} | {p} "] = min_dcf

                dcfs[f"{model.get_model_name():<40} | {desc:<6} | {p} "] = dcf

                miscal_loss[f"{model.get_model_name():<40} | {desc:<6} | {p} "] = dcf - min_dcf

                # used to compare models in the context of a specific application
                if p not in app_results.keys():
                    app_results[p] = {}
                app_results[p][f"{model.get_model_name():<40} | {desc:<6}"] = {
                    "dcf": dcf,
                    "min_dcf": min_dcf,
                    "miscal_loss": dcf - min_dcf
                }

                # for each combination of model / preprocessing, save/show bayes error plot
                if first_it:
                    #bayes_error_plot_binary(LVAL, scores, plot_title=f"{model.get_model_name()} / {desc}", start=-4, stop=4, save_path=f"./project/plots/bayes_error_plot_{model.get_model_name()}_{"_".join(desc.split())}.png", show=False)
                    bayes_error_plot_binary(LVAL, scores, plot_title=f"{model.get_model_name()} / {desc}", start=-4, stop=4, show=False)
                    
                    # save with the application lines (make sense just using effective prior)
                    
                    #for p in parameters:
                    #    plt.axvline(x = np.log(p[0]/(1-p[0])), color = 'r', alpha=0.4) 
                    #plt.savefig(f"./project/plots/bayes_error_plot_{model.get_model_name()}_{"_".join(desc.split())}_with_apps_line.png")
                    
                    #plt.savefig(f"./project/plots/bayes_error_plot_{model.get_model_name()}_{"_".join(desc.split())}.png")
                    #plt.show()
                    plt.clf()
                    first_it = False

                # print classification results
                print(f"{model.get_model_name()} | {"".join([f"{str(k)}: {v:.1f} | " for k, v in model.get_model_params().items()])} {desc:<6} | error_rate = {error_rate(LVAL, predictions)*100:05.2f}% | DCF = {dcf:.4f} | minDCF = {min_dcf:.4f} | miscalibration loss = {dcf-min_dcf:.4f}")

                # string used to compile tables
                to_print = " & ".join([to_print, f"{dcf:.4f} {min_dcf:.4f} {(dcf-min_dcf):.4f}"])
            #print(to_print)
    print()
    
    # order by min_dcf
    min_dcfs = dict(sorted(min_dcfs.items(), key=lambda item: item[1]))
    print()
    for k, v in min_dcfs.items():
        print(f"{k} -> minDCF = {v:.4f}")

    # order by dcf
    dcfs = dict(sorted(dcfs.items(), key=lambda item: item[1]))
    print()
    for k, v in dcfs.items():
        print(f"{k} -> DCF = {v:.4f}")

    # order by miscalibration loss
    miscal_loss = dict(sorted(miscal_loss.items(), key=lambda item: item[1]))
    print()
    for k, v in miscal_loss.items():
        print(f"{k} -> miscalibration loss = {v:.4f}")

    # order, for each application, by min_dcf, dcf or miscalibration loss
    for app_desc, app_dict in app_results.items():
        print(app_desc)
        app_dict = dict(sorted(app_dict.items(), key=lambda item: item[1]["min_dcf"]))
        #app_dict = dict(sorted(app_dict.items(), key=lambda item: item[1]["dcf"]))
        #app_dict = dict(sorted(app_dict.items(), key=lambda item: item[1]["miscal_loss"]))
        sum = 0
        for k, v in app_dict.items():
            print(f"{k} -> DCF = {v['dcf']:.4f} | minDCF = {v['min_dcf']:.4f} | miscal. loss = {v['miscal_loss']:.4f}")
            #print(f"{k} -> miscal. loss = {v['miscal_loss']:.4f}")
            sum += v["min_dcf"]
        print(f"{k} -> avg min_DCF = {sum/len(app_dict)}")

 """
    # chosen configuration
    """ final_config = [
        (MVGModel, DTR, DVAL, "no PCA"),
        (NaiveGModel, DTR, DVAL, "no PCA"),
        (TiedGModel, np.dot(P_pca_m6[:,:4].T, DTR), np.dot(P_pca_m6[:,:4].T, DVAL), "PCA 4"),
        (TiedNaiveGModel, np.dot(P_pca_m6[:,:2].T, DTR), np.dot(P_pca_m6[:,:2].T, DVAL), "PCA 2")
    ] """

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
    
    #ls[35] = np.float(0.013219411484660288)
    scores, predictions, KFLVAL = kfold_scores_pooling(D, L, LogisticRegression, {"label_dict": label_dict, "l": ls[35]}, preprocess_func=quadratic_feature_mapping)
    print(f"LogReg | QuadFeatMap | l = {ls[35]:.10f} | Error Rate = {error_rate(KFLVAL, predictions)*100:.2f}% | minDCF = {min_DCF_binary(pi, KFLVAL, scores):.4f} | actDCF = {empirical_bayes_risk_binary(pi, KFLVAL, scores):.4f}")
    

    # SELECTED MODEL:
    #   - Naive Bayes / no PCA (without calibration) (prior=0.1)
    #   - Non-prior-weighted LogReg Quadratic Feature Mapping (without calibration) using lambda=0.013219411484660288
    
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