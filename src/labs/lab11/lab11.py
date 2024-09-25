import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.models.training import kfold_scores_pooling, llr_calibration, model_fusion
from modules.models.gaussians import MVGModel, TiedGModel
from modules.models.support_vector_machine import SupportVectorMachine
from modules.data.dataset import load, split_db_2to1

from modules.utils.metrics import empirical_bayes_risk, empirical_bayes_risk_binary, min_DCF_binary
from modules.visualization.plots import bayes_error_plot_binary

import numpy as np
import matplotlib.pyplot as plt

# ignore overflow warning on exp (G = -Z / (1 + np.exp(Z * S, dtype=np.longdouble)))
np.seterr(over='ignore')

# CALIBRATION (as written in the slides):
# 1. use the normal validation set, with static split, as evaluator for performance
# 2. use k-fold scores pooling to calibrate scores using the previously extracted validation set

def initial_tests(D, L):
    # evidence that kfold change a lot the value of the metrics we compute
    # actDCF kfold valida = 0.4
    # actDCF normal split = 0.8333333 (using the same split as kfold, same data ordering)

    # debug idx
    #l_scores, predictions, RLVAL, idx = kfold_scores_pooling(D, L, MVGModel, {"label_dict": label_dict})
    l_scores, predictions, RLVAL = kfold_scores_pooling(D, L, MVGModel, {"label_dict": label_dict})
    print(empirical_bayes_risk(RLVAL, predictions, label_dict, [.5, .5]))

    """ D = D[:, idx]
    L = L[idx] """
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    mvg = MVGModel(DTR, LTR, label_dict)
    mvg.fit()
    mvg_predictions, mvg_scores = mvg.predict(DVAL)
    print(empirical_bayes_risk(LVAL, mvg_predictions, label_dict, [.5, .5]))

    #print(np.all(RLVAL[-15:] == LVAL))

    (val_l_scores, val_predictions, val_LVAL), calibration_model = llr_calibration(mvg_scores, LVAL, label_dict)
    print(empirical_bayes_risk(val_LVAL, val_predictions, label_dict, np.exp(calibration_model.l_priors)))

    # BEST POLYNOMIAL FROM LAB09
    # K = xi^(1/2) =  1 | C =    1 | kernel = polynomial ([2, 1]) | dual loss = 3.569987e+00 | error rate =  2.9% | min DCF = 0.0556 | actDCF = 0.0556
    svm = SupportVectorMachine(DTR, LTR, label_dict, C=1, xi=1, kernel_args=[2, 1])
    svm.fit()
    svm_predictions, svm_scores = svm.predict(DVAL)

    stacked_scores = np.vstack((mvg_scores, svm_scores))
    (fusion_val_scores, fusion_val_predictions, fusion_val_LVAL), fusion_model = model_fusion(stacked_scores, LVAL, label_dict)
    print(empirical_bayes_risk(fusion_val_LVAL, fusion_val_predictions, label_dict, np.exp(fusion_model.l_priors)))

def print_minDCF_actDCF_bayes_plot(s_1, s_2, L,):
    # print minDCF and actDCF for both system
    print(f"s_1:\tminDCF = {min_DCF_binary(prior=.2, L=L, llr=s_1):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=L, llr=s_1):.3f}")
    print(f"s_2:\tminDCF = {min_DCF_binary(prior=.2, L=L, llr=s_2):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=L, llr=s_2):.3f}")

    # plot bayes error plots for both systems
    bayes_error_plot_binary(L=L, llr=s_1)
    bayes_error_plot_binary(L=L, llr=s_2)
    plt.show()

if __name__=="__main__":
    D, L, label_dict = load("labs/data/iris.csv")
    D = D[:, L!=0]-1
    L = L[L!=0]-1
    label_dict = {k:(v-1) for (k,v) in label_dict.items() if v!= 0}

    #initial_tests(D, L)

    s_1 = np.load("labs/lab11/Data/scores_1.npy")
    s_2 = np.load("labs/lab11/Data/scores_2.npy")
    L = np.load("labs/lab11/Data/labels.npy")

    e_1 = np.load("labs/lab11/Data/eval_scores_1.npy")
    e_2 = np.load("labs/lab11/Data/eval_scores_2.npy")
    EL = np.load("labs/lab11/Data/eval_labels.npy")

    #print_minDCF_actDCF_bayes_plot()

    # CALIBRATION - SINGLE FOLD APPROACH
    print("single-fold approach")

    # calibration and calibration validation sets
    SCAL1, SVAL1 = s_1[::3], np.hstack([s_1[1::3], s_1[2::3]])
    SCAL2, SVAL2 = s_2[::3], np.hstack([s_2[1::3], s_2[2::3]])
    LCAL, LVAL = L[::3], np.hstack([L[1::3], L[2::3]])

    """ bayes_error_plot_binary(L=LVAL, llr=SVAL1)
    bayes_error_plot_binary(L=LVAL, llr=SVAL2)
    plt.show() """

    print()

    # calibrate using SCAL scores and assess the quality using SVAL scores
    label_dict = {"0":0, "1":1}
    (SVAL1C, PVAL1, _), calibration_model_1 = llr_calibration(llr=SCAL1, L=LCAL, label_dict=label_dict, llr_val=SVAL1, LVAL=LVAL, l_priors=np.log([1-.2,.2]))

    print(f"raw 1:\tminDCF = {min_DCF_binary(prior=.2, L=LVAL, llr=SVAL1):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=LVAL, llr=SVAL1):.3f}")
    print(f"cal 1:\tminDCF = {min_DCF_binary(prior=.2, L=LVAL, llr=SVAL1C):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=LVAL, llr=SVAL1C):.3f}")

    (SVAL2C, PVAL2, _), calibration_model_2 = llr_calibration(llr=SCAL2, L=LCAL, label_dict=label_dict, llr_val=SVAL2, LVAL=LVAL, l_priors=np.log([1-.2,.2]))
    print(f"raw 2:\tminDCF = {min_DCF_binary(prior=.2, L=LVAL, llr=SVAL2):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=LVAL, llr=SVAL2):.3f}")
    print(f"cal 2:\tminDCF = {min_DCF_binary(prior=.2, L=LVAL, llr=SVAL2C):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=LVAL, llr=SVAL2C):.3f}")

    # evaluate the calibration on the evaluation set
    print()

    e_1c_1f = calibration_model_1.get_scores(e_1)
    e_2c_1f = calibration_model_2.get_scores(e_2)

    print(f"raw 1:\tminDCF = {min_DCF_binary(prior=.2, L=EL, llr=e_1):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=EL, llr=e_1):.3f}")
    print(f"cal 1:\tminDCF = {min_DCF_binary(prior=.2, L=EL, llr=e_1c_1f):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=EL, llr=e_1c_1f):.3f}")

    print(f"raw 2:\tminDCF = {min_DCF_binary(prior=.2, L=EL, llr=e_2):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=EL, llr=e_2):.3f}")
    print(f"cal 2:\tminDCF = {min_DCF_binary(prior=.2, L=EL, llr=e_2c_1f):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=EL, llr=e_2c_1f):.3f}")

    # calibration evidence on calibrationo validation set
    bayes_error_plot_binary(L=LVAL, llr=SVAL1, plot_title="System 1 - calibration validation set", act_DCF_prefix="(pre-cal)")
    bayes_error_plot_binary(L=LVAL, llr=SVAL1C, plot_min_DCF=False, act_DCF_prefix="(cal)")
    #plt.show()

    bayes_error_plot_binary(L=LVAL, llr=SVAL2, plot_title="System 2 - calibration validation set", act_DCF_prefix="(pre-cal)")
    bayes_error_plot_binary(L=LVAL, llr=SVAL2C, plot_min_DCF=False, act_DCF_prefix="(cal)")
    #plt.show()

    # calibration evidence on evaluation set
    bayes_error_plot_binary(L=EL, llr=e_1, plot_title="System 1 - evaluation set", act_DCF_prefix="(pre-cal)")
    bayes_error_plot_binary(L=EL, llr=e_1c_1f, plot_min_DCF=False, act_DCF_prefix="(cal)")
    #plt.show()

    bayes_error_plot_binary(L=EL, llr=e_2, plot_title="System 2 - evaluation set", act_DCF_prefix="(pre-cal)")
    bayes_error_plot_binary(L=EL, llr=e_2c_1f, plot_min_DCF=False, act_DCF_prefix="(cal)")
    #plt.show()


    # K-fold approach to calibration

    print("K-fold approach")

    k = 5
    (s_1c, p_1, l_1c), calibration_model_1 = llr_calibration(llr=s_1, L=L, label_dict=label_dict, k=k, l_priors=np.log([1-.2,.2]))
    (s_2c, p_2, l_2c), calibration_model_2 = llr_calibration(llr=s_2, L=L, label_dict=label_dict, k=k, l_priors=np.log([1-.2,.2]))

    print(f"raw 1:\tminDCF = {min_DCF_binary(prior=.2, L=L, llr=s_1):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=L, llr=s_1):.3f}")
    print(f"cal 1:\tminDCF = {min_DCF_binary(prior=.2, L=l_1c, llr=s_1c):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=l_1c, llr=s_1c):.3f}")

    print(f"raw 2:\tminDCF = {min_DCF_binary(prior=.2, L=L, llr=s_2):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=L, llr=s_2):.3f}")
    print(f"cal 2:\tminDCF = {min_DCF_binary(prior=.2, L=l_2c, llr=s_2c):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=l_2c, llr=s_2c):.3f}")

    print()
    e_1c = calibration_model_1.get_scores(e_1)
    e_2c = calibration_model_2.get_scores(e_2)

    print(f"raw 1:\tminDCF = {min_DCF_binary(prior=.2, L=EL, llr=e_1):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=EL, llr=e_1):.3f}")
    print(f"cal 1:\tminDCF = {min_DCF_binary(prior=.2, L=EL, llr=e_1c):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=EL, llr=e_1c):.3f}")

    print(f"raw 2:\tminDCF = {min_DCF_binary(prior=.2, L=EL, llr=e_2):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=EL, llr=e_2):.3f}")
    print(f"cal 2:\tminDCF = {min_DCF_binary(prior=.2, L=EL, llr=e_2c):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=EL, llr=e_2c):.3f}")

    # calibration evidence on calibrationo validation set
    bayes_error_plot_binary(L=L, llr=s_1, plot_title="System 1 - calibration validation set", act_DCF_prefix="(pre-cal)")
    bayes_error_plot_binary(L=l_1c, llr=s_1c, plot_min_DCF=False, act_DCF_prefix="(cal)")
    #plt.show()

    bayes_error_plot_binary(L=L, llr=s_2, plot_title="System 2 - calibration validation set", act_DCF_prefix="(pre-cal)")
    bayes_error_plot_binary(L=l_2c, llr=s_2c, plot_min_DCF=False, act_DCF_prefix="(cal)")
    #plt.show()

    """ # calibration evidence on evaluation set
    bayes_error_plot_binary(L=EL, llr=e_1, plot_title="System 1 - evaluation set", act_DCF_prefix="(pre-cal)")
    bayes_error_plot_binary(L=EL, llr=e_1c, plot_min_DCF=False, act_DCF_prefix="(cal)")
    plt.show()

    bayes_error_plot_binary(L=EL, llr=e_2, plot_title="System 2 - evaluation set", act_DCF_prefix="(pre-cal)")
    bayes_error_plot_binary(L=EL, llr=e_2c, plot_min_DCF=False, act_DCF_prefix="(cal)")
    plt.show() """

    # SCORE FUSION

    # SINGLE FOLD APPROACH
    s_12 = np.vstack([s_1,s_2])
    e_12 = np.vstack([e_1, e_2])

    SFUS, SVAL = s_12[:,::3], np.hstack([s_12[:,1::3], s_12[:,2::3]])
    LFUS, LVAL = L[::3], np.hstack([L[1::3], L[2::3]])

    (f_scores, f_predictions, _), fusion_model = model_fusion(SFUS, LFUS, label_dict, SVAL, LVAL, l_priors=np.log([1-.2,.2]))
    
    print("SINGLE FOLD FUSION VALIDATION SET")
    print(f"cal 1:\tminDCF = {min_DCF_binary(prior=.2, L=LVAL, llr=SVAL1C):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=LVAL, llr=SVAL1C):.3f}")
    print(f"cal 2:\tminDCF = {min_DCF_binary(prior=.2, L=LVAL, llr=SVAL2C):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=LVAL, llr=SVAL2C):.3f}")
    print(f"fusion:\tminDCF = {min_DCF_binary(prior=.2, L=LVAL, llr=f_scores):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=LVAL, llr=f_scores):.3f}")

    plt.clf()
    bayes_error_plot_binary(L=LVAL, llr=SVAL1C, plot_title="Validation Set - single fold approach", act_DCF_prefix="S1 -", min_DCF_prefix="S1 -")
    bayes_error_plot_binary(L=LVAL, llr=SVAL2C, plot_title="Validation Set - single fold approach", act_DCF_prefix="S2 -", min_DCF_prefix="S2 -")
    bayes_error_plot_binary(L=LVAL, llr=f_scores, plot_title="Validation Set - single fold approach", act_DCF_prefix="FUSION -", min_DCF_prefix="FUSION -")
    plt.show()

    f_predictions, f_scores = fusion_model.predict(e_12)

    print("SINGLE FOLD FUSION EVALUATION SET")
    print(f"cal 1:\tminDCF = {min_DCF_binary(prior=.2, L=EL, llr=e_1c_1f):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=EL, llr=e_1c_1f):.3f}")
    print(f"cal 2:\tminDCF = {min_DCF_binary(prior=.2, L=EL, llr=e_2c_1f):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=EL, llr=e_2c_1f):.3f}")
    print(f"fusion:\tminDCF = {min_DCF_binary(prior=.2, L=EL, llr=f_scores):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=EL, llr=f_scores):.3f}")

    bayes_error_plot_binary(L=EL, llr=e_1c_1f, plot_title="Evaluation Set - single fold approach", act_DCF_prefix="S1 -", min_DCF_prefix="S1 -")
    bayes_error_plot_binary(L=EL, llr=e_2c_1f, plot_title="Evaluation Set - single fold approach", act_DCF_prefix="S2 -", min_DCF_prefix="S2 -")
    bayes_error_plot_binary(L=EL, llr=f_scores, plot_title="Evaluation Set - single fold approach", act_DCF_prefix="FUSION -", min_DCF_prefix="FUSION -")
    plt.show()


    # K FOLD APPROACH
    
    k=5
    (f_scores, f_predictions, f_L), fusion_model = model_fusion(s_12, L, label_dict, k=k, l_priors=np.log([1-.2,.2]))

    print(f"{k} FOLD FUSION VALIDATION SET")
    print(f"cal 1:\tminDCF = {min_DCF_binary(prior=.2, L=l_1c, llr=s_1c):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=l_1c, llr=s_1c):.3f}")
    print(f"cal 2:\tminDCF = {min_DCF_binary(prior=.2, L=l_2c, llr=s_2c):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=l_2c, llr=s_2c):.3f}")
    print(f"fusion:\tminDCF = {min_DCF_binary(prior=.2, L=f_L, llr=f_scores):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=f_L, llr=f_scores):.3f}")
    
    plt.clf()
    bayes_error_plot_binary(L=l_1c, llr=s_1c, plot_title=f"Validation Set - k fold approach (k={k})", act_DCF_prefix="S1 -", min_DCF_prefix="S1 -")
    bayes_error_plot_binary(L=l_2c, llr=s_2c, plot_title=f"Validation Set - k fold approach (k={k})", act_DCF_prefix="S2 -", min_DCF_prefix="S2 -")
    bayes_error_plot_binary(L=f_L, llr=f_scores, plot_title=f"Validation Set - k fold approach (k={k})", act_DCF_prefix="FUSION -", min_DCF_prefix="FUSION -")
    plt.show()

    f_predictions, f_scores = fusion_model.predict(e_12)

    print(f"{k} FOLD FUSION VALIDATION SET")
    print(f"cal 1:\tminDCF = {min_DCF_binary(prior=.2, L=EL, llr=e_1c):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=EL, llr=e_1c):.3f}")
    print(f"cal 2:\tminDCF = {min_DCF_binary(prior=.2, L=EL, llr=e_2c):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=EL, llr=e_2c):.3f}")
    print(f"fusion:\tminDCF = {min_DCF_binary(prior=.2, L=EL, llr=f_scores):.3f} | actDCF = {empirical_bayes_risk_binary(prior=.2, L=EL, llr=f_scores):.3f}")

    bayes_error_plot_binary(L=EL, llr=e_1c_1f, plot_title=f"Evaluation Set - k fold approach (k={k})", act_DCF_prefix="S1 -", min_DCF_prefix="S1 -")
    bayes_error_plot_binary(L=EL, llr=e_2c_1f, plot_title=f"Evaluation Set - k fold approach (k={k})", act_DCF_prefix="S2 -", min_DCF_prefix="S2 -")
    bayes_error_plot_binary(L=EL, llr=f_scores, plot_title=f"Evaluation Set - k fold approach (k={k})", act_DCF_prefix="FUSION -", min_DCF_prefix="FUSION -")
    plt.show()