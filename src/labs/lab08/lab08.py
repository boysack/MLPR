import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.data.dataset import load, split_db_2to1
from modules.utils.metrics import error_rate, min_DCF_binary, empirical_bayes_risk_binary, empirical_bayes_risk, conf_matrix
from modules.models.logistic_regression import LogisticRegression

import numpy as np

if __name__ == "__main__":
    D, L, label_dict = load("labs/data/iris.csv")
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2) return D, L
    label_dict = {k:v for k,v in label_dict.items() if v != 0}
    label_dict["Iris-virginica"] = 0
    label_dict = dict(sorted(label_dict.items(), key=lambda item: item[1], reverse=False))

    #D, L, label_dict = load("project/data/trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    from modules.features.dimensionality_reduction import pca
    #P, V, DTR = pca(DTR, m=6)
    #DVAL = np.dot(P.T, DVAL)

    ls = [10**-3, 10**-1, 1]
    print()
    for l in ls:
        print(f"lambda = {l}")
        # don't specify any prior, i.e. the prior applied in training will be the same as the empirical one
        lr = LogisticRegression(DTR, LTR, l, label_dict)
        final_loss = lr.fit()
        predictions, l_scores = lr.predict(DVAL)

        # prof: use prediction of the model used using the threshold calculated with emp_priors, BUT use a different prior to
        # evaluate empirical bayes risk (WHY?)
        
        act_DCF_b = empirical_bayes_risk_binary(prior=0.5, L=LVAL, llr=l_scores)
        min_DCF, t = min_DCF_binary(prior=0.5, L=LVAL, llr=l_scores, return_threshold=True)
        print(f"final_loss = {final_loss} | error_rate = {error_rate(LVAL, predictions)*100:0.1f}% | minDCF = {min_DCF:0.4f} | actDCF = {act_DCF_b:0.4f}", end="\n")

    print()
    for l in ls:
        print(f"lambda = {l}")
        # specify prior, i.e. the prior applied in training will be the one specified
        lr = LogisticRegression(DTR, LTR, l, label_dict, l_priors=np.log([0.2, 0.8]))
        final_loss = lr.fit()
        predictions, l_scores = lr.predict(DVAL)
        
        act_DCF_b = empirical_bayes_risk_binary(prior=0.8, L=LVAL, llr=l_scores)
        min_DCF, t = min_DCF_binary(prior=0.8, L=LVAL, llr=l_scores, return_threshold=True)
        print(f"final_loss = {final_loss} | error_rate = {error_rate(LVAL, predictions)*100:0.1f}% | minDCF = {min_DCF:0.4f} | actDCF = {act_DCF_b:0.4f}", end="\n")

    