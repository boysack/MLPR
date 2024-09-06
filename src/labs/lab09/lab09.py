import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.data.dataset import load, split_db_2to1
from modules.models.support_vector_machine import SupportVectorMachine
from modules.utils.metrics import error_rate, empirical_bayes_risk_binary, min_DCF_binary

from modules.utils.operations import row, col

import numpy as np

if __name__=="__main__":
    D, L, label_dict = load("labs/data/iris.csv")
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2) return D, L
    label_dict = {k:v for k,v in label_dict.items() if v != 0}
    label_dict["Iris-virginica"] = 0
    label_dict = dict(sorted(label_dict.items(), key=lambda item: item[1], reverse=False))

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    K = [1, 10]
    C = [0.1, 1.0, 10.0]

    """ for k in K:
        for c in C:
            svm = SupportVectorMachine(DTR, LTR, label_dict, C=c, xi=k**2)
            svm.fit()
            predictions, scores = svm.predict(DVAL)
            err = error_rate(LVAL, predictions)
            act_DCF = empirical_bayes_risk_binary(prior=.5, L=LVAL, predictions=predictions)
            min_DCF = min_DCF_binary(prior=.5, L=LVAL, llr=scores)
            
            print(f"K = xi^(1/2) = {k: >2} | C = {c: >4} | primal loss = {svm.svm_primal_obj_binary(svm.w):#.6e} | dual loss = {-svm.svm_dual_obj_binary(svm.alpha)[0].item():#.6e} | duality gap = {svm.get_duality_gap():#.6e} | error rate = {err*100: >4.1f}% | min DCF = {min_DCF:0.4f} | actDCF = {act_DCF:0.4f}")
    """
    print()
    kernel = "polynomial"
    kernel_args = [[2,0], [2,1]]

    K = [0, 1]
    C = [1]

    np.random.seed(5)

    for i, ka in enumerate(kernel_args):
        for c in C:
            for k in K:
                svm = SupportVectorMachine(DTR, LTR, label_dict, C=c, xi=k**2, kernel=kernel, kernel_args=list(ka))
                loss = svm.fit()
            
                predictions, scores = svm.predict(DVAL)
                err = error_rate(LVAL, predictions)
                act_DCF = empirical_bayes_risk_binary(prior=.5, L=LVAL, predictions=predictions)
                min_DCF = min_DCF_binary(prior=.5, L=LVAL, llr=scores)
                
                print(f"K = xi^(1/2) = {k: >2} | C = {c: >4} | kernel = {svm.kernel: >3} ({svm.kernel_args[:-1]}) | dual loss = {loss['dual_loss']:#.6e} | error rate = {err*100: >4.1f}% | min DCF = {min_DCF:0.4f} | actDCF = {act_DCF:0.4f}")
                del svm
                #print(svm.alpha)
                #print(svm.w)

    kernel = "radial_basis_function"
    kernel_args = [[1.0], [10.0]]

    for i, ka in enumerate(kernel_args):
        for c in C:
            for k in K:
                svm = SupportVectorMachine(DTR, LTR, label_dict, C=c, xi=k**2, kernel=kernel, kernel_args=list(ka))
                loss = svm.fit()
            
                predictions, scores = svm.predict(DVAL)
                err = error_rate(LVAL, predictions)
                act_DCF = empirical_bayes_risk_binary(prior=.5, L=LVAL, predictions=predictions)
                min_DCF = min_DCF_binary(prior=.5, L=LVAL, llr=scores)
                
                print(f"K = xi^(1/2) = {k: >2} | C = {c: >4} | kernel = {svm.kernel: >3} ({svm.kernel_args[:-1]}) | dual loss = {loss['dual_loss']:#.6e} | error rate = {err*100: >4.1f}% | min DCF = {min_DCF:0.4f} | actDCF = {act_DCF:0.4f}")
                del svm
                #print(svm.alpha)
                #print(svm.w)


    # 10 min each
    """ D, L, label_dict = load("project/data/trainData.txt")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    svm = SupportVectorMachine(DTR, LTR, label_dict, C=1, xi=1)
    svm.fit() """

