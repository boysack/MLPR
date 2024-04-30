from project.packages.utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

class MVG:
    def __init__(self, D, L, label_dict):
        self.D = D
        self.L = L
        self.label_dict = label_dict
    
    def fit(self):
        # what if, in the split, there's no item belonging to one class?
        parameters = {}
        for _, label_int in self.label_dict.items():
            Dc = D[:,L==label_int]
            mu = mean(Dc)
            C = cov(Dc)
            p = (mu, C)
            parameters[label_int] = p
        self.parameters = parameters
    
    def predict(self, D):
        l_posteriors = None
        for label_str, label_int in self.label_dict.items():
            l_likelihoods = logpdf_GAU_ND(D, self.parameters[label_int][0], self.parameters[label_int][1])
            l_prior = np.log(self.L[self.L==label_int].size/self.L.size)
            l_joints = l_likelihoods + l_prior
            # maybe I can move this outside the loop
            if l_posteriors is None:
                l_posteriors = l_joints
            else:
                l_posteriors = np.vstack((l_posteriors, l_joints))
        l_marginals = logsumexp(l_posteriors, axis=0)
        l_posteriors -= l_marginals

        return np.exp(l_posteriors), np.argmax(l_posteriors, axis=0)
    
    def fast_predict(self, D):
        # use vectorized operation in the end of the for loop
        l_likelihoods = None
        l_priors = None
        for label_str, label_int in self.label_dict.items():
            l_l = logpdf_GAU_ND(D, self.parameters[label_int][0], self.parameters[label_int][1])
            l_p = np.array(np.log(self.L[self.L==label_int].size/self.L.size))
            if l_likelihoods is None:
                l_likelihoods = l_l
            else:
                l_likelihoods = np.vstack((l_likelihoods, l_l))
            if l_priors is None:
                l_priors = l_p
            else:
                l_priors = np.vstack((l_priors, l_p))
        l_posteriors = l_likelihoods + l_priors
        l_marginals = logsumexp(l_posteriors, axis=0)
        l_posteriors -= l_marginals

        return np.exp(l_posteriors), np.argmax(l_posteriors, axis=0)
    
    def fast_fast_predict(self, D):
        # list comprehension (usually faster than for loops) and vectorized operation (is actually the fastest, probably going to keep this function as the main one)
        results = [(logpdf_GAU_ND(D, self.parameters[label_int][0], self.parameters[label_int][1]), np.log(self.L[self.L==label_int].size/self.L.size)) for label_int in self.label_dict.values()]
        l_likelihoods, l_priors = zip(*results)
        l_likelihoods = np.array(l_likelihoods)
        l_priors = col(np.array(l_priors))
        
        l_joints = l_likelihoods + l_priors
        l_marginals = logsumexp(l_joints, axis=0)
        l_posteriors = l_joints - l_marginals

        return np.exp(l_posteriors), np.argmax(l_posteriors, axis=0), [l_marginals, l_posteriors, l_joints]

class naive_MVG:
    def __init__(self, D, L, label_dict):
        self.D = D
        self.L = L
        self.label_dict = label_dict
    
    def fit(self):
        # what if, in the split, there's no item belonging to one class?
        parameters = {}
        for _, label_int in self.label_dict.items():
            Dc = D[:,L==label_int]
            mu = mean(Dc)
            C = np.identity(Dc.shape[0]) * cov(Dc)
            p = (mu, C)
            parameters[label_int] = p
        self.parameters = parameters

    def fast_fast_predict(self, D):
        # list comprehension (usually faster than for loops) and vectorized operation (is actually the fastest, probably going to keep this function as the main one)
        results = [(logpdf_GAU_ND(D, self.parameters[label_int][0], self.parameters[label_int][1]), np.log(self.L[self.L==label_int].size/self.L.size)) for label_int in self.label_dict.values()]
        l_likelihoods, l_priors = zip(*results)
        l_likelihoods = np.array(l_likelihoods)
        l_priors = col(np.array(l_priors))
        
        l_joints = l_likelihoods + l_priors
        l_marginals = logsumexp(l_joints, axis=0)
        l_posteriors = l_joints - l_marginals

        return np.exp(l_posteriors), np.argmax(l_posteriors, axis=0), [l_marginals, l_posteriors, l_joints]

class tied_MVG:
    def __init__(self, D, L, label_dict):
        self.D = D
        self.L = L
        self.label_dict = label_dict
    
    def fit(self):
        # what if, in the split, there's no item belonging to one class?
        parameters = {}
        C = cov(D)
        for _, label_int in self.label_dict.items():
            Dc = D[:,L==label_int]
            mu = mean(Dc)
            p = (mu, C)
            parameters[label_int] = p
        self.parameters = parameters

    def fast_fast_predict(self, D):
        # list comprehension (usually faster than for loops) and vectorized operation (is actually the fastest, probably going to keep this function as the main one)
        results = [(logpdf_GAU_ND(D, self.parameters[label_int][0], self.parameters[label_int][1]), np.log(self.L[self.L==label_int].size/self.L.size)) for label_int in self.label_dict.values()]
        l_likelihoods, l_priors = zip(*results)
        l_likelihoods = np.array(l_likelihoods)
        l_priors = col(np.array(l_priors))
        
        l_joints = l_likelihoods + l_priors
        l_marginals = logsumexp(l_joints, axis=0)
        l_posteriors = l_joints - l_marginals

        return np.exp(l_posteriors), np.argmax(l_posteriors, axis=0), [l_marginals, l_posteriors, l_joints]

if __name__=="__main__":
    D, L, label_dict = load("labs/data/iris.csv")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    ### MULTIVARIATE GAUSSIAN MODEL ###
    print("### MULTIVARIATE GAUSSIAN MODEL ###")
    mvg = MVG(DTR, LTR, label_dict)
    mvg.fit()
    """ for label_int, parameters in mvg.parameters.items():
        print(f"class {label_int}")
        print(f"mean")
        print(parameters[0])
        print(f"covariance matrix")
        print(parameters[1]) """
    posteriors, predictions, others = mvg.fast_fast_predict(DVAL)
    print(posteriors)
    print(predictions)
    print(LVAL)

    correct_p = (LVAL==predictions).sum()
    wrong_p = LVAL.size-correct_p

    accuracy = correct_p/LVAL.size
    error_rate = wrong_p/LVAL.size

    print(f"error_rate: {error_rate}")
    print(f"accuracy: {accuracy}")

    ### NAIVE GAUSSIAN MODEL ###
    print("### NAIVE GAUSSIAN MODEL ###")
    mvg = naive_MVG(DTR, LTR, label_dict)
    mvg.fit()
    for label_int, parameters in mvg.parameters.items():
        print(f"class {label_int}")
        print(f"mean")
        print(parameters[0])
        print(f"covariance matrix")
        print(parameters[1])

    posteriors, predictions, _ = mvg.fast_fast_predict(DVAL)
    print(posteriors)
    print(predictions)
    print(LVAL)

    correct_p = (LVAL==predictions).sum()
    wrong_p = LVAL.size-correct_p

    accuracy = correct_p/LVAL.size
    error_rate = wrong_p/LVAL.size

    print(f"error_rate: {error_rate}")
    print(f"accuracy: {accuracy}")

    ### TIED GAUSSIAN MODEL ###
    print("### TIED GAUSSIAN MODEL ###")
    mvg = tied_MVG(DTR, LTR, label_dict)
    mvg.fit()
    for label_int, parameters in mvg.parameters.items():
        print(f"class {label_int}")
        print(f"mean")
        print(parameters[0])
        print(f"covariance matrix")
        print(parameters[1])

    posteriors, predictions, _ = mvg.fast_fast_predict(DVAL)
    print(posteriors)
    print(predictions)
    print(LVAL)

    correct_p = (LVAL==predictions).sum()
    wrong_p = LVAL.size-correct_p

    accuracy = correct_p/LVAL.size
    error_rate = wrong_p/LVAL.size

    print(f"error_rate: {error_rate}")
    print(f"accuracy: {accuracy}")

    #### CHECK SPEED OF THE IMPLEMENTED FUNCTIONS ####
    """ from time import time

    start = time()
    mvg.predict(DVAL)
    end = time()
    pred = end-start
    print(f"elapsed: {end-start}")

    start = time()
    mvg.fast_predict(DVAL)
    end = time()
    fast_pred = end-start
    print(f"elapsed: {end-start}")

    start = time()
    mvg.fast_fast_predict(DVAL)
    end = time()
    fast_fast_pred = end-start
    print(f"elapsed: {end-start}")

    if fast_fast_pred < fast_pred and fast_fast_pred < pred:
        print("fast_fast_pred")
    elif fast_pred < fast_fast_pred and fast_pred < pred:
        print("fast_pred")
    else:
        print("pred") """

    # CHECK PROF SOLUTION (PROBABLY WRONG OR FROM DIFFERENT DATA, SINCE EVEN IN THE PDF THE PROFESSOR SAYS THAT ERROR SHOULD COME
    # OUT OF 0.04, BUT IT ACTUALLY IS 0.02)
    """ import os
    file_names = os.listdir("labs/lab05/Solution")
    file_names = [file_name for file_name in file_names if file_name.endswith("_MVG.npy")]
    values = []
    for file_name in file_names:
        if file_name.startswith("Posterior") or file_name.startswith("llr") or file_name.startswith("SJoint"):
            continue
        values.append(np.load(f"labs/lab05/Solution/{file_name}"))
    
    for idx, value in enumerate(values):
        print(trunc(value, 1)==trunc(others[idx], 1)) """
    
