import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.data.dataset import *
from modules.utils.operations import *
from modules.models.gaussians import logpdf_GAU_ND

import numpy as np
from scipy.special import logsumexp

class GaussianModel:
    def __init__(self, D, L, label_dict):
        self.D = D
        self.L = L
        self.label_dict = label_dict

    def fit(self):
        pass

    def debug_predict(self, D):
        # return everything
        # calculate log likelihoods and log priors of each class
        # list comprehension (usually faster than for loops) and vectorized operation (is actually the fastest, probably going to keep this function as the main one)
        results = [(logpdf_GAU_ND(D, self.parameters[label_int][0], self.parameters[label_int][1]), np.log(self.L[self.L==label_int].size/self.L.size)) for label_int in self.label_dict.values()]
        l_likelihoods, l_priors = zip(*results)
        l_likelihoods = np.array(l_likelihoods)
        l_priors = col(np.array(l_priors))
        #l_priors = col(np.array([1/len(self.label_dict)]*len(self.label_dict)))
        
        # calculate log joint probability
        l_joints = l_likelihoods + l_priors
        # calculate log marginal probability
        l_marginals = logsumexp(l_joints, axis=0)
        # calculate log posterior probability
        l_posteriors = l_joints - l_marginals

        # get the prediction index and change the value with the integer associated with that specific class
        predictions = np.argmax(l_posteriors, axis=0)
        values = list(self.label_dict.values())
        predictions = np.array([values[p] for p in predictions])
        
        # return the exponential value of the log likelihood, the argmax (the most likely to be the belonging class) and the other log probabilities (just to check)
        results = {
            "l_likelihoods": l_likelihoods,
            "l_marginals": l_marginals,
            "l_posteriors": l_posteriors,
            "l_joints": l_joints
        }

        return predictions, results

    def predict(self, D):
        # return just scores and prediction
        binary = (len(self.label_dict) == 2)

        # calculate log likelihoods and log priors of each class
        # list comprehension (usually faster than for loops) and vectorized operation (is actually the fastest, probably going to keep this function as the main one)
        results = [(logpdf_GAU_ND(D, self.parameters[label_int][0], self.parameters[label_int][1]), np.log(self.L[self.L==label_int].size/self.L.size)) for label_int in self.label_dict.values()]
        l_likelihoods, l_priors = zip(*results)
        l_likelihoods = np.array(l_likelihoods)
        l_priors = col(np.array(l_priors))

        #print(f"check: {np.all((logpdf_GAU_ND(D, self.parameters[1][0], self.parameters[1][1]) == l_likelihoods[0,:]))}")
        
        # ma conviene davvero?
        values = list(label_dict.values())
        if binary:
            # llr = log(P(x|0)/P(x|1))
            llr = l_likelihoods[0, :] - l_likelihoods[1, :]
            # t = - log(P(0)/P(1))
            threshold =  l_priors[1, 0] - l_priors[0, 0]
            l_scores = llr - threshold

            predictions = np.empty(llr.shape)
            predictions[l_scores >= 0] = values[0]
            predictions[l_scores < 0] = values[1]
        else:
            l_scores = l_likelihoods + l_priors
            predictions = np.argmax(l_scores, axis=0)
            predictions = np.array([values[p] for p in predictions])
        
        return predictions, l_scores

class MVGModel(GaussianModel):
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
    
    def slow_predict(self, D):
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

class NaiveGModel(GaussianModel):    
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

class TiedGModel(GaussianModel):
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

def error_rate(L, predictions):
    wrong_p = (L!=predictions).sum()
    error_rate = wrong_p/L.size
    return error_rate

from modules.models.gaussians import TiedGModel

if __name__=="__main__":
    D, L, label_dict = load("labs/data/iris.csv")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    ### MULTIVARIATE GAUSSIAN MODEL ###
    # mine: error rate = 2.00%
    # prof: error rate = 4.00%
    print("### MULTIVARIATE GAUSSIAN MODEL ###")
    mvg = MVGModel(DTR, LTR, label_dict)
    mvg.fit()
    """ for label_int, parameters in mvg.parameters.items():
        print(f"class {label_int}")
        print(f"mean")
        print(parameters[0])
        print(f"covariance matrix")
        print(parameters[1]) """
    predictions, results = mvg.debug_predict(DVAL)
    """ print(posteriors)
    print(predictions)
    print(LVAL) """

    e_r = error_rate(LVAL, predictions)
    print(f"error_rate: {e_r*100:.2f}%")
    print(f"accuracy: {(1-e_r)*100:.2f}%")

    #load solution and compare
    """ sol_l_marginals = np.load("labs/lab05/Solution/logMarginal_MVG.npy")
    sol_l_posteriors = np.load("labs/lab05/Solution/logPosterior_MVG.npy")
    sol_l_joints = np.load("labs/lab05/Solution/logSJoint_MVG.npy")

    print(trunc(sol_l_posteriors, decs=2)==trunc(results["l_posteriors"], decs=2)) """

    ### NAIVE GAUSSIAN MODEL ###
    # mine: error rate = 4.00%
    # prof: error rate = 4.00% - OK
    print("### NAIVE GAUSSIAN MODEL ###")
    mvg = NaiveGModel(DTR, LTR, label_dict)
    mvg.fit()
    """ for label_int, parameters in mvg.parameters.items():
        print(f"class {label_int}")
        print(f"mean")
        print(parameters[0])
        print(f"covariance matrix")
        print(parameters[1]) """

    predictions, results = mvg.debug_predict(DVAL)
    """ print(posteriors)
    print(predictions)
    print(LVAL) """

    e_r = error_rate(LVAL, predictions)
    print(f"error_rate: {e_r*100:.2f}%")
    print(f"accuracy: {(1-e_r)*100:.2f}%")

    ### TIED GAUSSIAN MODEL ###
    # mine: error rate = 2.00%
    # prof: error rate = 2.00% - OK
    print("### TIED GAUSSIAN MODEL ###")
    mvg = TiedGModel(DTR, LTR, label_dict)
    mvg.fit()
    """ for label_int, parameters in mvg.parameters.items():
        print(f"class {label_int}")
        print(f"mean")
        print(parameters[0])
        print(f"covariance matrix")
        print(parameters[1]) """

    predictions, results = mvg.debug_predict(DVAL)
    """ print(posteriors)
    print(predictions)
    print(LVAL) """

    e_r = error_rate(LVAL, predictions)
    print(f"error_rate: {e_r*100:.2f}%")
    print(f"accuracy: {(1-e_r)*100:.2f}%")

    ### BINARY CLASSIFICATION PROBLEM ###
    # mine: error rate = 5.88%
    # prof: error rate = 8.80%
    print("### BINARY CLASSIFICATION PROBLEM ###")

    # remove all the labels that doesn't match with "Iris-versicolor" and "Iris-virginica" (did it like this since I want to make it 
    # general for more than just one label to remove)
    label_dict = {label_str: label_int for label_str, label_int in label_dict.items() if label_str == "Iris-versicolor" or label_str == "Iris-virginica"}
    D = D[:, np.isin(L, list(label_dict.values()))]
    L = L[np.isin(L, list(label_dict.values()))]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    mvg = MVGModel(DTR, LTR, label_dict)
    mvg.fit()
    predictions, _ = mvg.predict(DVAL)
    debug_predictions, results = mvg.debug_predict(DVAL)
    
    # print(np.all(predictions == debug_predictions)) # OK
    e_r = error_rate(LVAL, predictions)
    print(f"error_rate: {e_r*100:.2f}%")
    print(f"accuracy: {(1-e_r)*100:.2f}%")
    
    #### CHECK SPEED OF THE IMPLEMENTED FUNCTIONS ####

    print("\n\nSPEED!!")
    from time import time

    start = time()
    mvg.slow_predict(DVAL)
    end = time()
    pred = end-start
    print(f"elapsed: {end-start}")

    start = time()
    mvg.fast_predict(DVAL)
    end = time()
    fast_pred = end-start
    print(f"elapsed: {end-start}")

    start = time()
    # the chosen one
    mvg.predict(DVAL)
    end = time()
    fast_fast_pred = end-start
    print(f"elapsed: {end-start}")

    if fast_fast_pred < fast_pred and fast_fast_pred < pred:
        print("fast_fast_pred")
    elif fast_pred < fast_fast_pred and fast_pred < pred:
        print("fast_pred")
    else:
        print("pred")

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
    

    # CHECK THE SPEED OF THE BINARY VERSION OF PREDICTION FUNCTION
    """ from time import time

    mvg = MVGModel(DTR, LTR, label_dict)
    mvg.fit()
    start = time()
    mvg.predict(DVAL, True)
    end = time()
    b = end-start
    print(f" binary -> elapsed time: {b}")

    start = time()
    mvg.predict(DVAL, False)
    end = time()
    nb = end-start
    print(f"~binary -> elapsed time: {nb}")

    if b < nb:
        print("binary")
    else:
        print("not binary") """

    