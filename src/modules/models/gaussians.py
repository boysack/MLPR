import numpy as np
from scipy.special import logsumexp
from modules.utils.operations import col, mean, cov

class GaussianModel:
    def __init__(self, D, L, label_dict, l_priors = None):
        self.D = D
        self.L = L
        self.label_dict = label_dict
        #??
        #self.parameters = {}
        # used for calibration (futher developement to do)
        self.l_priors = l_priors

    def fit(self):
        pass

    # TODO: insert the binary stuff
    # TODO: insert prior stuff
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

    # TODO: separate the llr calculation from the prediction, as suggested by the professor in lab05
    def predict(self, D):
        # return just scores and prediction
        binary = (len(self.label_dict) == 2)

        # calculate log likelihoods and log priors of each class
        # if present, use the priors already provided
        if self.l_priors is None:
            # list comprehension (usually faster than for loops) and vectorized operation (is actually the fastest, probably going to keep this function as the main one)
            results = [(logpdf_GAU_ND(D, self.parameters[label_int][0], self.parameters[label_int][1]), np.log(self.L[self.L==label_int].size/self.L.size)) for label_int in self.label_dict.values()]
        else:
            results = [(logpdf_GAU_ND(D, self.parameters[label_int][0], self.parameters[label_int][1]), self.l_priors[label_int]) for label_int in self.label_dict.values()]
        
        l_likelihoods, l_priors = zip(*results)
        l_likelihoods = np.array(l_likelihoods)
        l_priors = col(np.array(l_priors))

        #print(f"check: {np.all((logpdf_GAU_ND(D, self.parameters[1][0], self.parameters[1][1]) == l_likelihoods[0,:]))}")
        
        # ma conviene davvero?
        values = list(self.label_dict.values())
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
            Dc = self.D[:,self.L==label_int]
            mu = mean(Dc)
            C = cov(Dc)
            p = (mu, C)
            parameters[label_int] = p
        self.parameters = parameters

class NaiveGModel(GaussianModel):    
    def fit(self):
        # what if, in the split, there's no item belonging to one class?
        parameters = {}
        for _, label_int in self.label_dict.items():
            Dc = self.D[:,self.L==label_int]
            mu = mean(Dc)
            C = np.identity(Dc.shape[0]) * cov(Dc)
            p = (mu, C)
            parameters[label_int] = p
        self.parameters = parameters

class TiedGModel(GaussianModel):
    def fit(self):
        # what if, in the split, there's no item belonging to one class?
        parameters = {}
        C = cov(self.D)
        for _, label_int in self.label_dict.items():
            Dc = self.D[:,self.L==label_int]
            mu = mean(Dc)
            p = (mu, C)
            parameters[label_int] = p
        self.parameters = parameters

def error_rate(L, predictions):
    wrong_p = (L!=predictions).sum()
    error_rate = wrong_p/L.size
    return error_rate


# TODO: change x in D (since it works for a whole dataset)
# TODO: find a way to apply the same calculation if there's a series of parameters (means and covariance matrices). In this way
    # externally can be applied the logpdf_GAU_ND without any for loop for each parameter member
def logpdf_GAU_ND(x, mu = None, C = None):
    if np.isscalar(C):
        logdet = np.log(C)
        inv = 1/C
    else:
        logdet = np.linalg.slogdet(C)[1]
        inv = np.linalg.inv(C)
    M = x.shape[0]
    x_c = x - mu
    # fix the fact that C could be even a scalar
    return -(M/2) * np.log(2*np.pi) - .5 * logdet - .5 * (np.dot(x_c.T, inv).T * x_c).sum(0)

def loglikelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()