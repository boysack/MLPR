import numpy as np
from scipy.special import logsumexp
from modules.utils.operations import col, mean, cov
from modules.models.model import Model

class GaussianModel(Model):
    def __init__(self, D, L, label_dict, l_priors = None, cost_matrix = None):
        self.D = D
        self.L = L
        self.label_dict = label_dict

        size = len(self.label_dict)
        if size == 1:
            raise Exception("Classes must be at least two")
        if size != np.unique(L).size:
            raise Exception("There's some missing class in your split")
        
        # if label_dict is not {key_00: 0, key_01: 1}, the problem is not considered binary
        vals = list(label_dict.values())
        self.binary = (vals.count(0) == 1 and vals.count(1) == 1 and size == 2)

        if l_priors is not None:
            l_priors = col(l_priors)
        self.l_priors = l_priors

        if cost_matrix is None:
            self.cost_matrix = np.ones((len(label_dict), len(label_dict)))
            np.fill_diagonal(self.cost_matrix, 0)
        else:
            self.cost_matrix = cost_matrix

    def fit(self):
        pass

    def debug_predict(self, D):
        if self.l_priors is None:
            results = [(logpdf_GAU_ND(D, self.parameters[label_int][0], self.parameters[label_int][1]), np.log(self.L[self.L==label_int].size/self.L.size)) for label_int in self.label_dict.values()]
        else:
            results = [(logpdf_GAU_ND(D, self.parameters[label_int][0], self.parameters[label_int][1]), self.l_priors[label_int]) for label_int in self.label_dict.values()]

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
        # get costs
        b_costs = expected_bayes_cost(self.cost_matrix, l_posteriors, is_log=True)
        # get the prediction
        predictions = np.argmin(b_costs, axis=0)
        values = list(self.label_dict.values())
        predictions = np.array([values[p] for p in predictions])
        
        results = {
            "l_likelihoods": l_likelihoods,
            "l_marginals": l_marginals,
            "l_posteriors": l_posteriors,
            "l_joints": l_joints,
            "b_costs": b_costs
        }

        return predictions, results

    def predict(self, D, bin = None):
        if bin is None:
            bin = self.binary
        l_scores = self.get_scores(D, bin)
        predictions = self.get_predictions(l_scores, bin)
        return predictions, l_scores
    
    def get_scores(self, D, bin = None):
        if bin is None:
            bin = self.binary

        l_likelihoods = np.array([logpdf_GAU_ND(D, self.parameters[label_int][0], self.parameters[label_int][1]) for label_int in self.label_dict.values()])

        if bin:
            # PUT ALWAYS THE CLASS 1 AT THE NUMERATOR AND THE CLASS 0 AT THE DENOMINATOR (JUST FOR INTERPRETATION PURPOSES)
            true_idx = list(self.label_dict.values()).index(1)
            false_idx = (true_idx + 1 ) % 2

            # loglikelihood ratios
            # llr = log(P(x|true)/P(x|false)) (where true = 1, and false = 0)
            l_scores = l_likelihoods[true_idx, :] - l_likelihoods[false_idx, :]
        else:
            # loglikelihoods
            l_scores = l_likelihoods
        
        return l_scores

    def get_predictions(self, l_scores, bin = None):
        if bin is None:
            bin = self.binary

        if self.l_priors is None:
            self.l_priors = col(np.array([np.log(self.L[self.L==label_int].size/self.L.size) for label_int in self.label_dict.values()]))
        
        values = list(self.label_dict.values())
        if bin:
            # PUT ALWAYS THE CLASS 1 AT THE NUMERATOR AND THE CLASS 0 AT THE DENOMINATOR
            true_idx = values.index(1)
            false_idx = (true_idx + 1 ) % 2

            priors = np.exp(self.l_priors)
            threshold = -1 * np.log(priors[true_idx]*self.cost_matrix[false_idx, true_idx]/
                                    (priors[false_idx]*self.cost_matrix[true_idx, false_idx]))
            
            predictions = np.empty(l_scores.shape)
            predictions[l_scores > threshold] = values[true_idx]
            predictions[l_scores <= threshold] = values[false_idx]

        else:
            l_posteriors = l_scores +  self.l_priors
            b_costs = expected_bayes_cost(self.cost_matrix, l_posteriors, is_log=True)
            predictions = np.argmin(b_costs, axis=0)
            predictions = np.array([values[p] for p in predictions])
        
        return predictions.astype(int)
    
    # TODO: check if binary?
    def set_threshold_from_priors_binary(self, t_prior):
        self.l_priors = col(np.log([(1-t_prior), t_prior]))
        self.cost_matrix = np.ones((2,2))
        np.fill_diagonal(self.cost_matrix, 0)

class MVGModel(GaussianModel):
    def fit(self):
        parameters = {}
        for label_int in self.label_dict.values():
            Dc = self.D[:,self.L==label_int]
            mu = mean(Dc)
            C = cov(Dc)
            p = (mu, C)
            parameters[label_int] = p
        self.parameters = parameters

class NaiveGModel(GaussianModel):    
    def fit(self):
        parameters = {}
        for label_int in self.label_dict.values():
            Dc = self.D[:,self.L==label_int]
            mu = mean(Dc)
            C = np.identity(Dc.shape[0]) * cov(Dc)
            p = (mu, C)
            parameters[label_int] = p
        self.parameters = parameters

class TiedGModel(GaussianModel):
    def fit(self):
        parameters = {}
        C = 0
        for label_int in self.label_dict.values():
            Dc = self.D[:,self.L==label_int]
            mu = mean(Dc)
            Dc = Dc - mu
            parameters[label_int] = mu
            C += np.dot(Dc, Dc.T)
        C /= self.D.shape[1]
        self.parameters = {k:(mu, C) for (k, mu) in parameters.items()}

# TODO: implement TiedNaive

def logpdf_GAU_ND(x, mu = None, C = None):
    if np.isscalar(C):
        logdet = np.log(C)
        inv = 1/C
    else:
        logdet = np.linalg.slogdet(C)[1]
        inv = np.linalg.inv(C)
    M = x.shape[0]
    x_c = x - mu
    return -(M/2) * np.log(2*np.pi) - .5 * logdet - .5 * (np.dot(x_c.T, inv).T * x_c).sum(0)

def loglikelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()

# TODO: move??
def expected_bayes_cost(C, posteriors, is_log = False):
    if is_log is True:
        posteriors = np.exp(posteriors)
    
    return np.dot(C, posteriors)