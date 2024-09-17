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

class TiedNaiveGModel(GaussianModel):
    def fit(self):
        parameters = {}
        C = 0
        for label_int in self.label_dict.values():
            Dc = self.D[:,self.L==label_int]
            mu = mean(Dc)
            Dc = Dc - mu
            parameters[label_int] = mu
            C += np.dot(Dc, Dc.T)
        C /= (self.D.shape[1]) * np.eye(Dc.shape[0])
        self.parameters = {k:(mu, C) for (k, mu) in parameters.items()}

class GaussianMixtureModel(Model):
    def __init__(self, D, L = None, label_dict = None, l_priors = None, gmm = None, n = None, psi = 0.01, d = 10**-100, alpha = 0.1):
        self.D = D
        # pass no L or no label dictionary to simulate a distribution over the whole dataset
        if L is None or label_dict is None:
            self.L = np.zeros(D.shape[1])
            self.label_dict = {"0": 0}
        else:
            self.L = L
            self.label_dict = label_dict

        if l_priors is None:
            self.l_priors = col(np.array([np.log(self.L[self.L==label_int].size/self.L.size) for label_int in self.label_dict.values()]))
        else:
            self.l_priors = col(l_priors)

        # Gaussian Mixture Model parameters
        if gmm is None:
            self.gmm = {}
            for label_int in self.label_dict.values():
                # LBG
                self.gmm[label_int] = [(1, mean(D[:, self.L==label_int]), covariance_constraining(psi, cov(D[:, self.L==label_int])))]
        else:
            # set manually initial point
            self.gmm = gmm

        self.psi = psi
        
        # Number of Gaussian component
        # TODO: deal even with number that aren't multiple of 2?
        if n is not None:
            self.n = (2**np.round(np.log2(n))).astype(int)
        else:
            self.n = None
            
        # stopping criterion for fitting function
        self.d = d

        # LBG algorithm displacement
        self.alpha = alpha

    def fit(self):
        avg_loss = self.em()
        if self.n is not None:
            # LBG
            n = len(self.gmm[0])
            avg_loss = [avg_loss]
            while n < self.n:
                self.lbg()
                avg_loss.append(self.em())
                n = len(self.gmm[0])
        return avg_loss
        
    def em(self):
        c_l_likelihood = {}
        for c in self.label_dict.values():
            old_l_likelihood = -np.inf
            l_likelihood = 0

            while True:
                l_marginal, _, l_posterior_g = logpdf_GMM(self.D[:, self.L==c], self.gmm[c])
                l_likelihood = l_marginal.sum()
                if l_likelihood - old_l_likelihood < self.d:
                    old_l_likelihood = l_likelihood
                    break
                old_l_likelihood = l_likelihood

                posterior_g = np.exp(l_posterior_g)

                self.gmm[c] = self.maximize(posterior_g, c)
            
            c_l_likelihood[c] = l_likelihood/self.D.shape[1]

        return c_l_likelihood

    def maximize(self, posterior_g):
        pass
    
    def lbg(self):
        for c in self.label_dict.values():
            new_gmm = []
            for g in self.gmm[c]:
                w_g = g[0]
                mu_g = g[1]
                C_g = g[2]
                U, s, Vh = np.linalg.svd(C_g)
                d = U[:, 0:1] * s[0]**0.5 * self.alpha
                new_gmm.extend(
                    [(w_g/2, mu_g - d, C_g),
                    (w_g/2, mu_g + d, C_g)]
                )
            self.gmm[c] = new_gmm

    def predict(self, D):
        # TODO: implement binary?
        l_scores = self.get_scores(D)
        predictions = self.get_predictions(l_scores)

        return predictions, l_scores

    def get_scores(self, D):
        l_likelihoods = np.array([logpdf_GMM(D, self.gmm[label_int])[0] for label_int in self.label_dict.values()])

        return l_likelihoods

    def get_predictions(self, l_scores):
        l_posteriors = l_scores +  self.l_priors
        #b_costs = expected_bayes_cost(self.cost_matrix, l_posteriors, is_log=True) #??
        #predictions = np.argmin(b_costs, axis=0)
        predictions = np.argmax(l_posteriors, axis=0)
        predictions = np.array([list(self.label_dict.values())[p] for p in predictions])
    
        return predictions.astype(int)
    
    # TODO: get, for the further models, if it makes sense to implement it here
    # used to plot bayes error plot
    def set_threshold_from_priors_binary(self):
        pass

class MVGMModel(GaussianMixtureModel):
    def maximize(self, posterior_g, c):
        Z_g = posterior_g.sum(1)
        F_g = (posterior_g[:, np.newaxis, :] * self.D[:, self.L==c]).sum(2)
        S_g = np.dot(posterior_g[:, np.newaxis, :] * self.D[:, self.L==c], self.D[:, self.L==c].T)

        # TODO: is it really beneficial? try to iteratively calculate (and check if it's right this way)
        mu = (F_g/Z_g[:, np.newaxis]).T
        C = (S_g/Z_g[:, np.newaxis, np.newaxis]).T - np.einsum('if,jf->ijf', mu, mu)
        w = Z_g/Z_g.sum()

        return [(w[i], col(mu[:, i]), covariance_constraining(self.psi, C[:, :, i])) for i in range(len(self.gmm[0]))]

class NaiveGMModel(GaussianMixtureModel):    
    def maximize(self, posterior_g, c):
        Z_g = posterior_g.sum(1)
        F_g = (posterior_g[:, np.newaxis, :] * self.D[:, self.L==c]).sum(2)
        S_g = np.dot(posterior_g[:, np.newaxis, :] * self.D[:, self.L==c], self.D[:, self.L==c].T)

        # TODO: is it really beneficial? try to iteratively calculate (and check if it's right this way)
        mu = (F_g/Z_g[:, np.newaxis]).T
        C = (S_g/Z_g[:, np.newaxis, np.newaxis]).T - np.einsum('if,jf->ijf', mu, mu)
        w = Z_g/Z_g.sum()

        return [(w[i], col(mu[:, i]), covariance_constraining(self.psi, (C[:, :, i]) * np.eye(C.shape[0]))) for i in range(len(self.gmm[0]))]
    
class TiedGMModel(GaussianMixtureModel):
    def maximize(self, posterior_g, c):
        Z_g = posterior_g.sum(1)
        F_g = (posterior_g[:, np.newaxis, :] * self.D[:, self.L==c]).sum(2)
        S_g = np.dot(posterior_g[:, np.newaxis, :] * self.D[:, self.L==c], self.D[:, self.L==c].T)

        # TODO: is it really beneficial? try to iteratively calculate (and check if it's right this way)
        mu = (F_g/Z_g[:, np.newaxis]).T
        C = (S_g/Z_g[:, np.newaxis, np.newaxis]).T - np.einsum('if,jf->ijf', mu, mu)
        w = Z_g/Z_g.sum()

        C = sum([w[i]*C[:,:,i] for i in range(len(self.gmm[0]))])

        return [(w[i], col(mu[:, i]), covariance_constraining(self.psi, C)) for i in range(len(self.gmm[0]))]

def logpdf_GAU_ND(D, mu = None, C = None):
    if np.isscalar(C):
        logdet = np.log(C)
        inv = 1/C
    else:
        logdet = np.linalg.slogdet(C)[1]
        inv = np.linalg.inv(C)
    M = D.shape[0]
    D_c = D - mu
    return -(M/2) * np.log(2*np.pi) - .5 * logdet - .5 * (np.dot(D_c.T, inv).T * D_c).sum(0)

# used for maximum likelihood evaluation (actually not used in practice)
def loglikelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()

def expected_bayes_cost(C, posteriors, is_log = False):
    if is_log is True:
        posteriors = np.exp(posteriors)
    
    return np.dot(C, posteriors)

def logpdf_GMM(D, gmm):
    # TODO: check if broadcasting is exploitable (since for too much high number of gaussian i think there's overfitting, probably some low number of it is
    # sufficient to obtain maximum performance by this model, i.e. the list implementation is sufficient and more readable)
    l_joint_g = np.empty((len(gmm), D.shape[1]))
    for i, g in enumerate(gmm):
        w, mu, C = g
        # log f_x(x) + log w = log N(x|mu,C) + log w
        l_joint_g[i, :] = logpdf_GAU_ND(D, mu, C) + np.log(w)

    # log( sum_g( exp( l_joint ) ) )
    l_marginal_g = logsumexp(l_joint_g, axis=0)
    l_posterior_g = l_joint_g - l_marginal_g

    return l_marginal_g, l_joint_g, l_posterior_g

def covariance_constraining(psi, C):
    U, s, _ = np.linalg.svd(C)
    s[s<psi] = psi
    C = U @ ((s*U).T)

    return C

def from_list_of_tuples_in_multidimensional_ndarray():
    pass