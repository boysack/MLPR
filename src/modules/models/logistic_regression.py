from modules.models.model import Model
from modules.utils.operations import row, col

from scipy.optimize import fmin_l_bfgs_b
import numpy as np

#TODO: label_dict?

class LogisticRegression(Model):
    def __init__(self, D, L, label_dict, l = None, w = None, l_priors = None):
        # if single feature, just add a dimension if D is a raveled array
        if len(D.shape) == 1:
            D = D[np.newaxis, :]
        self.D = D
        # TODO: correctly calculating Z involves that the labels are ranged from 0 to 1 (happen just when manually changing
        # the label dictionary)      
        self.L = L
        # TODO: check if for logreg and svm is necessary to pass label_dictionary
        self.label_dict = label_dict
        
        if l is None:
            self.l = 0
        else:
            self.l = l

        if len(label_dict) != 2:
            raise Exception("Logistic Regression is implemented just for binary problems!")
        self.binary = True

        if w is None:
            self.w = np.random.randn(D.shape[0] + 1).astype(np.float64)
        else:
            self.w = w

        # TODO: is it better to set the log ratio to be zero
        self.l_emp_priors = col(np.log(np.array([self.L[self.L==l].size for l in self.label_dict.values()])/self.L.size))

        if l_priors is None:
            # non prior weighted, use empirical one
            l_priors = self.l_emp_priors
        else:
            # prior weighted
            l_priors = col(l_priors)

        self.l_priors = l_priors

    def fit(self):
        if np.all(self.l_emp_priors == self.l_priors):
            # non prior weighted
            self.w, loss, _ = fmin_l_bfgs_b(func = self.logreg_obj_binary, x0 = self.w)
        else:
            self.w, loss, _ = fmin_l_bfgs_b(func = self.logreg_obj_binary_prior_weighted, x0 = self.w)

        return loss

    # TODO: check if implementable on the parent class (almost the same in gaussian)
    def predict(self, D):
        if len(D.shape) == 1:
            D = D[np.newaxis, :]
        l_scores = self.get_scores(D)
        predictions = self.get_predictions(l_scores)
        return predictions, l_scores

    # TODO: adapt to multiclass case
    def get_scores(self, D):
        if len(D.shape) == 1:
            D = D[np.newaxis, :]
        w, b = self.w[:-1], self.w[-1]
        llr = np.dot(w, D) + b - (self.l_priors[1] - self.l_priors[0])
        return llr

    # TODO: is it certain that scores > 0 belong to the 1 class?
    def get_predictions(self, l_scores):
        l_scores = l_scores + (self.l_priors[1] - self.l_priors[0])
        predictions = np.empty(l_scores.shape)
        predictions[l_scores > 0] = 1
        predictions[l_scores <= 0] = 0

        return predictions
    
    # TODO: not used anymore
    def set_threshold_from_priors_binary(self, t_prior):
        self.l_priors = col(np.log([(1-t_prior), t_prior]))

    # it's actually as a prior weighted by the empirical priors
    def logreg_obj_binary(self, w):
        w, b = w[0:-1], w[-1]

        n = self.D.shape[1]
        S = np.dot(w, self.D) + b
        Z = 2 * self.L - 1

        f = self.l/2 * np.linalg.norm(w)**2 + np.logaddexp(0, -Z * S).sum()/n
        
        # TODO: sometimes, overflow in runtime
        # TODO: is np.longdouble really beneficial?
        G = -Z / (1 + np.exp(Z * S, dtype=np.longdouble))
        #G = -Z / (1 + np.exp(np.logaddexp(Z, S)))
        fprime = np.append((self.l * w + (G * self.D).sum(1)/n).ravel(), G.sum()/n)

        return f, fprime
    
    def logreg_obj_binary_prior_weighted(self, w):
        w, b = w[0:-1], w[-1]

        n_t = self.L[self.L==1].size
        n_f = self.L[self.L==0].size
        priors = np.exp(self.l_priors)
        X_i = np.empty(self.L.shape)
        X_i[self.L==0] = priors[0]/n_f
        X_i[self.L==1] = priors[1]/n_t
        
        S = np.dot(w, self.D) + b
        Z = 2 * self.L - 1

        f = self.l/2 * np.linalg.norm(w)**2 + (np.logaddexp(0, -Z * S) * X_i).sum()

        G = -Z / (1 + np.exp(Z * S))
        fprime = np.append((self.l * w + ((G * self.D) * X_i).sum(1)).ravel(), (G * X_i).sum())

        return f, fprime
        

# TODO: maybe in the object make the logreg call the external function