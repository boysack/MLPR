from modules.models.model import Model
from modules.utils.operations import row, col

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class SupportVectorMachine(Model):
    def __init__(self, D, L, label_dict, C = None, xi = 1, alpha = None, kernel = None, kernel_args = []):
        self.D = np.vstack((D, np.full((1, D.shape[1]), xi**(1/2))))
        # TODO: correctly calculating Z involves that the labels are ranged from 0 to 1
        # TODO: we assume L is ravel! check if necessary to ravel manually
        self.L = L
        # can be used further in the code, so save in memory
        self.Z = 2 * L - 1

        self.label_dict = label_dict

        # hyperparameters
        self.xi = xi
        # if None => hard margin
        self.C = C

        self.kernel_dict = {
            "polynomial": polynomial_kernel,
            "radial_basis_function": radial_basis_function_kernel
        }

        if kernel in self.kernel_dict:
            self.kernel = kernel
            self.kernel_func = self.kernel_dict[self.kernel]
            kernel_args.append(xi)
            self.kernel_args = kernel_args
        else:
            self.kernel = None
            self.kernel_func = original_kernel
            self.kernel_args = []
        

        self.binary = True

        if alpha is None:
            self.alpha = np.random.randn(D.shape[1]).astype(np.float64)
        else:
            # TODO: dimension check
            self.alpha = alpha

        self.w = primal_from_dual(self.alpha, self.D, self.Z)

    def fit(self):
        if self.C is None:
            self.alpha, dual_loss, _ = fmin_l_bfgs_b(func = self.svm_dual_obj_binary, factr=1.0, x0 = self.alpha)
        else:
            self.alpha, dual_loss, _ = fmin_l_bfgs_b(func = self.svm_dual_obj_binary, factr=1.0, bounds=[(0, self.C)] * self.alpha.size, x0 = self.alpha)

        self.w = primal_from_dual(self.alpha, self.D, self.Z)

        loss = {
            "dual_loss": - dual_loss,
            "primal_loss": self.svm_primal_obj_binary(self.w)
        }

        return loss

    def predict(self, D):
        scores = self.get_scores(D)
        predictions = self.get_predictions(scores)
        return predictions, scores
    
    # TODO: adapt to kernel
    def get_scores(self, D):
        D = np.vstack((D, np.full((1, D.shape[1]), self.xi**(1/2))))

        # it would work even with raveled alpha and Z
        
        return (col(self.alpha.ravel() * self.Z) * self.kernel_func(self.D, D, *self.kernel_args)).sum(0)

    def get_predictions(self, scores):
        predictions = np.empty(scores.shape)
        predictions[scores > 0] = 1
        predictions[scores <= 0] = 0

        return predictions
    
    # TODO: implemented in Gaussian and LogReg. Do it make sense here? If not => create different parent class for the two models
    def set_threshold_from_priors_binary(self):
        pass

    # TODO: svm hereditarily binary? not necessary to specify "binary"?
    # TODO: adapt to kernel
    def svm_dual_obj_binary(self, alpha):
        h_H = hat_H(self.D, Z=self.Z, kernel_func=self.kernel_func, kernel_args=self.kernel_args)

        f = np.dot(np.dot(row(alpha), h_H), col(alpha))/2 - np.dot(row(alpha), np.ones(col(alpha).shape))
        fprime = (np.dot(h_H, col(alpha)) - 1).ravel()
        return f, fprime
    
    def svm_primal_obj_binary(self, w):
        if self.C is None:
            C = 1
        else:
            C = self.C

        return np.linalg.norm(self.w)**2 /2 + C * np.maximum(0, 1 - self.Z * np.dot(row(w), self.D)).sum()

    def get_duality_gap(self):
        return self.svm_primal_obj_binary(self.w) + self.svm_dual_obj_binary(self.alpha)[0].item()

def original_kernel(D_1, D_2):

    return np.dot(D_1.T, D_2)

def polynomial_kernel(D_1, D_2, d, c, xi, remove_last = True):
    #k(x_1, x_2) = (x_1^T * x_2 + 1)^d
    if remove_last:
        D_1 = D_1 [:-1,:]
        D_2 = D_2 [:-1,:]

    return (np.dot(D_1.T, D_2) + c) ** d + xi

def radial_basis_function_kernel(D_1, D_2, g, xi, remove_last = True):
    #k(x_1, x_2) = e^(-gamma||x_1-x_2||^2)
    if remove_last:
        D_1 = D_1 [:-1,:]
        D_2 = D_2 [:-1,:]
    # || x_1 - x_2 ||^2 = || x_1 ||^2 + || x_2 ||^2 - 2 x_1^T x_2
    squared_norms_1 = np.sum(D_1 ** 2, axis=0).reshape(-1, 1)
    squared_norms_2 = np.sum(D_2 ** 2, axis=0).reshape(-1, 1)
    pairwise_squared_distances = squared_norms_1 + squared_norms_2.T - 2 * np.dot(D_1.T, D_2)

    return np.exp( -g * pairwise_squared_distances) + xi

# OK WITH MONODIMENSIONAL ALPHA AND Z
def primal_from_dual(alpha, D, Z):
    return (alpha.ravel() * Z.ravel() * D).sum(1)

# OK WITH MONODIMENSIONAL ALPHA AND Z
def hat_H(D, L = None, Z = None, kernel_func = original_kernel, kernel_args = []):
    if Z is None:
        if L is None:
            raise Exception("One between L and Z parameter must be passed to the function.")
        Z = 2 * L - 1
    #print(*kernel_args)
    h_G = kernel_func(D, D, *kernel_args)
    h_Z = np.dot(col(Z), row(Z))

    return h_G * h_Z