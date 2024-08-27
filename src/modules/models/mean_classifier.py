import numpy as np
from modules.utils.operations import col
from modules.features.dimensionality_reduction import lda

class LdaBinaryClassifier():
    def __init__(self, D, L, label_dict):
        if len(label_dict) != 2:
            raise Exception("Classification must be binary")
        self.w = 1
        if D.shape[0] > 1:
            self.w, D = lda(D, L)

        self.D = D
        self.L = L
        self.label_dict = label_dict
        self.invert = False

    def fit(self):
        m_first = self.D[0, self.L==list(self.label_dict.values())[0]].mean()
        m_second = self.D[0, self.L==list(self.label_dict.values())[1]].mean()
        
        self.threshold = (m_first + m_second) / 2.0
        if m_first < m_second:
            self.invert = True

    def predict(self, D):
        predictions = np.zeros(shape=(D.shape[1]), dtype=np.int32)
        first = list(self.label_dict.values())[0]
        second = list(self.label_dict.values())[1]

        if self.invert == True:
            t = first
            first = second
            second = t

        if D.shape[0] > 1:
            D = np.dot(self.w.T, D)
        D = D.ravel()

        predictions[D[0] >= self.threshold] = first
        predictions[D[0] < self.threshold] = second

        return predictions

