import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from modules.utils.operations import *

def logpdf_GAU_ND(x, mu = None, C = None):
    M = x.shape[0]
    x_c = x - mu
    return -(M/2) * np.log(2*np.pi) - .5 * np.linalg.slogdet(C)[1] - .5 * (np.dot(x_c.T, np.linalg.inv(C)).T * x_c).sum(0)

def loglikelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()

if __name__=="__main__":

    #### TEST SOLUTION ####

    XND = np.load('labs/lab04/Solution/XND.npy')
    mu = np.load('labs/lab04/Solution/muND.npy')
    C = np.load('labs/lab04/Solution/CND.npy') 
    pdfSol = np.load('labs/lab04/Solution/llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print(np.abs(pdfSol - pdfGau).max()) # OK
    
    print(pdfGau.sum()) 
    print(pdfSol.sum()) # here OK BUT in the slides is -270.70478023795044

    X1D = np.load('labs/lab04/Solution/X1D.npy')
    m_ML = mean(X1D)
    C_ML = cov(X1D)
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(row(XPlot), m_ML, C_ML)))
    #plt.show()

    print(loglikelihood(X1D, m_ML, C_ML))