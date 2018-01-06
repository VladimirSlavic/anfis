import numpy as np

def lab_func(ai, bi, x):
    mu = 1 / (1 + np.exp(bi.T * (x - ai.T)))
    # mu = 1 / (1 + np.exp(bi*(x - ai)))
    # return mu
