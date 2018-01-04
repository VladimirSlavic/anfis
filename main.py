import numpy as np

from network.anfisnetwork import Anfis

X = np.array([[1, 2],
              [3, 3]])


a = np.array([[1, 1],
              [2, 2]])
b = np.array([[2, 2],
               [3, 1]])
f = np.array([[1, 1],
              [1, 2]])
r = np.array([0, 0])

print(a[:, 0].reshape(-1, 1))

