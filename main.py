import numpy as np

from network.anfisnetwork import Anfis

X = np.array([[1, 2],
              [3, 3]])
y = np.array([4, 8])

#
# a = np.array([[1, 1],
#               [2, 2]])
# b = np.array([[2, 2],
#                [3, 1]])
# f = np.array([[1, 1],
#               [1, 2]])
# r = np.array([0, 0])
#
# print(a[:, 0].reshape(-1, 1))


# w = np.array([[1, 4]])
# ff = w.reshape(-1, 1)
#
# rez = ff[range(2)] - w
#
#
# print(rez)
#
# print(np.sum(rez, axis=1, keepdims=True))

anfis = Anfis()
loss, grads = anfis.loss(X, y)
print(loss)
