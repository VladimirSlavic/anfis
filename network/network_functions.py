import numpy as np


def forward_pass_rules(X, a, b):
    """

    :param X: one input (1, 2)
    :param a: (num_rules, 2)
    :param b: (num_rules, 2)
    :return:
    """
    cache = (X, a, b)

    result = b * (X - a)
    result = 1 / (1 + np.exp(result))
    return result, cache


def forward_pass_function(X, f, bias):
    """

    :param X: one input (1, 2)
    :param a: (num_rules, 2)
    :param b: (num_rules, 2)
    :return:
    """
    cache = (X, f, bias)

    result = X.dot(f) + bias
    return result, cache


def quadratic_loss(y, o_k):
    N = y.shape[0]
    loss = (1 / N) * (y - o_k) ** 2
    gradient = - (2/N) * (y - o_k)

    return loss, gradient
