import numpy as np
from functions import funcs
from network.network_functions import *


class Anfis:
    def __init__(self, num_rules=2, input_dims=2, output_classes=1, reg=0.0, weight_dev=1e-2, dtype=np.float64,
                 **kwargs):
        self.num_rules = num_rules
        self.dtype = dtype
        self.params = {}
        self.input_dims = input_dims
        self.reg = reg
        self.rule_function = kwargs.pop('rule_function', 'lab_func')
        # implementiraj razliciti tip funkcija za koristit
        self.batch_size = 20
        self.params['rules'] = np.zeros(shape=(num_rules, input_dims))

        self.params['a'] = np.array([[1, 3],
                                     [2, 2]])
        self.params['b'] = np.array([[2, 1],
                                     [3, 1]])
        self.params['f'] = np.array([[1, 1],
                                     [1, 2]])
        self.params['r'] = np.array([0, 0])

        # self.params['a'] = np.random.normal(scale=weight_dev, size=(input_dims, num_rules))
        # self.params['b'] = np.random.normal(scale=weight_dev, size=(input_dims, num_rules))
        # self.params['f'] = np.random.normal(scale=weight_dev, size=(input_dims, num_rules))
        # self.params['r'] = np.zeros(shape=(num_rules))

        if not hasattr(funcs, self.rule_function):
            raise ValueError('Invalid rule function')
        self.rule_function = getattr(funcs, self.rule_function)

    def loss(self, X, y=None):

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        self.cache = {}

        # forwardpass
        score = X
        N = X.shape[0]
        W = np.zeros(shape=(N, self.num_rules))
        o_k = np.zeros(shape=N)
        func_res = np.zeros(shape=(N, self.num_rules))
        mu_a = np.zeros(shape=(N, self.num_rules))
        mu_b = np.zeros(shape=(N, self.num_rules))
        x_less_a = np.zeros(shape=(N, self.num_rules))
        y_less_a = np.zeros(shape=(N, self.num_rules))

        # forward pass calculation
        for i in range(N):
            x_less_a[i] = X[i, 0] - self.params['a'][0]
            s1 = np.exp(self.params['b'][0] * (X[i, 0] - self.params['a'][0]))
            mu_a[i] = 1 / (1 + s1)
            y_less_a[i] = X[i, 1] - self.params['a'][1]
            s2 = np.exp(self.params['b'][1] * (X[i, 1] - self.params['a'][1]))
            mu_b[i] = 1 / (1 + s2)
            # s1 = np.vstack((s1, s2))
            W[i] = mu_a[i] * mu_b[i]  # s1 * s2 #np.sum(s1, axis=0)

        # sad je W = shape(N, m)

        row_sum = np.sum(W, axis=1, keepdims=True)
        row_sum[abs(row_sum - 1e-6) < 1e-6] = 1e-6
        W_average = W / row_sum

        f = X.dot(self.params['f']) + self.params['r']  # NxM
        # tezine puta f pa suma po njima da se dobije rezultat
        z = W_average.dot(f.T)#tu ne NxN
        output = np.sum(z, axis=0)  # ovo su sada output (N,)

        if mode == 'test':
            return output

        loss, grads = 0.0, {}

        loss, upstream_gradient = quadratic_loss(y=y, o_k=output)

        w_sum_squared = np.sum(W, axis=1)
        dz = upstream_gradient * w_sum_squared
        dp = upstream_gradient * output * X[:, 0]
        dq = upstream_gradient * output * X[:, 1]
        dr = upstream_gradient * output

        # izracunat zi - zj

        dW = np.zeros(shape=(self.num_rules, N))  # transpniraj na kraju da odgovara klasicnom W

        for i in range(self.num_rules):
            s = 0
            for j in range(self.num_rules):
                if i == j: continue

                s += (f[:, i] - f[:, j]) * W[:, j]
            dW[i, :] = s / w_sum_squared

        dW = dW.reshape(N, self.num_rules)

        da1 = upstream_gradient * self.params['b'][0] * dW * mu_a * (1 - mu_a) * mu_b
        db1 = upstream_gradient * dW * (-(x_less_a) * mu_a * (1 - mu_a) * mu_b)

        da2 = upstream_gradient * self.params['b'][1] * dW * mu_b * (1 - mu_b) * mu_a
        db2 = upstream_gradient * dW * (-(y_less_a) * mu_b * (1 - mu_b) * mu_a)



        return loss, grads
