import numpy as np
from functions import funcs
from functions.funcs import sigmoid
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

        self.params['a'] = np.random.uniform(low=-1, high=1, size=(input_dims, num_rules))
        self.params['b'] = np.random.uniform(low=-1, high=1, size=(input_dims, num_rules))
        self.params['f'] = np.random.uniform(low=-1, high=1, size=(input_dims, num_rules))
        self.params['r'] = np.zeros(shape=num_rules)

        if not hasattr(funcs, self.rule_function):
            raise ValueError('Invalid rule function')
        self.rule_function = getattr(funcs, self.rule_function)

    def loss(self, X, y=None):

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        self.cache = {}

        # forwardpass
        N = X.shape[0]
        W = np.zeros(shape=(N, self.num_rules))
        mu_a = np.zeros(shape=(N, self.num_rules))
        mu_b = np.zeros(shape=(N, self.num_rules))
        x_less_a = np.zeros(shape=(N, self.num_rules))
        y_less_a = np.zeros(shape=(N, self.num_rules))

        # forward pass calculation

        np.seterr(all='raise')
        for i in range(N):
            entry = (X[i, 0] - self.params['a'][0])
            x_less_a[i] = entry
            entry = self.params['b'][0] * entry
            sig_calc = sigmoid(entry)
            mu_a[i] = sig_calc

            entry = X[i, 1] - self.params['a'][1]
            y_less_a[i] = entry
            entry = self.params['b'][1] * entry
            sig_calc = sigmoid(entry)
            mu_b[i] = sig_calc
            W[i] = mu_a[i] * mu_b[i]  # s1 * s2 #np.sum(s1, axis=0)

        # sad je W = shape(N, m)

        row_sum = np.sum(W, axis=1, keepdims=True)
        row_sum[abs(row_sum - 1e-6) <= 1e-6] = 1e-6
        W_average = W / row_sum

        f = X.dot(self.params['f']) + self.params['r']  # NxM
        # tezine puta f pa suma po njima da se dobije rezultat
        z = W_average * f
        output = np.sum(z, axis=1)  # ovo su sada output (N,)

        if mode == 'test':
            return output

        loss, grads = 0.0, {}

        loss, upstream_gradient = quadratic_loss(y=y, o_k=output)

        upstream_gradient_reshaped = upstream_gradient.reshape(1, N)

        x_reshaped = X[:, 0].reshape(N, 1)
        y_reshaped = X[:, 1].reshape(N, 1)

        w_sum_squared = np.sum(W, axis=1)**2
        #dz = upstream_gradient_reshaped.dot(W_average)
        dp = upstream_gradient_reshaped.dot(x_reshaped * W_average)  # (x_reshaped.dot(upstream_gradient_reshaped)).dot(W_average)
        dq = upstream_gradient_reshaped.dot(y_reshaped * W_average)
        dr = upstream_gradient_reshaped.dot(W_average).reshape(self.num_rules, )
        grads['r'] = dr
        f_gradient = np.vstack((dp, dq))
        grads['f'] = f_gradient

        dW = np.zeros(shape=(self.num_rules, N))  # transpniraj na kraju da odgovara klasicnom W

        for i in range(self.num_rules):
            s = 0
            for j in range(self.num_rules):
                if i == j: continue
                # test1 = z[:, i]
                # test2 = z[:, j]
                # test3 = W[:, j]
                # test4 = z[:, i] - z[:, j]
                # test5 = (z[:, i] - z[:, j]) * W[:, j]
                # test6 = W[:, j]
                s += (z[:, i] - z[:, j]) * W[:, j] #<- Z, a ne F kretenu
            dW[i, :] = s / w_sum_squared

        dW = dW.T

        da1 = np.dot(upstream_gradient_reshaped, (self.params['b'][0] * dW * mu_a * (1 - mu_a) * mu_b))  # 1xm
        db1 = np.dot(upstream_gradient_reshaped, (dW * (-x_less_a * mu_a * (1 - mu_a) * mu_b)))

        da2 = np.dot(upstream_gradient_reshaped, (self.params['b'][1] * dW * mu_b * (1 - mu_b) * mu_a))
        db2 = np.dot(upstream_gradient_reshaped, (dW * (-y_less_a * mu_b * (1 - mu_b) * mu_a)))

        grads['a'] = np.vstack((da1, da2))
        grads['b'] = np.vstack((db1, db2))

        return loss, grads
