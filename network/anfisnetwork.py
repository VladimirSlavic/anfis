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

        self.params['a'] = np.array([[1, 1],
                                     [2, 2]])
        self.params['b'] = np.array([[2, 2],
                                     [3, 1]])
        self.params['f'] = np.array([[1, 1],
                                     [1, 2]])
        self.params['r'] = np.array([0, 0])

        # self.params['a'] = np.zeros(shape=(num_rules, input_dims))
        # self.params['b'] = np.zeros(shape=(num_rules, input_dims))
        # self.params['f'] = np.zeros(shape=(input_dims, input_dims))
        # self.params['r'] = np.zeros(shape=(input_dims))
        #
        #
        #
        # for i in range(num_rules):
        #     for j in range(input_dims):
        #         self.params['a' + str(j) + str(i)]
        #         self.params['b' + str(j) + str(i)]

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

        for i in range(N):
            cache_key = 'cache_rule' + str(i)
            cache_func_key = 'cache_func' + str(i)
            cache_sum_key = 'w_sum' + str(i)
            forward_pass_key = "forward_pass_rule" + str(i)

            score = X[i]
            score, self.cache[cache_key] = forward_pass_rules(score, self.params['a'], self.params['b'])
            self.cache[forward_pass_key] = score
            score = np.prod(score, axis=1)  # [w1, w2, w3.....]
            sum = np.sum(score)  # w1 + w2 + w3 + w4
            self.cache[cache_sum_key] = sum
            score /= sum  # w1'. w2'
            W[i] = score.reshape(1, score.shape[0])
            func_res[i], self.cache[cache_func_key] = forward_pass_function(X[i], self.params['f'],
                                                                            self.params['r'])  # 1x2
            output = W[i].dot(func_res[i].T)  # 1x1
            o_k[i] = output

        if mode == 'test':
            return output

        loss, grads = 0.0, {}

        loss, upstream_gradient = quadratic_loss(y=y, o_k=o_k)
        grads['f'] = np.zeros(shape=self.params['f'].shape)
        grads['a'] = np.zeros(shape=self.params['a'].shape)
        grads['b'] = np.zeros(shape=self.params['b'].shape)

        for i in range(N):
            cache_key = 'cache_rule' + str(i)
            cache_func_key = 'cache_func' + str(i)
            cache_sum_key = 'w_sum' + str(i)
            f_key = "f" + str(i)
            r_key = "r" + str(i)
            p_key = "p" + str(i)
            q_key = "q" + str(i)
            forward_pass_key = "forward_pass_rule" + str(i)

            grads[p_key] = upstream_gradient[i] * W * X[:, 0].reshape(-1, 1)
            grads[q_key] = upstream_gradient[i] * W * X[:, 1].reshape(-1, 1)

            grads[r_key] = upstream_gradient[i] * W[i]
            grad_w = np.sum(W * (func_res[i] - func_res))
            grad_w /= (np.sum(W) ** 2)

            grad_a = upstream_gradient[i] * grad_w * (X[i] - self.cache[forward_pass_key]) * self.cache[
                forward_pass_key] * (1 - self.cache[forward_pass_key])

    pass
