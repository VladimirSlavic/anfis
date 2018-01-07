import numpy as np

from network.network_functions import quadratic_loss
from optimization import optimization


class Controller:
    def __init__(self, model, data, **kwargs) -> None:
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.epoch = 0

        num_val_examples = self.X_val.shape[0] if self.X_val is not None else None

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 20)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', self.X_train.shape[0])
        self.num_val_samples = kwargs.pop('num_val_samples', num_val_examples)

        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        if not hasattr(optimization, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optimization, self.update_rule)
        self._reset()

    def _reset(self):
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        N = self.X_train.shape[0]
        batch_mask = np.random.choice(N, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        loss, grads = self.model.loss(X_batch, y_batch)

        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def accuracy(self, X, y, num_samples=None, batch_size=20):

        if X is None:
            return

        N = self.X_train.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        num_batches = N // batch_size
        if N % num_samples != 0:
            num_batches += 1
        sum_loss = 0

        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            predictions = self.model.loss(X[start:end])
            loss, grad = quadratic_loss(y=y[start:end], o_k=predictions)
            sum_loss += loss
        # return average loss
        return sum_loss / (num_batches * batch_size)

    def predict(self, X):
        predictions = self.model.loss(X)
        return predictions

    def train(self):
        N = self.X_train.shape[0]
        iterations_per_epoch = max(N // self.batch_size, 1)
        num_iterations = iterations_per_epoch * self.num_epochs

        for t in range(num_iterations):

            self._step()
            new_epoch = (t + 1) % iterations_per_epoch == 0
            if new_epoch:
                self.epoch += 1

            first_iteration = (t == 0)
            last_iteration = (t == num_iterations - 1)
            should_print = (t % self.print_every) == 0

            if first_iteration or last_iteration or should_print:

                train_loss_average = self.accuracy(self.X_train, self.y_train, num_samples=N)
                result = 'training loss average is: {}'.format(train_loss_average)
                if self.X_val is not None:
                    validation_loss_average = self.accuracy(self.X_val, self.y_val, num_samples=self.X_val.shape[0])
                    result += ', validation loss average is: {}'.format(validation_loss_average)
                print('epoch: {}, {}'.format(self.epoch, result))

