import numpy as np

from network.anfisnetwork import Anfis
from network.controller import Controller

X = np.array([[1, -3],
              [1, -2],
              [1, -1],
              [1, 0],
              [1, 1],
              [1, 2],
              [1, 3],
              [1, 4],
              [2, -3],
              [2, -2],
              [2, -1],
              [2, 0],
              [2, 1],
              [2, 2],
              [2, 3],
              [2, 4],
              [3, -3],
              [3, -2],
              [3, -1],
              [3, 0],
              [3, 1],
              [3, 2],
              [3, 3],
              [3, 4],
              [4, -3],
              [4, -2],
              [4, -1],
              [4, 0],
              [4, 1],
              [4, 2],
              [4, 3],
              [4, 4]
              ])

# print(X[:,0])
y = np.zeros(shape=(X.shape[0],))
for i in range(X.shape[0]):
    y[i] = X[i, 0] ** 2 + X[i, 1] ** 2

#
# xx = np.array([1, 2, 3, 4])
# yy = np.array([1, 4, 6, 4])
# print(xx/yy)

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
# self, num_rules=2, input_dims=2, output_classes=1, reg=0.0, weight_dev=1e-2, dtype=np.float64,
#                  **kwargs):
model = Anfis(num_rules=2, input_dims=2, output_classes=1)
data = {'X_train': X,
        'y_train': y,
        'X_val': None,
        'y_val': None}
# model = NeuralNet(hidden_dims=self.hidden_layer_dimens, input_dims=self.input_dim, num_classes=self.output_dimension,
#                           loss_type=self.loss, function=self.activation, dtype=np.float128)
#
# self.solver = Controller(model, small_data,
#                          print_every=1000, num_epochs=20000, batch_size=50,
#                          update_rule='sgd',
#                          optim_config={
#                              'learning_rate': learning_rate,
#                          })

controller = Controller(model, data, print_every=20, num_epochs=20000, batch_size=10, update_rule='sgd',
                        optim_config={
                            'learning_rate': 0.001
                        })

controller.train()

X_test = np.array([
    [1, -3],
    [1, -2],
    [1, -1],
    [1, 0],
    [1, 1],
    [1, 2],
])
prediction = controller.predict(X_test)
print(prediction)
print(y[:6])
