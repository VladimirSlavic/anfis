import numpy as np
import math
from network.anfisnetwork import Anfis
from network.controller import Controller

_range = [i for i in range(-4, 5, 1)]
X_train = None
y_train = None

func_ = lambda x, y: ((x - 1) ** 2 + (y + 2) ** 2 - 5 * x * y + 3) * math.cos(x / 5) ** 2

for i in _range:
    for j in _range:
        if X_train is None:
            X_train = np.array([[i, j]])
            y_train = np.array([func_(i, j)])
        else:
            X_train = np.vstack((X_train, np.array([i, j])))
            y_train = np.hstack((y_train, func_(i, j)))

test = [(0.3, 2), (0.5, 0.5), (-1.3, 1.3), (-3.67, 2.15)]

X_test = None
y_test = None

for pair in test:
    if X_test is None:
        X_test = np.array([[pair[0], pair[1]]])
        y_test = np.array([func_(pair[0], pair[1])])
    else:
        X_test = np.vstack((X_test, np.array([pair[0], pair[1]])))
        y_test = np.hstack((y_test, np.array([func_(pair[0], pair[1])])))


model = Anfis(num_rules=5, input_dims=2, output_classes=1)
data = {'X_train': X_train,
        'y_train': y_train,
        'X_val': None,
        'y_val': None}

controller = Controller(model, data, print_every=20, num_epochs=20000, batch_size=10, update_rule='sgd',
                        optim_config={
                            'learning_rate': 0.01
                        })

controller.train()
# mask = np.random.choice(X_train.shape[0], 6)
# X_test = X_train[mask]
# y_test = y_train[mask]

prediction = controller.predict(X_test)
print(prediction)
print(y_test)
