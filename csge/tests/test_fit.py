

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

from csge.csge import CSGERegressor as CSGE


class f1(BaseEstimator):
    def __init__(self):
        BaseEstimator.__init__(self)
        self._estimator_type = 'regressor'
    def fit(self, X, y):
        return
    def predict(self, X):
        return np.reshape(np.sin(X), [-1, 1])

class f2(BaseEstimator):
    def __init__(self):
        BaseEstimator.__init__(self)
        self._estimator_type = 'regressor'
    def fit(self, X, y):
        return
    def predict(self, X):
        return np.reshape(np.sin(X) + 10, [-1, 1])
def g(x, offset):
    return np.sin(x) + offset

x_axis = np.arange(-2*np.pi, 2*np.pi, 0.1)
x_axis = np.reshape(x_axis, [-1, 1])
targets = g(x_axis, 4)
targets = np.reshape(targets, [-1, 1])

model = CSGE([('f1', f1()), ('f2', f2())], error_function=mean_absolute_error)
model.eta = [1, 0, 0]
model.fit(x_axis, targets)

y0 = model.predict(x_axis)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(x_axis, targets, '-', color='gray', label='target')
plt.plot(x_axis, y0, '--r', label='prediction')
plt.grid(True)
plt.legend()
plt.show()


assert np.isclose(y0, targets).all()