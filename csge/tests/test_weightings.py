import numpy as np
import pandas as pd
import os
import sklearn
from csge.csge import CoopetitiveSoftGatingEnsemble as CSGE

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

class TestBasicRegression:
    def testGlobalWeighting(self):

        class f1:
            def fit(self, X, y):
                return
            def predict(self, X):
                return np.reshape(np.sin(X), [-1, 1])
        class f2:
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

        model = CSGE([f1, f2], error_function=mean_absolute_error)
        model.eta = [1, 0, 0]
        model.fit(x_axis, targets)
        
        y0 = model.predict(x_axis)
        assert np.isclose(y0, targets).all()
    
    def testLocalWeighting(self):
        class f1:
            def fit(self, X, y):
                return
            def predict(self, X):
                return np.reshape(np.sin(X), [-1, 1])
        class f2:
            def fit(self, X, y):
                return
            def predict(self, X):
                return np.reshape(np.sin(X) + 10, [-1, 1])
        def g(x):
            res = []
            for val in x:
                offset = 0
                if val<= 15 and val>10:
                    offset = 10
                res.append(np.sin(val) + offset)
            return np.array(res)
        x_axis = np.arange(0, 20, 0.1)
        x_axis = np.reshape(x_axis, [-1, 1])
        targets = g(x_axis)
        targets = np.reshape(targets, [-1, 1])
        model = CSGE([f1, f2], error_function=mean_absolute_error, model_forecast_local_error=RandomForestRegressor)
        model.eta = [0, 3.5, 0]
        model.fit(x_axis, targets)
        y0 = model.predict(x_axis)
        assert np.isclose(y0, targets, atol=10).all()
    
    def testTimeWeighting(self):
        class f1:
            def fit(self, X, y):
                return
            def predict(self, X):
                res = np.sin(X).reshape(-1)
                return res
        class f2:
            def fit(self, X, y):
                return
            def predict(self, X):
                res = (np.sin(X) + 10).reshape(-1)
                return res
        def g(x, t):
            res = np.zeros([len(x), len(t)])
            for i, val_x in enumerate(x):
                for j, val_t in enumerate(t):
                    res_val = np.sin(val_x)
                    if val_t >= 3:
                        res_val += 10
                    res[i, j] = res_val
            return res
        x_axis = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
        x_axis = np.reshape(x_axis, [-1, 1])
        t_axis = np.arange(0, 6, 1)
        t_axis = np.reshape(t_axis, [-1, 1])
        targets = g(x_axis, t_axis)

        x_axis_csge = x_axis.repeat(6, axis=1).reshape(-1, 1)
        targets_csge = targets.reshape(-1, 1)
        model = CSGE([f1, f2], error_function=mean_absolute_error)
        model.eta = [0, 0, 3.5]
        model.leadtime_k = 6
        model.fit(x_axis_csge, targets_csge)
        y0 = model.predict(x_axis_csge)
        assert np.isclose(y0, targets).all()

