import numpy as np
import pandas as pd
import os
import sklearn
from csge.csge import CoopetitiveSoftGatingEnsemble as CSGE

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

class TestFeatureSplitter:
    def setUp(self):
        self.seed = 1337
        self.leadtime = 10
        self.path = 'datasets/DAF_ICON_Synthetic_Wind_Power_processed/'
        power_data_all = pd.DataFrame()
        
        for path_file in os.listdir(self.path):
            
            if path_file[-2:] != 'h5':
                continue
            with pd.HDFStore(self.path + path_file) as store:
                data_power = store.get('powerdata')
            power_data_all = pd.concat([power_data_all, data_power])
        self.power_data_all = power_data_all.sort_index()

    def test_timelag(self):
        class f1:
            def __init__(self):
                self._estimator_type = 'regressor'
            def fit(self, X, y):
                return
            def predict(self, X):
                return np.reshape(np.sin(X), [-1, 1])
        class f2:
            def __init__(self):
                self._estimator_type = 'regressor'
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
        model = CSGE([f1, f2], error_function=mean_absolute_error, model_forecast_local_error=RandomForestRegressor, time_lag=3)
        model.eta = [0, 3.5, 0]
        model.fit(x_axis, targets)
        y0 = model.predict(x_axis)
        assert np.abs(np.mean(y0 - targets)) < 0.2

    def test_timelag_leadtime(self):
        class f1:
            def __init__(self):
                self._estimator_type = 'regressor'
            def fit(self, X, y):
                return
            def predict(self, X):
                res = np.sin(X).reshape(-1)
                return res
        class f2:
            def __init__(self):
                self._estimator_type = 'regressor'
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
        model = CSGE([f1, f2], error_function=mean_absolute_error, time_lag=3)
        model.eta = [0, 0, 3.5]
        model.leadtime_k = 6
        model.fit(x_axis_csge, targets_csge)
        y0 = model.predict(x_axis_csge).reshape(-1, 6)
        assert np.isclose(y0, targets).all()


    def test_timelag_featuresplit_leadtime(self):
        self.setUp()
        leadtime = 10
        power_data = self.power_data_all.sample(n=1000, random_state = self.seed)
        input_df = power_data.loc[:, power_data.columns != 'PowerGeneration']
        target_df = power_data['PowerGeneration']

        feature_indizes = [
            [0,1,4,7],
            [1,5,8,13,14,15,16],
            [20,21,22,23],
            [1,2,3,4,5,6,7,8,9,10]
        ]
        

        X = input_df.to_numpy()
        y = target_df.to_numpy()
        X_res = np.zeros([int(leadtime * np.floor(X.shape[0] / leadtime)), X.shape[1] + 1])
        for ix in range(int(np.floor(X.shape[0] / leadtime))):
            for jx in range(leadtime):
                X_res[ix * leadtime + jx] = np.append(X[ix * leadtime], jx)
        X = X_res
        y = y[:X_res.shape[0]]

        kf = sklearn.model_selection.KFold(n_splits=10)
        kf.get_n_splits(X)

        errors = []


        eta=[3.5, 3.5, 3.5]
        f1 = sklearn.linear_model.LinearRegression
        f2 = sklearn.linear_model.Ridge
        f3 = sklearn.linear_model.Lasso
        f4 = sklearn.tree.DecisionTreeRegressor
        ensemble_csge = CSGE(
            ensembles_types = [f1, f2, f3, f4],
            feature_indizes=feature_indizes,
            error_function=mean_absolute_error,
            eta=eta,
            leadtime_k = leadtime,
            time_lag=3
            )

        for train_index, test_index in kf.split(X):
            X_train = X[train_index]
            y_train = y[train_index]

            X_test = X[test_index]
            y_test = y[test_index]
            ensemble_csge.fit(X_train, y_train)
            pred = ensemble_csge.predict(X_test)
            error = [mean_absolute_error(pred.flatten()[ld::leadtime], y_test[ld::leadtime]) for ld in range(leadtime)]
            errors.append(error)
        print(np.mean(errors))
        assert np.mean(errors) < 0.3