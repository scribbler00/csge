import numpy as np
import pandas as pd
import os
import sklearn
from csge.csge import CoopetitiveSoftGatingEnsemble as CSGE

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

class TestDateloader:
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
    def test_timeseries_loader(self):
        self.setUp()
        leadtime = 10
        power_data = self.power_data_all.sample(n=1000, random_state = self.seed)
        input_df = power_data.loc[:, power_data.columns != 'PowerGeneration']
        target_df = power_data['PowerGeneration']
        

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
            ensembles = [f1, f2, f3, f4],
            error_function=mean_absolute_error,
            eta=eta,
            leadtime_k = leadtime,
            type='regression',
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
        assert np.mean(errors) < 0.3