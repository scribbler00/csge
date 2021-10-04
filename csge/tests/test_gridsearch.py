import numpy as np
import pandas as pd
import os
import sklearn
from csge.csge import CSGERegressor as CSGE

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


import random

class TestGridsearch:
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

    def testGridsearch(self):
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

        eta=[3.5, 3.5, 100]
        f1 = sklearn.linear_model.LinearRegression
        f2 = sklearn.linear_model.Ridge
        f3 = sklearn.linear_model.Lasso
        f4 = sklearn.tree.DecisionTreeRegressor

        ensemble_csge = CSGE(
            #ensembles = [LinearRegression, sklearn.svm.SVR, tree.DecisionTreeRegressor],
            ensembles_types = [f1, f2, f3, f4],
            error_function=mean_absolute_error,
            #eta=[3.5, 3.5, 0],
            eta=eta,
            leadtime_k = leadtime,
            #type='regression',
            )
        
        etas = []
        etas.append([3.5, 3.5, 3.5])
        etas.append([3.5, 1, 3.5])
        etas.append([3.5, 0, 3.5])
        etas.append([3.5, 3.5, 1])
        etas.append([3.5, 3.5, 0])
        etas.append([1, 3.5, 3.5])
        etas.append([1, 1, 3.5])
        etas.append([1, 0, 3.5])
        etas.append([1, 3.5, 1])
        etas.append([1, 3.5, 0])
        etas.append([0, 3.5, 3.5])
        etas.append([0, 1, 3.5])
        etas.append([0, 0, 3.5])
        etas.append([0, 3.5, 1])
        etas.append([0, 3.5, 0])

        parameters = {}
        parameters['eta'] = etas

        clf = GridSearchCV(ensemble_csge, parameters, scoring='neg_mean_absolute_error')
        clf.fit(X, y)
        res_df = pd.DataFrame(clf.cv_results_)

        assert len(res_df) == len(etas)