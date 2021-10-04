import numpy as np
import pandas as pd
import os
import sklearn
from csge.csge import CSGERegressor as CSGE

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

class TestPredictions:
    def testPredictProba(self):
        np.random.seed(1337)
        dataset = sklearn.datasets.load_iris()
        X = dataset['data']
        y = dataset['target']
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
        
        d1 = {'kernel':'linear'}
        d2 = {'kernel':'rbf'}
        params = [d1, d2, {}]


        ensemble = CSGE(
        ensembles_types = [sklearn.svm.SVC, sklearn.svm.SVC, sklearn.tree.DecisionTreeClassifier],
        error_function=sklearn.metrics.accuracy_score,
        eta=[3.5, 3.5, 0],
        ensemble_parameters = params,
        probability=True
        )
        
        ensemble.fit(X_train, y_train)
        
        predictions = ensemble.predict(X_test)
        predictions_probabilities = ensemble.predict_proba(X_test)
        assert (np.argmax(predictions_probabilities, axis=1) == predictions.flatten()).all()