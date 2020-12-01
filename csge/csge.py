from sklearn.base import BaseEstimator

from itertools import product

from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize, optimize
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score


class CoopetitiveSoftGatingEnsemble(BaseEstimator):
    def __init__(
        self,
        ensembles,
        error_function=mean_squared_error,
        optimization_method="Newton-CG",
        eta=[3.5, 3.5, 3.5],
        n_jobs=1,
    ):

        self.ensembles = ensembles
        self.error_function = error_function
        self.optimization_method = optimization_method
        self.eta = eta
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.global_errors = []
        for ensemble in self.ensembles:
            scores = cross_val_score(ensemble(), X, y, cv=3)
            self.global_errors.append(scores)

    def predict(self, X):
        return self.global_errors
