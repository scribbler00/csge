from sklearn.base import BaseEstimator

from itertools import product

from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize, optimize
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.exceptions import NotFittedError


class CoopetitiveSoftGatingEnsemble(BaseEstimator):
    def __init__(
        self,
        ensembles,
        error_function=mean_squared_error,
        optimization_method="Newton-CG",
        n_cv_out_of_sample_error=3,
        eta=[3.5, 3.5, 3.5],
        n_jobs=1,
    ):
        # TODO: ensembles as list or dict,in case of list default parameters, otherwise those provided by the dict
        # TODO: add check if parameters for ensemble member are correct
        # TODO: add function to get name from ensemble member

        self.ensembles = ensembles
        self.error_function = error_function
        self.optimization_method = optimization_method
        self.eta = eta
        self.n_jobs = n_jobs
        self.n_cv_out_of_sample_error = n_cv_out_of_sample_error
        self.ensemble_members = None

    def _get_global_error(self, X, y):
        if self.ensemble_members is None:
            raise NotFittedError("Ensemble members are not fitted.")

        global_errors = []
        num_samples = len(X)

        for ensemble_member in self.ensemble_members:
            cur_ensemble_member_error = 0
            for idx, (_, test_index) in enumerate(self.train_test_indexes):
                pred = ensemble_member[idx].predict(X[test_index])
                cur_ensemble_member_error += self.error_function(y[test_index], pred)

            cur_ensemble_member_error /= self.n_cv_out_of_sample_error
            global_errors.append(cur_ensemble_member_error)

        self.global_errors = np.hstack(
            [
                np.ones(num_samples).reshape((-1, 1)) * np.mean(val)
                for val in global_errors
            ]
        )

    def fit(self, X, y):
        # TODO: add time dependent weighting
        self.ensemble_members = []
        num_samples = len(X)

        kf = KFold(n_splits=self.n_cv_out_of_sample_error)
        self.train_test_indexes = [
            (train_index, test_index) for train_index, test_index in kf.split(X)
        ]

        # fit ensemble members
        for ensemble in self.ensembles:
            kf = KFold(n_splits=self.n_cv_out_of_sample_error)
            cv_ensembles = []
            for train_index, _ in self.train_test_indexes:
                # TODO: add params
                model = ensemble().fit(X[train_index], y[train_index].ravel())
                cv_ensembles.append(model)
            self.ensemble_members.append(cv_ensembles)

    def predict(self, X):
        return self.ensemble_members