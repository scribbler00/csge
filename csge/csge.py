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
        self.error_matrix = None

    def _get_global_error(self):
        self.global_errors = np.ones_like(self.error_matrix) * self.error_matrix.mean(
            1
        ).reshape(-1, 1)

    def _create_error_matrix(self, X, y):
        self.error_matrix = None
        global_errors = []
        num_samples = len(X)
        # TODO what in case of dict for len(self.ensembles)
        self.error_matrix = np.ones((len(self.ensemble_members), num_samples))

        for ens_id, ensemble_member in enumerate(self.ensemble_members):
            for idx, (_, test_index) in enumerate(self.train_test_indexes):
                preds = ensemble_member[idx].predict(X[test_index]).reshape(-1, 1)
                for sample_id in range(len(test_index)):
                    self.error_matrix[
                        ens_id, sample_id * idx + sample_id
                    ] = self.error_function(
                        np.array(y[test_index][sample_id]), np.array(preds[sample_id])
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

        # TODO: do we need to refit on complete data?

        # same as the local error
        self._create_error_matrix(X, y)
        self._get_global_error()

        # TODO: soft max
        # TODO: fit to pred local error
        # TODO: weighting

    def predict(self, X):
        return self.ensemble_members