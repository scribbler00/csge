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
from sklearn.ensemble import RandomForestRegressor

from csge import utils


class CoopetitiveSoftGatingEnsemble(BaseEstimator):
    def __init__(
        self,
        ensembles,
        error_function=mean_squared_error,
        optimization_method="Newton-CG",
        n_cv_out_of_sample_error=3,
        model_forecast_local_error=RandomForestRegressor,
        eta=[3.5, 3.5, 3.5],
        n_jobs=1,
    ):
        # TODO: ensembles as list or dict,in case of list default parameters, otherwise those provided by the dict
        # TODO: add check if parameters for ensemble member are correct
        # TODO: add function to get name from ensemble member

        self.ensembles_types = ensembles
        self.error_function = error_function
        self.optimization_method = optimization_method
        self.eta = eta
        self.n_jobs = n_jobs
        self.n_cv_out_of_sample_error = n_cv_out_of_sample_error
        self.ensemble_members = None
        self.error_matrix = None
        self.ensemble_parameters = None
        self.model_forecast_local_error = model_forecast_local_error

    def sum_all_weights(self, weights):
        sum_of_all_weights = np.full((self.leadtime_k, self.number_of_targets), 0.0)

        for weight in self.weights:
            sum_of_all_weights = np.add(sum_of_all_weights, weight.weight)

        return sum_of_all_weights

    def set_ensemble_parameters(self, parameters = []):
        try:
            assert len(parameters) == len(self.ensembles_types)
            for params in parameters:
                assert type(params) == dict
        except AssertionError:
            print("Invalid amount of parameters (dim 0 has to match 'ensembles_types')")
            return
        self.ensemble_parameters = parameters

    def _assign_params(self, index, model):
        if self.ensemble_parameters is not None:
            model.set_params(**self.ensemble_parameters[index])
        return model


    def _normalize_weighting(self, weights):
        sum_weights = weights.sum(1).reshape(-1, 1)

        return weights / sum_weights

    def _get_global_error(self):
        # self.global_errors = np.ones_like(self.error_matrix) / self.error_matrix.mean(0)
        self.global_errors = (1 / self.error_matrix.mean(0)).reshape(1, -1)

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
        self.error_matrix = self.error_matrix.transpose()

    def fit(self, X, y):
        # TODO: add time dependent weighting
        self.ensemble_members = []
        num_samples = len(X)

        kf = KFold(n_splits=self.n_cv_out_of_sample_error)
        self.train_test_indexes = [
            (train_index, test_index) for train_index, test_index in kf.split(X)
        ]

        # fit ensemble members to create out of sample errors
        self.fit_out_of_sample_ensembles(X, y)

        # same as the local error
        self._create_error_matrix(X, y)
        self._get_global_error()

        # refit ensemble members on complete data
        self.fit_ensembles_for_prediction(X, y)

        # fit to pred local error
        self.fit_local_error_forecast(X)

        # TODO: implement time dependent weighting
        # TODO: implement soft gating

    def fit_local_error_forecast(self, X):
        self.local_error_forecaster = []

        for id_ens in range(len(self.ensemble_members)):
            self.local_error_forecaster.append(
                self.model_forecast_local_error().fit(X, self.error_matrix[:, id_ens])
            )

    def fit_out_of_sample_ensembles(self, X, y):
        for i, ensemble in enumerate(self.ensembles_types):
            kf = KFold(n_splits=self.n_cv_out_of_sample_error)
            cv_ensembles = []
            for train_index, _ in self.train_test_indexes:
                # TODO: add params
                model = ensemble()
                model = self._assign_params(i, model)
                model.fit(X[train_index], y[train_index].ravel())
                cv_ensembles.append(model)
            self.ensemble_members.append(cv_ensembles)

    def fit_ensembles_for_prediction(self, X, y):
        self.ensemble_members = []
        for i, ensemble in enumerate(self.ensembles_types):
            model = ensemble()
            model = self._assign_params(i, model)
            model.fit(X, y.ravel())
            self.ensemble_members.append(model)

    def _pred_all_ensembles(self, X):
        predictions = np.zeros((len(X), len(self.ensemble_members)))
        for id_em, ensemble_member in enumerate(self.ensemble_members):
            predictions[:, id_em] = ensemble_member.predict(X)

        return predictions

    def _pred_local_error(self, X):
        local_errors = np.zeros((len(X), len(self.ensemble_members)))
        for id_em, local_error_member in enumerate(self.local_error_forecaster):
            local_errors[:, id_em] = local_error_member.predict(X)

        return local_errors

    def _weight_forecasts(self, X, predictions):
        normalized_global_error = self._normalize_weighting(self.global_errors)
        normalized_global_error = utils.soft_gating_formular(normalized_global_error, self.eta[0])

        self.local_errors = 1 / self._pred_local_error(X)
        self.local_errors = utils.soft_gating_formular(self.local_errors, self.eta[1])

        final_weighting = self.local_errors * normalized_global_error
        self.final_weighting = self._normalize_weighting(final_weighting)

        return (predictions * self.final_weighting).sum(1)

    def predict(self, X):
        predictions_ensembles = self._pred_all_ensembles(X)
        predictions = self._weight_forecasts(X, predictions_ensembles)

        return predictions