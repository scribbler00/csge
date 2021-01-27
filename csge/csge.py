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
        """

        Parameters
        ----------
        ensembles : list of classes
            all classes must have .fit() and .predict()
        error_function : function
            the metric/function used to calculate the error for a given input
        optimization_method : function
            currently not used
        n_cv_out_of_sample_error : int
            defines the amount of chunks the train data is to be split
        model_forecast_local_error : function
            method used to calculate the local error
        eta : int
            defines the linearity of the weighting. Greater values result in a more selective sensemble
        n_jobs : int
            currently not used
        """
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
        """
        Calculate the sum of a given list of weights
        Parameters
        ----------
        weights : list of numpy.array
            the weights of all ensemble members that are to be summed
        Returns
        -------
        numpy.array
            sum of the given weights
        """
        #ToDo: leadtime_k, number_of_targets not defined
        sum_of_all_weights = np.full((self.leadtime_k, self.number_of_targets), 0.0)

        for weight in self.weights:
            sum_of_all_weights = np.add(sum_of_all_weights, weight.weight)

        return sum_of_all_weights

    def set_ensemble_parameters(self, parameters=[]):
        """
        Use the given list to set the ensemble member's parameters
        Parameters
        ----------
        parameters : list of dictionaries
            each dictionary contains pairs of 'parameter: value'
            order is equal to self.ensembles_types
        Returns
        -------

        """
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
            try:
                model.set_params(**self.ensemble_parameters[index])
            except ValueError:
                print(f"Unable to assign parameters to {model}.\n")
                print(f"Given parameters: {list(self.ensemble_parameters[index].keys())}.\n")
                print(f"Valid parameters: {list(model.get_params().keys())}.\n")
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
        """
        Fit the ensemble members to the given input data

        First: create multiple objects of each ensemble member, and fit each to a subset of the input.
        Second: use the ensemble members to calculate the global error
        Third: create one object for each ensemble type, and fit it to the whole dataset
        Fourth: create one predictor for each ensemble type, and fit it to predict the local error for a given input

        Parameters
        ----------
        X : numpy.array
            shape: [samples, features]
        y : numpy.array
            shape: [samples, 1]

        Returns
        -------

        """
        # TODO: add time dependent weighting
        self.ensemble_members = []
        num_samples = len(X)

        kf = KFold(n_splits=self.n_cv_out_of_sample_error)
        self.train_test_indexes = [
            (train_index, test_index) for train_index, test_index in kf.split(X)
        ]

        # fit ensemble members to create out of sample errors
        self._fit_out_of_sample_ensembles(X, y)

        # same as the local error
        self._create_error_matrix(X, y)
        self._get_global_error()

        # refit ensemble members on complete data
        self._fit_ensembles_for_prediction(X, y)

        # fit to pred local error
        self._fit_local_error_forecast(X)

        # TODO: implement time dependent weighting
        # TODO: implement soft gating

    def _fit_local_error_forecast(self, X):
        self.local_error_forecaster = []

        for id_ens in range(len(self.ensemble_members)):
            self.local_error_forecaster.append(
                self.model_forecast_local_error().fit(X, self.error_matrix[:, id_ens])
            )

    def _fit_out_of_sample_ensembles(self, X, y):
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

    def _fit_ensembles_for_prediction(self, X, y):
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
        # Normalize the global error of all ensemble members.
        # Apply the softgating function to select the linearity.
        # Shape: [1, len(ensemble_members)]
        normalized_global_error = self._normalize_weighting(self.global_errors)
        normalized_global_error = utils.soft_gating_formular(normalized_global_error, self.eta[0])

        # Predict the local error of all ensemble members for each input separately.
        # Apply the softgating function to select the linearity.
        # Shape: [len(X), len(ensemble_members)]
        self.local_errors = 1 / self._pred_local_error(X)
        self.local_errors = utils.soft_gating_formular(self.local_errors, self.eta[1])


        # Multiply both error types.
        # This weights the local errors, i.e. the error of a given input, with the average error of the ensemble member
        # over the whole input space it has been trained on.
        # This results in selecting the best best ensemble member for a given input (local space), with respect to
        # its overall performance.
        final_weighting = self.local_errors * normalized_global_error
        self.final_weighting = self._normalize_weighting(final_weighting)

        return (predictions * self.final_weighting).sum(1)

    def predict(self, X):
        """
        Predict with all enselbme members, and combine their results according to multiple error types.
        Parameters
        ----------
        X : numpy.array
            shape: [samples, features]

        Returns
        -------
        numpy.array
            shape: [samples, 1]
        """
        predictions_ensembles = self._pred_all_ensembles(X)
        predictions = self._weight_forecasts(X, predictions_ensembles)

        return predictions