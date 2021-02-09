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
        model_forecast_time_error=RandomForestRegressor,
        eta=[3.5, 3.5, 3.5],
        eps = 0.00000001,
        n_jobs=1,
        leadtime_k=1
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
        self.eps = eps
        self.n_jobs = n_jobs
        self.n_cv_out_of_sample_error = n_cv_out_of_sample_error
        self.ensemble_members = None
        self.error_matrix = None
        self.ensemble_parameters = None
        self.model_forecast_local_error = model_forecast_local_error
        self.model_forecast_time_error = model_forecast_time_error
        self.leadtime_k = leadtime_k
        self.t = None

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
        if len(weights.shape) == 2:
            sum_weights = weights.sum(1).reshape(-1, 1)
            weights /= sum_weights
        else:
            sum_weights = weights.sum(2)#.sum(1).reshape(-1, 1, 1)
            weights /= np.expand_dims(sum_weights, 2)

        return weights

    def _get_global_error(self):
        # self.global_errors = np.ones_like(self.error_matrix) / self.error_matrix.mean(0)
        self.global_errors = (1 / self.error_matrix[:,:,0].mean(0)).reshape(1, -1)

    def _create_error_matrix(self, X, y):
        self.error_matrix = None
        global_errors = []
        num_samples = len(X)
        # TODO what in case of dict for len(self.ensembles)
        self.error_matrix = np.ones((self.leadtime_k, len(self.ensemble_members), num_samples))
        for ens_id, ensemble_member in enumerate(self.ensemble_members):
            for idx, (_, test_index) in enumerate(self.train_test_indexes):
                if self.leadtime_k != 1:
                    preds = ensemble_member[idx].predict(X[test_index], self.t)
                else:
                    preds = ensemble_member[idx].predict(X[test_index])
                for t in range(self.leadtime_k):

                    for sample_id in range(len(test_index)):
                        self.error_matrix[
                            t, ens_id, len(test_index) * idx + sample_id
                        ] = self.error_function(
                            np.array([y[test_index][sample_id][t]]), np.array([preds[sample_id, t]])
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

        self._fit_time_error_forecast(self.t)

        # TODO: implement time dependent weighting
        # TODO: implement soft gating

    def _fit_local_error_forecast(self, X):
        self.local_error_forecaster = []

        for id_ens in range(len(self.ensemble_members)):
            self.local_error_forecaster.append(
                self.model_forecast_local_error().fit(X, self.error_matrix[:, id_ens, 0])
            )

    def _fit_time_error_forecast(self, t):
        self.time_error_forecaster = []
        time_error_matrix = self.error_matrix
        time_error_matrix = time_error_matrix.mean(0).transpose()
        t = np.reshape(t, [-1, 1])
        for id_ens in range(len(self.ensemble_members)):
            self.time_error_forecaster.append(
                self.model_forecast_time_error().fit(t, time_error_matrix[:, id_ens])
            )

    def _fit_out_of_sample_ensembles(self, X, y):
        if self.leadtime_k != 1:
            self.t = np.arange(0, self.leadtime_k, 1)
        else:
            self.t = np.array([0])
        for i, ensemble in enumerate(self.ensembles_types):
            kf = KFold(n_splits=self.n_cv_out_of_sample_error)
            cv_ensembles = []
            for train_index, _ in self.train_test_indexes:
                # TODO: add params
                model = ensemble()
                model = self._assign_params(i, model)
                if self.leadtime_k != 1:
                    model.fit(X[train_index], self.t, y[train_index].ravel())
                else:
                    model.fit(X[train_index], y[train_index].ravel())
                cv_ensembles.append(model)
            self.ensemble_members.append(cv_ensembles)

    def _fit_ensembles_for_prediction(self, X, y):
        self.ensemble_members = []
        for i, ensemble in enumerate(self.ensembles_types):
            model = ensemble()
            model = self._assign_params(i, model)
            if self.leadtime_k != 1:
                model.fit(X, self.t, y.ravel())
            else:
                model.fit(X, y.ravel())
            self.ensemble_members.append(model)

    def _pred_all_ensembles(self, X, t):
        predictions = np.zeros((len(X), len(t), len(self.ensemble_members)))
        for id_em, ensemble_member in enumerate(self.ensemble_members):
            if self.leadtime_k != 1:
                predictions[:, :, id_em] = ensemble_member.predict(X, t)
            else:
                predictions[:, :, id_em] = ensemble_member.predict(X)

        return predictions

    def _pred_local_error(self, X):
        local_errors = np.zeros((len(X), len(self.ensemble_members)))
        for id_em, local_error_member in enumerate(self.local_error_forecaster):
            local_errors[:, id_em] = local_error_member.predict(X)

        return local_errors

    def _pred_time_error(self, t):
        time_errors = np.zeros((len(t), len(self.ensemble_members)))
        for id_em, tim_error_member in enumerate(self.time_error_forecaster):
            time_errors[:, id_em] = tim_error_member.predict(t)

        return time_errors

    def _weight_forecasts(self, X, t, predictions):
        # Normalize the global error of all ensemble members.
        # Apply the softgating function to select the linearity.
        # Shape: [1, len(ensemble_members)]
        normalized_global_error = self._normalize_weighting(self.global_errors)
        normalized_global_error = utils.soft_gating_formular(normalized_global_error, self.eta[0])
        # Predict the local error of all ensemble members for each input separately.
        # Apply the softgating function to select the linearity.
        # Shape: [len(X), len(ensemble_members)]
        self.local_errors = 1 / (self._pred_local_error(X) + self.eps)
        self.local_errors = utils.soft_gating_formular(self.local_errors, self.eta[1])
        self.time_errors = 1 / (self._pred_time_error(t) + self.eps)
        self.time_errors = utils.soft_gating_formular(self.time_errors, self.eta[2])

        # Multiply both error types.
        # This weights the local errors, i.e. the error of a given input, with the average error of the ensemble member
        # over the whole input space it has been trained on.
        # This results in selecting the best best ensemble member for a given input (local space), with respect to
        # its overall performance.
        combined_weighting = self.local_errors * normalized_global_error
        final_weighting = []
        #ToDo: Search better way to create this array
        for i in range(combined_weighting.shape[0]):
            final_weighting.append(combined_weighting[i] * self.time_errors)
        final_weighting = np.array(final_weighting)
        final_weighting = 1 / final_weighting
        #return final_weighting
        self.final_weighting = self._normalize_weighting(final_weighting)
        return (predictions * self.final_weighting).sum(2)#.sum(1)

    def predict(self, X, t=None):
        """
        Predict with all ensemble members, and combine their results according to multiple error types.
        Parameters
        ----------
        X : numpy.array
            shape: [samples, features]

        Returns
        -------
        numpy.array
            shape: [samples, 1]
        """
        if t is None:
            t = np.array([[0]])
        predictions_ensembles = self._pred_all_ensembles(X, t)
        predictions = self._weight_forecasts(X, t, predictions_ensembles)

        return predictions