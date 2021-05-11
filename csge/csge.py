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
        ensembles: list,
        error_function = mean_squared_error,
        optimization_method = "Newton-CG",
        n_cv_out_of_sample_error: int = 3,
        model_forecast_local_error = RandomForestRegressor,
        eta: list = [3.5, 3.5, 3.5],
        eps: float = 0.00000001,
        n_jobs: int = 1,
        leadtime_k: int = 1,
        #ToDo: Replace type by selction of 1-
        type: str = 'regression',
        ensemble_parameters: list = [],
        probability: bool = False
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
        eta : list of int
            defines the linearity of the weightings. Greater values result in a more selective sensemble
            the first, second and third elemend adjust global, local, and time-dependent weighting, respectively
        eps : float
            extremely small value to prevent division by 0
        n_jobs : int
            currently not used
        leadtime_k : int
            the maximum amount of time steps that are to be considered
        type : string
            defined whether the CSGE is used to regression or classification
        ensemble_parameters : list
            a list containing a dictionary of parameters for each ensemble member
        probability : True
            determines whether `predict_proba` is enabled for classification
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
        self.model_forecast_local_error = model_forecast_local_error
        self.leadtime_k = leadtime_k
        self.type = type
        self.probability = probability
        
        self.ensemble_members = None
        self.error_matrix = None
        self.ensemble_parameters = None
        self.should_fit=True
        self.t = None
        self.start_indizes = None
        self.flatten_indizes = None
        
        if ensemble_parameters != []:
            self._set_ensemble_parameters(ensemble_parameters)

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
        #ToDo: Gridsearch to test viability
        sum_of_all_weights = np.full((self.leadtime_k, self.number_of_targets), 0.0)

        for weight in self.weights:
            sum_of_all_weights = np.add(sum_of_all_weights, weight.weight)

        return sum_of_all_weights

    def set_ensembles(self, ensembles: list):
        """
        Replace the ensemble members of the class by a list of manually selected, pretrained ensembles

        Parameters
        ----------
        ensembles : list
            contains an object for each ensemble type
        """
        # duplicate the ensemble member to create the error_matrix
        self.ensemble_members=[[ensemble_member] * self.n_cv_out_of_sample_error for ensemble_member in ensembles]
        self.should_fit = False

    def _set_ensemble_parameters(self, parameters: list):
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
    
    def _assign_params(self, index: int, model):
        """
        Assign the given parameters to a specific model
        Parameters
        ----------
        index : int
            index of the parameters of the ensemble member in the list `self.ensemble_parameters`
        model : object
            the ensemble member whose parameters are to be set

        Returns
        -------
        object
            ensemble member with set parameters
        """
        if self.ensemble_parameters is not None:
            try:
                model.set_params(**self.ensemble_parameters[index])
            except ValueError:
                print(f"Unable to assign parameters to {model}.\n")
                print(f"Given parameters: {list(self.ensemble_parameters[index].keys())}.\n")
                print(f"Valid parameters: {list(model.get_params().keys())}.\n")
        return model

    def _transform_time_matrix(self, ts_idx: np.ndarray):
        self.start_indizes = [0]
        self.flatten_indizes = []
        ts_id_old = -1
        row_id = 0
        coords = []
        for i, ts_id in enumerate(ts_idx):
            if ts_id <= ts_id_old:
                self.start_indizes.append(i)
                row_id += 1
            ts_id_old = ts_id
            self.flatten_indizes.append(row_id * self.leadtime_k + ts_id)

            coords.append([row_id, ts_id])
        return coords

    def _normalize_weighting(self, weights: np.ndarray):
        """
        Normalize the given matrix over all ensemble members
        Parameters
        ----------
        weights : numpy.array
            two/three-dimensional array, where the last dimension is ensemble_members

        Returns
        -------
        numpy.array
            normalized array
        """
        if len(weights.shape) == 2:
            sum_weights = weights.sum(1).reshape(-1, 1)
            weights /= sum_weights
        else:
            sum_weights = weights.sum(2)
            weights /= np.expand_dims(sum_weights, 2)

        return weights

    def _get_global_error(self):
        """
        Average the error matrix over the x-axis to generate a global error for a given ensemble member
        Returns
        -------

        """
        # the slicing is due to this error not being time-dependent, thus only one index of this axis suffices
        self.global_errors = (1 / self.error_matrix[:,:,0].mean(0)).reshape(1, -1)

    def _create_error_matrix(self, X: np.ndarray, target_ids: np.ndarray, y: np.ndarray):
        """
        Create a matrix containing errors for all x-indizes, timestamps, and ensemble-members
        Parameters
        ----------
        X : numpy.array
            shape: [samples, features]
            contains the input data
        target_ids : numpy.ndarray
            shape: [samples, 1]
            one entry for each sample, determining the forecast horizon of it
        y : numpy array
            shape: [samples, targets]
            contains the target data

        Returns
        -------

        """
        self.error_matrix = None
        num_samples = len(X)
        self.error_matrix = np.ones((self.leadtime_k, len(self.ensemble_members), int(num_samples/self.leadtime_k)))
        for ens_id, ensemble_member in enumerate(self.ensemble_members):
            # for each ensemble type, use the idx-th member of this type to predict the data is has not been trained on
            for idx, (_, test_index) in enumerate(self.train_test_indexes):
                for sample_id in range(len(test_index)):
                    pred = ensemble_member[idx].predict(X[test_index][sample_id].reshape(1, -1))[0]
                    error = self.error_function(
                        np.array(y[test_index][sample_id]), np.array([pred])
                    )
                    # in case of classification, accuracy instead of error is needed
                    if self.type == 'classification':
                        error = 1-error
                    # put the calculated error in the matrix at the associated forecast range
                    self.error_matrix[target_ids[sample_id], ens_id, int((len(test_index) * idx + sample_id)/self.leadtime_k)] = error
        self.error_matrix = self.error_matrix.transpose()
        # shape: [num_sample_creations, num_ensemble_types, leadtime_k]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the ensemble members to the given input data
        
        - either, create multiple objects of each ensemble member, and fit each to a subset of the input.
        - or: Use the pretrained ensemble members
        - use the ensemble members to calculate the global error
        - create one object for each ensemble type, and fit it to the whole dataset
        - create one predictor for each ensemble type, and fit it to predict the local error for a given input

        Parameters
        ----------
        X : numpy.array
            shape: [samples, features]
        y : numpy.array
            shape: [samples, 1]

        Returns
        -------

        """
        #ToDo: X as a matrix
        #[[1,2,3]->TS1]
        #[1, 2, 3]->TS2]
        #[1, 2, 3]->TS3]
        #[1, 2, 3]->TS4]
        #[1, 2, 3]->TS5]
        #[1, 2, 3]->TS1]
        #Step 2: Pass array [1,2,...,24,1,2,...]
        X_feat = X
        ts_idx = np.zeros([X_feat.shape[0], 1]).astype(int)
        if X.shape[1] != 1 and self.leadtime_k > 1:
            ts_idx = X[:, -1].astype(int)
            X_feat = X_feat[:, :-1]
        else:
            ts_idx = np.arange(0, self.leadtime_k).reshape(-1, 1).repeat(int(X_feat.shape[0] / self.leadtime_k),
                                                                axis=1).transpose().reshape(-1, 1)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        num_samples = len(X)

        kf = KFold(n_splits=self.n_cv_out_of_sample_error)
        self.train_test_indexes = [
            (train_index, test_index) for train_index, test_index in kf.split(X)
        ]


        if not self.should_fit:
            self._create_error_matrix(X_feat, ts_idx, y)
            self._get_global_error()
            # since all ensemble members of one type are duplicates, simply select the first one
            self.ensemble_members = [ensemble[0] for ensemble in self.ensemble_members]
            
        else:
            self.ensemble_members = []
            # fit ensemble members to create out of sample errors
            self._fit_out_of_sample_ensembles(X_feat, y)

            # same as the local error
            self._create_error_matrix(X_feat, ts_idx, y)
            self._get_global_error()
            # refit ensemble members on complete data
            self._fit_ensembles_for_prediction(X_feat, y)

        # fit predictor to local error
        # Only use every leadtime_k-th element, as X-values are the same for a number of steps
        self._fit_local_error_forecast(X_feat[::self.leadtime_k])

        # calculate the time-based error matrix
        self._calculate_time_error()

    def _fit_local_error_forecast(self, X: np.ndarray):
        self.local_error_forecaster = []

        for id_ens in range(len(self.ensemble_members)):
            self.local_error_forecaster.append(
                self.model_forecast_local_error().fit(X, self.error_matrix[:, id_ens, 0])
            )

    def _calculate_time_error(self):
        # as the number of timesteps and the values are always constant,
        # there is no need to train a predictor
        self.time_error_forecaster = []
        self.time_error_matrix = self.error_matrix.mean(0).transpose()

    def _fit_out_of_sample_ensembles(self, X: np.ndarray, y: np.ndarray):
        if self.leadtime_k != 1:
            self.t = np.arange(0, self.leadtime_k, 1)
        else:
            self.t = np.array([0])
            # ravel to remove add. dimension unnecessary for fitting
            y = y.ravel()
        self.t = self.t.reshape(-1, 1)
        for i, ensemble in enumerate(self.ensembles_types):
            cv_ensembles = []
            for train_index, _ in self.train_test_indexes:
                model = self._fit_one_model(ensemble, i, X[train_index], y[train_index])
                cv_ensembles.append(model)
            self.ensemble_members.append(cv_ensembles)

    def _fit_ensembles_for_prediction(self, X: np.ndarray, y: np.ndarray):
        if self.leadtime_k == 1:
            # ravel to remove add. dimension unnecessary for fitting
            y = y.ravel()
        self.ensemble_members = []
        for i, ensemble in enumerate(self.ensembles_types):
            model = self._fit_one_model(ensemble, i, X, y)
            self.ensemble_members.append(model)
    
    def _fit_one_model(self, ensemble, i: int, X: np.ndarray, y: np.ndarray):
        model = ensemble()
        # if predict_proba has been enabled, pass this parameter to all models to enable it
        if hasattr(model, 'probability'):
            model.probability = self.probability
        model = self._assign_params(i, model)
        model.fit(X, y)
        return model

    def _pred_all_ensembles(self, X: np.ndarray):
        predictions = np.zeros((int(len(X)/self.leadtime_k), self.leadtime_k, len(self.ensemble_members)))
        for id_em, ensemble_member in enumerate(self.ensemble_members):
            for i, sample in enumerate(X):
                pred = ensemble_member.predict(sample.reshape(1, -1))
                index_t = i%self.leadtime_k
                index_s = int(i/self.leadtime_k)
                predictions[index_s, index_t, id_em] = pred
            
            # Deprecated, for now
            # for i, sample in enumerate(X[::self.leadtime_k]):
            #     print(i)
            #     print(sample)
            #     pred = ensemble_member.predict(sample.reshape(1, -1))
            #     if len(pred.shape) == 1 or pred.shape[1] == 1:
            #         pred = pred.reshape(-1)
            #         pred = np.repeat(pred[:, np.newaxis], self.leadtime_k, axis=1)
            #     predictions[i, :, id_em] = pred

        return predictions

    def _pred_local_error(self, X: np.ndarray):
        local_errors = np.zeros((len(X), len(self.ensemble_members)))
        for id_em, local_error_member in enumerate(self.local_error_forecaster):
            local_errors[:, id_em] = local_error_member.predict(X)

        return local_errors

    def _calc_final_weighting(self, X: np.ndarray):
        # normalize the global error of all ensemble members.
        # apply the softgating function to select the linearity.
        # shape: [1, len(ensemble_members)]
        normalized_global_error = self._normalize_weighting(self.global_errors)
        normalized_global_error = utils.soft_gating_formular(normalized_global_error, self.eta[0])
        # predict the local error of all ensemble members for each input separately.
        # apply the softgating function to select the linearity.
        # shape: [num_sample_creations, len(ensemble_members)]
        self.local_errors = 1 / (self._pred_local_error(X[::self.leadtime_k]) + self.eps)
        self.local_errors = utils.soft_gating_formular(self.local_errors, self.eta[1])
        # calculate the time errors based on the time error matrix.
        # apply the softgating function to select the linearity.
        # shape: [leadtime_k, len(ensemble_members)]
        self.time_errors = 1 / (self.time_error_matrix + self.eps)
        self.time_errors = utils.soft_gating_formular(self.time_errors, self.eta[2])
        # multiply both error types.
        # this weights the local errors, i.e. the error of a given input, with the average error of the ensemble member
        # over the whole input space it has been trained on.
        # this results in selecting the best best ensemble member for a given input (local space), with respect to
        # its overall performance.
        
        combined_weighting = self.local_errors * normalized_global_error
        
        # add an additional dimension to both matrices to multiply them properly
        # this substitutes a multiplication of all forecast ranges iteratively
        time_errors_reshaped = np.expand_dims(self.time_errors, 0)
        combined_weighting_reshaped = np.expand_dims(combined_weighting, 1)
        final_weighting = combined_weighting_reshaped * time_errors_reshaped
        final_weighting = 1 / final_weighting
        self.final_weighting = self._normalize_weighting(final_weighting)

    def _weight_forecasts(self, X, predictions):
        self._calc_final_weighting(X)
        weighted_predictions = (predictions * self.final_weighting).sum(2)
        if self.type == 'classification':
            weighted_predictions = np.round(weighted_predictions)
        return weighted_predictions

    def predict(self, X: np.ndarray):
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
        num_samples = int(X.shape[0]/self.leadtime_k) * self.leadtime_k
        # currently: remove the first elements of X to be able to predict all values in a matrix.
        # ToDo: Enable matrizes with not all entries filled (fit and predict)
        X = X[-num_samples:]
        #X = X[:num_samples]
        X_feat = X
        if X.shape[1] != 1 and self.leadtime_k > 1:
            X_feat = X_feat[:, :-1]
        predictions_ensembles = self._pred_all_ensembles(X_feat)
        predictions = self._weight_forecasts(X_feat, predictions_ensembles)

        return predictions

    def predict_proba(self, X: np.ndarray):
        """
        Predict the probability of each label for each sample, using a weighted combination of all ensemble members
        Parameters
        ----------
        X : np.ndarray
            shape: [samples, features]

        Returns
        -------
        predictions : np.ndarray
            shape: [len(X), num_labels]
            the probability of each label for each sample
        """
        if self.type != 'classification':
            return
        self._calc_final_weighting(X)

        predictions = []
        for ensemble_member in self.ensemble_members:
            predictions.append(ensemble_member.predict_proba(X))
        predictions = np.array(predictions)
        predictions = predictions.swapaxes(0, 2)
        predictions = (np.expand_dims(predictions, 2) * np.expand_dims(self.final_weighting, 0)).sum(axis=3)
        predictions = predictions.squeeze(2).swapaxes(0, 1)

        return predictions