"""
    Author: Emanuele Albini

    Implementation of K-Nearest Neighbours Counterfactuals
"""

__all__ = ['KNNCounterfactuals']

from typing import Union

import numpy as np
from sklearn.neighbors import NearestNeighbors
from emutils.utils import keydefaultdict

from ..base import (
    BaseMultipleCounterfactualMethod,
    Model,
    Scaler,
    ListOf2DArrays,
)

from emutils.geometry.metrics import get_metric_name_and_params


class KNNCounterfactuals(BaseMultipleCounterfactualMethod):
    """Returns the K Nearest Neighbours of the query instance with a different prediction.

    """
    def __init__(
            self,
            model: Model,
            scaler: Union[None, Scaler],
            X: np.ndarray,
            nb_diverse_counterfactuals: Union[None, int, float] = None,
            n_neighbors: Union[None, int, float] = None,
            distance: str = None,
            max_samples: int = int(1e10),
            random_state: int = 2021,
            verbose: int = 0,
            **distance_params,
    ):
        """

        Args:
            model (Model): The model.
            scaler (Union[None, Scaler]): The scaler for the data.
            X (np.ndarray): The background dataset.
            nb_diverse_counterfactuals (Union[None, int, float], optional): Number of counterfactuals to generate. Defaults to None.
            n_neighbors (Union[None, int, float], optional): Number of neighbours to generate. Defaults to None.
                Note that this is an alias for nb_diverse_counterfactuals in this class.
            distance (str, optional): The distance metric to use for K-NN. Defaults to None.
            max_samples (int, optional): Number of samples of the background to use at most. Defaults to int(1e10).
            random_state (int, optional): Random seed. Defaults to 2021.
            verbose (int, optional): Level of verbosity. Defaults to 0.
            **distance_params: Additional parameters for the distance metric
        """

        assert nb_diverse_counterfactuals is not None or n_neighbors is not None, 'nb_diverse_counterfactuals or n_neighbors must be set.'

        super().__init__(model, scaler, random_state)

        self._metric, self._metric_params = get_metric_name_and_params(distance, **distance_params)
        self.__nb_diverse_counterfactuals = nb_diverse_counterfactuals
        self.__n_neighbors = n_neighbors
        self.max_samples = max_samples

        self.data = X
        self.verbose = verbose

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._raw_data = self.preprocess(data)
        if self.max_samples < len(self._raw_data):
            self._raw_data = self.sample(self._raw_data, n=self.max_samples)
        self._preds = self.model.predict(self._raw_data)

        # In the base class this two information are equivalent
        if self.__n_neighbors is None:
            self.__n_neighbors = self.__nb_diverse_counterfactuals
        if self.__nb_diverse_counterfactuals is None:
            self.__nb_diverse_counterfactuals = self.__n_neighbors

        def get_nb_of_items(nb):
            if np.isinf(nb):
                return keydefaultdict(lambda pred: self._data[pred].shape[0])
            elif isinstance(nb, int) and nb >= 1:
                return keydefaultdict(lambda pred: min(nb, self._data[pred].shape[0]))
            elif isinstance(nb, float) and nb <= 1.0 and nb > 0.0:
                return keydefaultdict(lambda pred: int(max(1, round(len(self._data[pred]) * nb))))
            else:
                raise ValueError(
                    'Invalid n_neighbors, it must be the number of neighbors (int) or the fraction of the dataset (float)'
                )

        self._n_neighbors = get_nb_of_items(self.__n_neighbors)
        self._nb_diverse_counterfactuals = get_nb_of_items(self.__nb_diverse_counterfactuals)

        # We will be searching neighbors of a different class
        self._data = keydefaultdict(lambda pred: self._raw_data[self._preds != pred])

        self._nn = keydefaultdict(lambda pred: NearestNeighbors(
            n_neighbors=self._n_neighbors[pred],
            metric=self._metric,
            p=self._metric_params['p'] if 'p' in self._metric_params else 2,
            metric_params=self._metric_params,
        ).fit(self.scale(self._data[pred])))

    def get_counterfactuals(self, X: np.ndarray) -> np.ndarray:
        """Generate the closest counterfactual for each query instance"""

        # Pre-process
        X = self.preprocess(X)

        preds = self.model.predict(X)
        preds_indices = {pred: np.argwhere(preds == pred).flatten() for pred in np.unique(preds)}

        X_counter = np.zeros_like(X)

        for pred, indices in preds_indices.items():
            _, neighbors_indices = self._nn[pred].kneighbors(self.scale(X), n_neighbors=1)
            X_counter[indices] = self._data[pred][neighbors_indices.flatten()]

        # Post-process
        X_counter = self.postprocess(X, X_counter, invalid='raise')

        return X_counter

    def get_multiple_counterfactuals(self, X: np.ndarray) -> ListOf2DArrays:
        """Generate the multiple closest counterfactuals for each query instance"""

        # Pre-condition
        assert self.__n_neighbors == self.__nb_diverse_counterfactuals, f'n_neighbors and nb_diverse_counterfactuals are set to different values ({self.__n_neighbors} != {self.__nb_diverse_counterfactuals}). When both are set they must be set to the same value.'

        # Pre-process
        X = self.preprocess(X)

        preds = self.model.predict(X)
        preds_indices = {pred: np.argwhere(preds == pred).flatten() for pred in np.unique(preds)}

        X_counter = [
            np.full((self._nb_diverse_counterfactuals[preds[i]], X.shape[1]), np.nan) for i in range(X.shape[0])
        ]

        for pred, indices in preds_indices.items():
            _, neighbors_indices = self._nn[pred].kneighbors(self.scale(X[indices]), n_neighbors=None)
            counters = self._data[pred][neighbors_indices.flatten()].reshape(len(indices),
                                                                             self._nb_diverse_counterfactuals[pred], -1)
            for e, i in enumerate(indices):
                # We use :counters[e].shape[0] so it raises an exception if shape are not coherent.
                X_counter[i][:counters[e].shape[0]] = counters[e]

        # Post-process
        X_counter = self.diverse_postprocess(X, X_counter, invalid='raise')

        return X_counter