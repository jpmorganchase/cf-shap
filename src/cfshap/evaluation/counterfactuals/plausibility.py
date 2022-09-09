from typing import Union

import numpy as np
from sklearn.neighbors import NearestNeighbors

from emutils.utils import keydefaultdict
from emutils.random import np_sample
from emutils.geometry.metrics import get_metric_name_and_params

from ...base import CounterfactualEvaluationScorer, BaseCounterfactualEvaluationScorer


class BaseNNDistance(CounterfactualEvaluationScorer, BaseCounterfactualEvaluationScorer):
    def __init__(
            self,
            scaler,
            X,
            distance,
            n_neighbors: Union[int, float] = 5,
            max_samples=int(1e10),
            random_state=2021,
            **distance_params,
    ):

        self._scaler = scaler
        self._metric, self._metric_params = get_metric_name_and_params(distance, **distance_params)
        self._n_neighbors = n_neighbors
        self._max_samples = max_samples
        self.random_state = random_state
        self.data = X

    @property
    def data(self):
        return self._data


class NNDistance(BaseNNDistance):
    """
        Plausibility metrics for counterfactuals.

        It computes the (average) distance from the K Nearest Neighbours of the point

        Note it considers all the points (regardless of their class as neighbours).
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @BaseNNDistance.data.setter
    def data(self, data):

        # Sample
        self._data = np_sample(np.asarray(data), random_state=self.random_state, n=self._max_samples, safe=True)

        if isinstance(self._n_neighbors, int) and self._n_neighbors >= 1:
            n_neighbors = min(self._n_neighbors, len(self._data))
        elif isinstance(self._n_neighbors, float) and self._n_neighbors <= 1.0 and self._n_neighbors > 0.0:
            n_neighbors = int(max(1, round(len(self._data) * self._n_neighbors)))
        else:
            raise ValueError(
                'Invalid n_neighbors, it must be the number of neighbors (int) or the fraction of the dataset (float)')

        # We will be searching neighbors
        self._nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=self._metric,
            p=self._metric_params['p'] if 'p' in self._metric_params else 2,
            metric_params=self._metric_params,
        ).fit(self._scaler.transform(self._data))

    def score(self, X: np.ndarray):
        X = np.asarray(X)

        avg_dist = np.full(X.shape[0], np.nan)
        nan_mask = np.any(np.isnan(X), axis=1)

        if (~nan_mask).sum() > 0:
            neigh_dist, _ = self._nn.kneighbors(self._scaler.transform(X[~nan_mask]), n_neighbors=None)
            neigh_dist = neigh_dist.mean(axis=1)
            avg_dist[~nan_mask] = neigh_dist

        return avg_dist


class yNNDistance(BaseNNDistance):
    """
        Plausibility metrics for counterfactuals.

        It computes the (average) distance from the K Nearest COUNTERFACTUAL Neighbours of the point

        Contrary to NNDistance, it considers as neighbours only points that have a different class.
    """
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    @BaseNNDistance.data.setter
    def data(self, data):
        self._raw_data = np.asarray(data)

        # Sample
        self._raw_data = np_sample(self._raw_data, random_state=self.random_state, n=self._max_samples, safe=True)

        # Predict
        self._preds = self.model.predict(self._raw_data)

        if isinstance(self._n_neighbors, int) and self._n_neighbors >= 1:
            n_neighbors = keydefaultdict(lambda pred: min(self._n_neighbors, len(self._data[pred])))
        elif isinstance(self._n_neighbors, float) and self._n_neighbors <= 1.0 and self._n_neighbors > 0.0:
            n_neighbors = keydefaultdict(lambda pred: int(max(1, round(len(self._data[pred]) * self._n_neighbors))))
        else:
            raise ValueError(
                'Invalid n_neighbors, it must be the number of neighbors (int) or the fraction of the dataset (float)')

        # We will be searching neighbors of a different class
        self._data = keydefaultdict(lambda pred: self._raw_data[self._preds == pred])

        self._nn = keydefaultdict(lambda pred: NearestNeighbors(
            n_neighbors=n_neighbors[pred],
            metric=self._metric,
            p=self._metric_params['p'] if 'p' in self._metric_params else 2,
            metric_params=self._metric_params,
        ).fit(self._scaler.transform(self._data[pred])))

    def score(self, X: np.ndarray):
        X = np.asarray(X)

        nan_mask = np.any(np.isnan(X), axis=1)

        preds = self.model.predict(X[~nan_mask])
        preds_indices = {pred: np.argwhere(preds == pred).flatten() for pred in np.unique(preds)}
        avg_dist_ = np.full(preds.shape[0], np.nan)

        for pred, indices in preds_indices.items():
            neigh_dist, _ = self._nn[pred].kneighbors(self._scaler.transform(X[~nan_mask][indices]), n_neighbors=None)
            neigh_dist = neigh_dist.mean(axis=1)
            avg_dist_[preds_indices[pred]] = neigh_dist

        avg_dist = np.full(X.shape[0], np.nan)
        avg_dist[~nan_mask] = avg_dist_
        return avg_dist
