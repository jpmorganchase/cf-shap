"""
    Author: Emanuele Albini

    Background data computed as samples with different label or prediction.
"""

from abc import ABC, abstractclassmethod
import numpy as np

from ..base import (
    Model,
    BaseBackgroundGenerator,
    ListOf2DArrays,
)

__all__ = ['DifferentLabelBackgroundGenerator', 'DifferentPredictionBackgroundGenerator']


class BaseDifferent_BackgrondGenerator(BaseBackgroundGenerator):
    def __init__(self, model, max_samples=None, random_state=None):
        super().__init__(model, None, random_state)

        self.max_samples = max_samples

    @property
    def data(self):
        return self._data

    def _save_data(self, data):
        data = data.copy()
        data = self.sample(data, self.max_samples)
        data.flags.writeable = False
        return data

    def _set_data(self, X, y):
        self._data = X
        self._data_per_class = {j: self._save_data(X[y != j]) for j in np.unique(y)}

    def get_backgrounds(self, X: np.ndarray) -> ListOf2DArrays:
        X = self.preprocess(X)
        y = self.model.predict(X)
        return [self._data_per_class[j] for j in y]


class DifferentPredictionBackgroundGenerator(BaseDifferent_BackgrondGenerator):
    """The background dataset will be constituted of the points with a different PREDICTION to that of the query instance.
    """
    def __init__(self, model: Model, data: np.ndarray, **kwargs):
        super().__init__(model, **kwargs)

        # Based on the prediction
        data = self.preprocess(data)
        self._set_data(data, self.model.predict(data))


class DifferentLabelBackgroundGenerator(BaseDifferent_BackgrondGenerator):
    """The background dataset will be constituted of the points with a different LABEL to that of the query instance.
    """
    def __init__(self, model: Model, X: np.ndarray, y: np.ndarray, **kwargs):
        super().__init__(model, **kwargs)

        # Based on the label
        X = self.preprocess(X)
        self._set_data(X, y)
