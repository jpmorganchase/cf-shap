"""
    Author: Emanuele Albini

    Feature Trend Estimators
"""

import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['TrendEstimator', 'DummyTrendEstimator']


class DummyTrendEstimator(BaseEstimator):
    def __init__(self, trends):
        self.trends = trends

    def predict(self, x=None, Y=None):
        return np.asarray(self.trends)


class TrendEstimator(BaseEstimator):
    def __init__(self, strategy='mean'):
        self.strategy = strategy

    @staticmethod
    def __step_function(x, y):
        return 2 * np.heaviside(y - x, 0) - 1

    def __preprocess(self, x, Y):
        if not isinstance(x, np.ndarray):
            raise ValueError('Must pass a NumPy array.')

        if not isinstance(Y, np.ndarray):
            raise ValueError('Must pass a NumPy array.')

        if len(Y.shape) != 2:
            raise ValueError('Y must be a 2D matrix.')

        if len(x.shape) != 1:
            x = x.flatten()

        if x.shape[0] != Y.shape[1]:
            raise ValueError('x and Y sizes must be coherent.')

        return x, Y

    def __preprocess3D(self, X, YY):
        if not isinstance(X, np.ndarray):
            raise ValueError('Must pass a NumPy array.')

        if not isinstance(YY[0], np.ndarray):
            raise ValueError('Must pass a list of NumPy array.')

        if len(X.shape) != 2:
            raise ValueError('X must be a 2D matrix.')

        if len(X) != len(YY):
            raise ValueError('X and Y must have the same length.')

        if X.shape[1] != YY[0].shape[1]:
            raise ValueError('X and Y[0] sizes must be coherent.')

        return X, YY

    def __call__(self, x, Y):
        x, Y = self.__preprocess(x, Y)

        if self.strategy == 'mean':
            return self.__step_function(x, Y.mean(axis=0))
        else:
            raise ValueError('Invalid strategy.')

    # NOTE: Duplicated for efficiency on 3D data
    def predict(self, X, YY):
        X, YY = self.__preprocess3D(X, YY)

        if self.strategy == 'mean':
            return np.array([self.__step_function(x, Y.mean(axis=0)) for x, Y in zip(X, YY)])
        else:
            raise ValueError('Invalid strategy.')
